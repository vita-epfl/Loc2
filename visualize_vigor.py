import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import ast
import configparser
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from dataloaders.dataloader_vigor_with_depth import VIGORDataset, transform_grd, transform_sat
from models.modules import DinoExtractor
from models.utils import e2eProbabilisticProcrustesSolver, weighted_procrustes_2d_with_scale
from models.vigor_matcher import VigorCrossViewMatcher


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.ini"
FIGURES_ROOT = PROJECT_ROOT / "figures"


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize VIGOR matches and depth-lifted overlays.")
    parser.add_argument("--area", type=str, default="samearea", choices=("samearea", "crossarea"))
    parser.add_argument("--dataset_split", type=str, default="test", choices=("train", "test"))
    parser.add_argument("--random_orientation", type=float, default=0.0)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=str(FIGURES_ROOT))
    parser.add_argument("--index", "--indices", dest="indices", type=int, nargs="+", required=True)
    parser.add_argument("--top_k", type=int, default=50, help="Number of top matches to draw.")
    parser.add_argument("--num_samples_matches", type=int, default=None)
    parser.add_argument("--max_depth", type=float, default=35.0)
    parser.add_argument("--initial_scale", type=float, default=0.1)
    parser.add_argument("--point_stride", type=int, default=1, help="Stride for overlay points; 1 keeps all pixels.")
    parser.add_argument("--point_size", type=float, default=0.2)
    parser.add_argument("--valid_depth_ratio", type=float, default=1.0)
    parser.add_argument("--overlay_projection", choices=("roll", "pose"), default="roll")
    parser.add_argument("--ransac", choices=("True", "False"), default="False")
    parser.add_argument("--th_soft_inlier", type=float, default=5.0)
    parser.add_argument("--th_inlier", type=float, default=2.5)
    parser.add_argument("--num_samples_matches_ransac", type=int, default=8192)
    parser.add_argument("--num_corr_2d_2d", type=int, default=3)
    parser.add_argument("--it_matches", type=int, default=20)
    parser.add_argument("--it_RANSAC_procrustes", type=int, default=100)
    parser.add_argument("--num_ref_steps", type=int, default=4)
    return parser.parse_args()


def load_config():
    config = configparser.ConfigParser()
    if not config.read(CONFIG_PATH):
        raise FileNotFoundError(f"Could not read config file at {CONFIG_PATH}")
    return config


def set_seeds(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True, warn_only=True)


def resolve_dataset_root(args, config):
    if args.dataset_root is not None:
        return args.dataset_root
    if config.has_option("VIGOR", "dataset_root"):
        return config.get("VIGOR", "dataset_root")
    return config.get("VIGOR", "scitas_dataset_root")


def create_metric_grid(grid_size, resolution, batch_size, device):
    axis = torch.linspace(-grid_size / 2, grid_size / 2, resolution, device=device)
    metric_x, metric_y = torch.meshgrid(axis, axis, indexing="ij")
    metric_coord = torch.stack((metric_x.reshape(-1), metric_y.reshape(-1)), dim=-1)
    return metric_coord.unsqueeze(0).repeat(batch_size, 1, 1)


def create_spherical_grids(ground_image_size, batch_size, device):
    phi = torch.linspace(0, 2 * np.pi, int(ground_image_size[1] / 14), device=device)
    theta = torch.linspace(0, np.pi, int(ground_image_size[0] / 14), device=device)
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")
    theta = theta.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    phi = phi.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    return theta, phi


def tensor_image_to_numpy(image):
    image_np = image.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image_np, 0.0, 1.0)


def build_ground_bev_coordinates(depth, grd_feature_shape, theta, phi, max_depth):
    batch_size = depth.shape[0]
    depth_downsampled = F.interpolate(depth, size=grd_feature_shape, mode="nearest")
    mask = (depth_downsampled < max_depth).flatten(1)

    grd_x = depth_downsampled * torch.sin(theta[:batch_size]) * torch.cos(phi[:batch_size])
    grd_y = depth_downsampled * torch.sin(theta[:batch_size]) * (-torch.sin(phi[:batch_size]))
    grd_z = depth_downsampled * torch.cos(theta[:batch_size])
    metric_coord_grd = torch.cat((grd_x.flatten(2), grd_y.flatten(2), grd_z.flatten(2)), 1).permute(0, 2, 1)
    return metric_coord_grd[:, :, :2], mask


def sample_pose_correspondences(matching_score, metric_coord_sat, bev_coord_grd, num_samples_matches):
    batch_size, _, num_kpts_grd = matching_score.shape
    matches_row = matching_score.flatten(1)
    sampled_matching_idx = torch.multinomial(matches_row, num_samples_matches)
    sat_indices_sampled = torch.div(sampled_matching_idx, num_kpts_grd, rounding_mode="trunc")
    grd_indices_sampled = sampled_matching_idx % num_kpts_grd
    batch_idx = torch.arange(batch_size, device=matching_score.device).view(batch_size, 1).expand(-1, num_samples_matches)
    sat_points = metric_coord_sat[batch_idx, sat_indices_sampled, :]
    grd_points = bev_coord_grd[batch_idx, grd_indices_sampled, :]
    weights = matches_row[batch_idx, sampled_matching_idx]
    return sat_points, grd_points, weights


def estimate_pose(args, matching_score, metric_coord_sat, bev_coord_grd, num_samples_matches):
    if args.ransac == "True":
        pose_solver = e2eProbabilisticProcrustesSolver(
            args.it_RANSAC_procrustes,
            args.it_matches,
            args.num_samples_matches_ransac,
            args.num_corr_2d_2d,
            args.num_ref_steps,
            args.th_inlier,
            args.th_soft_inlier,
            metric_coord_sat,
            bev_coord_grd,
        )
        rotation, translation, scale, _, _ = pose_solver.estimate_pose(matching_score, return_inliers=False)
        return rotation, translation, scale

    sat_points, grd_points, weights = sample_pose_correspondences(
        matching_score,
        metric_coord_sat,
        bev_coord_grd,
        num_samples_matches,
    )
    rotation, translation, scale, _ = weighted_procrustes_2d_with_scale(
        grd_points,
        sat_points,
        use_weights=True,
        use_mask=True,
        w=weights,
    )
    return rotation, translation, scale


def yaw_from_rotation(rotation):
    rotation_np = rotation.detach().cpu().numpy()
    return float(np.degrees(np.arctan2(rotation_np[1, 0], rotation_np[0, 0])))


def yaw_error_degrees(yaw, yaw_gt):
    diff = abs(yaw - yaw_gt)
    return float(min(diff, 360.0 - diff))


def compute_top_match_points(matching_score, grd_shape, sat_shape, grd_feature_shape, sat_bev_res, top_k):
    _, _, num_kpts_grd = matching_score.shape
    _, _, grd_h, grd_w = grd_shape
    _, _, sat_h, sat_w = sat_shape
    grd_feature_h, grd_feature_w = grd_feature_shape

    matches_row = matching_score.flatten(1)
    top_k = min(top_k, matches_row.shape[1])
    top_indices = torch.argsort(matches_row, descending=True)[0, :top_k]
    sat_indices = torch.div(top_indices, num_kpts_grd, rounding_mode="trunc")
    grd_indices = top_indices % num_kpts_grd

    sat_rows = torch.div(sat_indices, sat_bev_res, rounding_mode="trunc").float()
    sat_cols = (sat_indices % sat_bev_res).float()
    grd_rows = torch.div(grd_indices, grd_feature_w, rounding_mode="trunc").float()
    grd_cols = (grd_indices % grd_feature_w).float()

    sat_x = ((sat_cols + 0.5) / sat_bev_res * sat_w).detach().cpu().numpy()
    sat_y = ((sat_rows + 0.5) / sat_bev_res * sat_h).detach().cpu().numpy()
    grd_x = ((grd_cols + 0.5) / grd_feature_w * grd_w).detach().cpu().numpy()
    grd_y = ((grd_rows + 0.5) / grd_feature_h * grd_h).detach().cpu().numpy()

    grd_points = list(zip(grd_x, grd_y))
    sat_points = list(zip(sat_x, sat_y))
    return grd_points, sat_points


def draw_pose_marker(ax, center_x, center_y, yaw, marker, color, label, marker_size, quiver_scale):
    ax.scatter(
        center_x,
        center_y,
        s=marker_size,
        marker=marker,
        facecolor=color,
        label=label,
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    ax.quiver(
        center_x,
        center_y,
        -np.sin(np.radians(yaw)),
        np.cos(np.radians(yaw)),
        color=color,
        linewidths=0.5,
        scale=quiver_scale,
        width=0.012,
        zorder=4,
    )


def save_match_figure(path, grd, sat, grd_points, sat_points, tgt_px, pred_px, yaw_gt, yaw_pred):
    _, _, grd_h, grd_w = grd.shape
    sat_size = sat.shape[-1]
    gap = 10

    grd_to_show = tensor_image_to_numpy(grd[0])
    sat_square = F.interpolate(sat, size=(grd_h, grd_h), mode="bicubic", align_corners=False)
    sat_to_show = tensor_image_to_numpy(sat_square[0])

    combined_image = np.ones((grd_h, grd_w + gap + grd_h, 3), dtype=np.float32)
    combined_image[:, :grd_w, :] = grd_to_show
    combined_image[:, grd_w + gap:, :] = sat_to_show

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.imshow(combined_image)

    for (grd_x, grd_y), (sat_x, sat_y) in zip(grd_points, sat_points):
        sat_x_resized = sat_x / sat_size * grd_h + grd_w + gap
        sat_y_resized = sat_y / sat_size * grd_h
        ax.plot(
            [grd_x, sat_x_resized],
            [grd_y, sat_y_resized],
            marker="o",
            markersize=2,
            color="lime",
            linestyle="-",
            linewidth=0.8,
            alpha=0.8,
            zorder=1,
        )

    tgt_resized = tgt_px / sat_size * grd_h
    pred_resized = pred_px / sat_size * grd_h
    draw_pose_marker(
        ax,
        grd_w + gap + grd_h / 2 + tgt_resized[0, 1],
        grd_h / 2 + tgt_resized[0, 0],
        yaw_gt,
        "^",
        "green",
        "GT",
        160,
        25,
    )
    draw_pose_marker(
        ax,
        grd_w + gap + grd_h / 2 + pred_resized[0, 1],
        grd_h / 2 + pred_resized[0, 0],
        yaw_pred,
        "*",
        "gold",
        "Ours",
        180,
        25,
    )
    ax.legend(loc="upper right", framealpha=0.8, labelcolor="black")
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def save_pose_figure(path, sat, tgt_px, pred_px, yaw_gt, yaw_pred):
    sat_to_show = tensor_image_to_numpy(sat[0])
    sat_size = sat.shape[-1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(sat_to_show)
    draw_pose_marker(
        ax,
        sat_size / 2 + tgt_px[0, 1],
        sat_size / 2 + tgt_px[0, 0],
        yaw_gt,
        "^",
        "green",
        "GT",
        220,
        10,
    )
    draw_pose_marker(
        ax,
        sat_size / 2 + pred_px[0, 1],
        sat_size / 2 + pred_px[0, 0],
        yaw_pred,
        "*",
        "gold",
        "Ours",
        240,
        10,
    )
    ax.legend(loc="upper right", framealpha=0.8, labelcolor="black")
    ax.axis("off")
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def pano_to_point_cloud_bev(grd, depth, max_depth, point_stride, valid_depth_ratio):
    rgb = tensor_image_to_numpy(grd[0])
    depth_np = depth[0, 0].detach().cpu().numpy()

    if point_stride > 1:
        rgb = rgb[::point_stride, ::point_stride]
        depth_np = depth_np[::point_stride, ::point_stride]

    depth_np = np.clip(depth_np, 0.0, max_depth)
    image_h, image_w = depth_np.shape
    phi = np.linspace(0, 2 * np.pi, image_w, dtype=np.float32)
    theta = np.linspace(0, np.pi, image_h, dtype=np.float32)
    theta, phi = np.meshgrid(theta, phi, indexing="ij")

    x = depth_np * np.sin(theta) * np.cos(phi)
    y = -depth_np * np.sin(theta) * np.sin(phi)
    z = depth_np * np.cos(theta)
    xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    colors = rgb.reshape(-1, 3)
    depth_flat = depth_np.reshape(-1)

    depth_limit = max_depth * valid_depth_ratio
    if valid_depth_ratio < 1.0:
        mask = (depth_flat > 0) & (depth_flat < depth_limit)
    else:
        mask = depth_flat > 0
    xyz = xyz[mask]
    colors = colors[mask]
    depth_valid = depth_flat[mask]
    if xyz.shape[0] == 0:
        return np.empty((0, 3)), np.empty((0, 4))

    depth_norm = (depth_valid - depth_valid.min()) / (depth_limit - depth_valid.min() + 1e-8)
    alphas = np.exp(-depth_norm * 4.0)
    alphas[depth_valid >= max_depth - 1e-3] = 0.0
    alphas = np.clip(alphas, 0.0, 1.0)
    colors_with_alpha = np.concatenate([colors, alphas[:, None]], axis=1)
    return xyz, colors_with_alpha


def project_points_with_pose(xyz, colors, sat_size, sat_size_m, rotation, translation):
    if xyz.shape[0] == 0:
        return np.empty((0,)), np.empty((0,)), colors

    rotation_np = rotation.detach().cpu().numpy()
    translation_np = translation.detach().cpu().numpy()
    if translation_np.ndim > 1:
        translation_np = translation_np[0]
    sat_metric = xyz[:, :2] @ rotation_np.T + translation_np

    scale_pixel = sat_size / sat_size_m
    x_img = sat_size / 2 + sat_metric[:, 1] * scale_pixel
    y_img = sat_size / 2 + sat_metric[:, 0] * scale_pixel
    valid_mask = (x_img >= 0) & (x_img < sat_size) & (y_img >= 0) & (y_img < sat_size)
    return x_img[valid_mask], y_img[valid_mask], colors[valid_mask]


def project_points_with_roll(xyz, colors, sat_size, sat_size_m, pred_px):
    if xyz.shape[0] == 0:
        return np.empty((0,)), np.empty((0,)), colors

    scale_pixel = sat_size / sat_size_m
    x_img = sat_size / 2 + pred_px[0, 1] + xyz[:, 1] * scale_pixel
    y_img = sat_size / 2 + pred_px[0, 0] + xyz[:, 0] * scale_pixel
    valid_mask = (x_img >= 0) & (x_img < sat_size) & (y_img >= 0) & (y_img < sat_size)
    return x_img[valid_mask], y_img[valid_mask], colors[valid_mask]


def project_overlay_points(args, grd, depth, max_depth, sat_size, sat_size_m, pred_px, yaw_pred, rotation, translation):
    if args.overlay_projection == "roll":
        roll_pixels = int(-yaw_pred / 360.0 * grd.shape[-1])
        grd = torch.roll(grd, roll_pixels, dims=3)
        depth = torch.roll(depth, roll_pixels, dims=3)

    xyz, colors = pano_to_point_cloud_bev(
        grd,
        depth,
        max_depth,
        max(1, args.point_stride),
        args.valid_depth_ratio,
    )

    if args.overlay_projection == "roll":
        return project_points_with_roll(xyz, colors, sat_size, sat_size_m, pred_px)
    return project_points_with_pose(xyz, colors, sat_size, sat_size_m, rotation, translation)


def save_overlay_figure(path, sat, x_img, y_img, colors, pred_px, yaw_pred, point_size, show_pose):
    sat_to_show = tensor_image_to_numpy(sat[0])
    sat_size = sat.shape[-1]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(sat_to_show, alpha=0.4)
    if len(x_img):
        ax.scatter(x_img, y_img, c=colors, s=point_size, linewidths=0)
    if show_pose:
        draw_pose_marker(
            ax,
            sat_size / 2 + pred_px[0, 1],
            sat_size / 2 + pred_px[0, 0],
            yaw_pred,
            "*",
            "gold",
            "Ours",
            300,
            10,
        )
    ax.axis("off")
    ax.set_xlim([0, sat_size])
    ax.set_ylim([sat_size, 0])
    ax.set_aspect("equal")
    fig.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def visualize_index(args, dataset, index, models, config_values, device, output_dir):
    feature_extractor, matcher = models
    sat_bev_res, num_samples_matches, ground_image_size = config_values

    sample = dataset[index]
    if sample is None:
        print(f"Skipping index {index}: dataset returned None")
        return

    grd, depth, sat, tgt, rotation_gt, city, resolution = sample
    grd = grd.unsqueeze(0).to(device)
    depth = torch.clamp(depth.unsqueeze(0).to(device), 0, args.max_depth)
    sat = sat.unsqueeze(0).to(device)
    tgt = tgt.unsqueeze(0).to(device)
    rotation_gt = rotation_gt.unsqueeze(0).to(device)

    sat_size = sat.shape[-1]
    sat_size_m = sat_size * float(resolution)
    metric_coord_sat = create_metric_grid(sat_size_m, sat_bev_res, 1, device)
    theta, phi = create_spherical_grids(ground_image_size, 1, device)

    grd_feature = feature_extractor(grd)
    sat_feature = feature_extractor(sat)

    depth_for_matching = depth * args.initial_scale
    bev_coord_grd, mask = build_ground_bev_coordinates(
        depth_for_matching,
        grd_feature.shape[-2:],
        theta,
        phi,
        args.max_depth * args.initial_scale,
    )
    matching_score, _ = matcher(grd_feature, sat_feature, mask)

    rotation_pred, translation_pred_m, scale = estimate_pose(
        args,
        matching_score,
        metric_coord_sat,
        bev_coord_grd,
        num_samples_matches,
    )
    if translation_pred_m is None:
        print(f"Skipping index {index}: singular transformation matrix")
        return
    valid_pose = torch.isfinite(translation_pred_m).all() & torch.isfinite(rotation_pred).all() & torch.isfinite(scale).all()
    if not bool(valid_pose.item()):
        print(f"Skipping index {index}: invalid transformation estimate")
        return

    pred_px = (translation_pred_m / sat_size_m * sat_size)[0].detach().cpu().numpy()
    tgt_px = tgt[0].detach().cpu().numpy()
    yaw_pred = yaw_from_rotation(rotation_pred[0])
    yaw_gt = yaw_from_rotation(rotation_gt[0])
    yaw_error = yaw_error_degrees(yaw_pred, yaw_gt)
    translation_error_m = float(torch.norm(translation_pred_m[0] - tgt[0] * float(resolution), dim=-1).item())
    scale_value = float(scale[0, 0, 0].item())

    grd_points, sat_points = compute_top_match_points(
        matching_score,
        grd.shape,
        sat.shape,
        grd_feature.shape[-2:],
        sat_bev_res,
        args.top_k,
    )

    prefix = f"vigor_{args.area}_{args.dataset_split}_idx{index}"
    match_path = output_dir / f"{prefix}_matches.png"
    pose_path = output_dir / f"{prefix}_pose.png"
    overlay_initial_path = output_dir / f"{prefix}_overlay_initial.png"
    overlay_recovered_path = output_dir / f"{prefix}_overlay_recovered.png"

    save_match_figure(match_path, grd, sat, grd_points, sat_points, tgt_px, pred_px, yaw_gt, yaw_pred)
    save_pose_figure(pose_path, sat, tgt_px, pred_px, yaw_gt, yaw_pred)

    x_img, y_img, colors = project_overlay_points(
        args,
        grd,
        depth_for_matching,
        args.max_depth * args.initial_scale,
        sat_size,
        sat_size_m,
        pred_px,
        yaw_pred,
        rotation_pred[0],
        translation_pred_m[0],
    )
    save_overlay_figure(
        overlay_initial_path,
        sat,
        x_img,
        y_img,
        colors,
        pred_px,
        yaw_pred,
        args.point_size,
        show_pose=False,
    )

    if scale_value > 0:
        depth_recovered = depth_for_matching * scale
        x_img, y_img, colors = project_overlay_points(
            args,
            grd,
            depth_recovered,
            args.max_depth * args.initial_scale * scale_value,
            sat_size,
            sat_size_m,
            pred_px,
            yaw_pred,
            rotation_pred[0],
            translation_pred_m[0],
        )
        save_overlay_figure(
            overlay_recovered_path,
            sat,
            x_img,
            y_img,
            colors,
            pred_px,
            yaw_pred,
            args.point_size,
            show_pose=True,
        )
    else:
        overlay_recovered_path = None

    print(
        f"Index {index} ({city}): translation={translation_error_m:.3f} m, "
        f"yaw={yaw_error:.3f} deg, scale={scale_value:.4f}"
    )
    print(f"  saved {match_path}")
    print(f"  saved {pose_path}")
    print(f"  saved {overlay_initial_path}")
    if overlay_recovered_path is not None:
        print(f"  saved {overlay_recovered_path}")


def main():
    args = parse_args()
    config = load_config()
    set_seeds(config.getint("RandomSeed", "seed"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = resolve_dataset_root(args, config)
    dataset = VIGORDataset(
        root=dataset_root,
        split=args.area,
        train=args.dataset_split == "train",
        transform=(transform_grd, transform_sat),
        random_orientation=args.random_orientation,
    )

    sat_bev_res = config.getint("Model", "sat_bev_res")
    num_samples_matches = (
        args.num_samples_matches
        if args.num_samples_matches is not None
        else config.getint("Matching", "num_samples_matches")
    )
    ground_image_size = ast.literal_eval(config.get("VIGOR", "ground_image_size"))

    torch.cuda.empty_cache()
    feature_extractor = DinoExtractor().to(device)
    feature_extractor.eval()
    matcher = VigorCrossViewMatcher(device, sat_bev_res=sat_bev_res, embed_dim=1024).to(device)
    matcher.load_state_dict(torch.load(args.model_path, map_location=device))
    matcher.eval()

    models = (feature_extractor, matcher)
    config_values = (sat_bev_res, num_samples_matches, ground_image_size)
    with torch.no_grad():
        for index in args.indices:
            if index < 0 or index >= len(dataset):
                print(f"Skipping index {index}: valid range is [0, {len(dataset) - 1}]")
                continue
            visualize_index(args, dataset, index, models, config_values, device, output_dir)


if __name__ == "__main__":
    main()
