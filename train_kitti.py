import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import configparser
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataloaders.dataloader_kitti_with_depth import (
    GrdImg_H,
    GrdImg_W,
    SatGrdDataset,
    SatGrdDatasetTest,
    grdimage_transform,
    satmap_transform,
)
from models.kitti_matcher import KittiCrossViewMatcher
from models.loss import (
    compute_infonce_loss_match_all_with_scale_select_negatives,
    loss_bev_space,
)
from models.modules import DinoExtractor
from models.utils import weighted_procrustes_2d_with_scale


NUM_WORKERS = 4
NUM_EPOCHS = 100
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.ini"
SPLITS_DIR = PROJECT_ROOT / "KITTI_splits"
CHECKPOINT_ROOT = PROJECT_ROOT.parent / "checkpoints"
RESULTS_ROOT = PROJECT_ROOT.parent / "results"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--learning_rate", type=float, default=1e-4)
    parser.add_argument("-b", "--batch_size", type=int, default=224)
    parser.add_argument("--rotation_range", type=float, default=10.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--epoch_to_resume", type=int, default=0)
    parser.add_argument("--loss_grid_size", type=float, default=5.0)
    parser.add_argument("--max_depth", type=float, default=40.0)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--negative_distance_threshold", type=float, default=1.0)
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


def resolve_runtime_settings(config):
    if config.has_option("KITTI", "dataset_root"):
        dataset_root = config.get("KITTI", "dataset_root")
    else:
        dataset_root = config.get("KITTI", "scitas_dataset_root")
    settings = {
        "dataset_root": dataset_root,
        "shift_range_lat": config.getfloat("KITTI", "shift_range_lat"),
        "shift_range_lon": config.getfloat("KITTI", "shift_range_lon"),
        "grid_size_h": config.getfloat("KITTI", "grid_size_h"),
        "sat_bev_res": config.getint("Model", "sat_bev_res"),
        "num_samples_matches": config.getint("Matching", "num_samples_matches"),
        "num_virtual_point": config.getint("Loss", "num_virtual_point"),
    }
    return settings


def create_metric_grid(grid_size, resolution, batch_size, device):
    axis = torch.linspace(-grid_size / 2, grid_size / 2, resolution, device=device)
    metric_x, metric_y = torch.meshgrid(axis, axis, indexing="ij")
    metric_coord = torch.stack((metric_x.reshape(-1), metric_y.reshape(-1)), dim=-1)
    return metric_coord.unsqueeze(0).repeat(batch_size, 1, 1)


def create_image_grids(batch_size, device):
    u = torch.linspace(0, GrdImg_W / 14, int(GrdImg_W / 14), device=device)
    v = torch.linspace(0, GrdImg_H / 14, int(GrdImg_H / 14), device=device)
    v, u = torch.meshgrid(v, u, indexing="ij")
    v = v.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    u = u.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    return u, v


def build_experiment_label(args, settings):
    return (
        f"kitti_ori_{args.rotation_range}"
        f"_matches_{settings['num_samples_matches']}"
        f"_depth_{args.max_depth}"
        f"_beta_{args.beta}"
        f"_bev_{settings['sat_bev_res']}"
        f"_grid_{args.loss_grid_size}"
        f"_lr_{args.learning_rate}"
        f"_scale_True"
        f"_temp_{args.temperature}"
        f"_neg_{args.negative_distance_threshold}"
    )


def build_dataloaders(settings, rotation_range, batch_size):
    train_file = SPLITS_DIR / "train_files.txt"
    test1_file = SPLITS_DIR / "test1_files.txt"
    test2_file = SPLITS_DIR / "test2_files.txt"

    for split_path in (train_file, test1_file, test2_file):
        if not split_path.exists():
            raise FileNotFoundError(f"Missing KITTI split file at {split_path}")

    train_set = SatGrdDataset(
        root=settings["dataset_root"],
        file=train_file.as_posix(),
        transform=(satmap_transform, grdimage_transform),
        shift_range_lat=settings["shift_range_lat"],
        shift_range_lon=settings["shift_range_lon"],
        rotation_range=rotation_range,
    )
    test1_set = SatGrdDatasetTest(
        root=settings["dataset_root"],
        file=test1_file.as_posix(),
        transform=(satmap_transform, grdimage_transform),
        shift_range_lat=settings["shift_range_lat"],
        shift_range_lon=settings["shift_range_lon"],
        rotation_range=rotation_range,
    )
    test2_set = SatGrdDatasetTest(
        root=settings["dataset_root"],
        file=test2_file.as_posix(),
        transform=(satmap_transform, grdimage_transform),
        shift_range_lat=settings["shift_range_lat"],
        shift_range_lon=settings["shift_range_lon"],
        rotation_range=rotation_range,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )
    test1_loader = DataLoader(
        test1_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )
    test2_loader = DataLoader(
        test2_set,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )
    return train_loader, test1_loader, test2_loader


def extract_features(feature_extractor, grd, sat):
    with torch.no_grad():
        grd_feature = feature_extractor(grd)
        sat_feature = feature_extractor(sat)
    return grd_feature, sat_feature


def build_matching_inputs(depth, camera_k, grd_feature_shape, metric_coord_sat, u_grid, v_grid, max_depth):
    batch_size = depth.shape[0]
    camera_k = camera_k.clone()
    camera_k[:, 0, :] = camera_k[:, 0, :] / 14
    camera_k[:, 1, :] = camera_k[:, 1, :] / 14

    depth_downsampled = F.interpolate(depth, size=grd_feature_shape, mode="nearest")
    mask = ~(depth_downsampled == max_depth).flatten(1)

    fx = camera_k[:, 0, 0].view(batch_size, 1, 1, 1)
    fy = camera_k[:, 1, 1].view(batch_size, 1, 1, 1)
    cx = camera_k[:, 0, 2].view(batch_size, 1, 1, 1)
    cy = camera_k[:, 1, 2].view(batch_size, 1, 1, 1)

    grd_x = -depth_downsampled
    grd_y = (u_grid[:batch_size] - cx) * depth_downsampled / fx
    grd_z = (v_grid[:batch_size] - cy) * depth_downsampled / fy
    metric_coord_grd = torch.cat((grd_x.flatten(2), grd_y.flatten(2), grd_z.flatten(2)), 1).permute(0, 2, 1)
    bev_coord_grd = metric_coord_grd[:, :, :2]

    return metric_coord_sat[:batch_size], bev_coord_grd, mask


def sample_matches(matching_score, num_samples_matches):
    _, _, num_kpts_grd = matching_score.shape
    matches_row = matching_score.flatten(1)
    sampled_matching_idx = torch.multinomial(matches_row, num_samples_matches)
    sat_indices_sampled = torch.div(sampled_matching_idx, num_kpts_grd, rounding_mode="trunc")
    grd_indices_sampled = sampled_matching_idx % num_kpts_grd
    return matches_row, sampled_matching_idx, sat_indices_sampled, grd_indices_sampled


def gather_sampled_matches(metric_coord_sat, bev_coord_grd, matches_row, sampled_matching_idx, sat_indices_sampled, grd_indices_sampled):
    batch_size, num_samples = sampled_matching_idx.shape
    batch_idx = torch.arange(batch_size, device=metric_coord_sat.device).view(batch_size, 1).expand(-1, num_samples)
    sat_points = metric_coord_sat[batch_idx, sat_indices_sampled, :]
    grd_points = bev_coord_grd[batch_idx, grd_indices_sampled, :]
    weights = matches_row[batch_idx, sampled_matching_idx]
    return sat_points, grd_points, weights


def build_match_batch(model, depth, camera_k, grd_feature, sat_feature, metric_coord_sat, u_grid, v_grid, max_depth, num_samples_matches):
    metric_coord_sat_batch, bev_coord_grd, mask = build_matching_inputs(
        depth,
        camera_k,
        grd_feature.shape[-2:],
        metric_coord_sat,
        u_grid,
        v_grid,
        max_depth,
    )
    matching_score, matching_score_original = model(grd_feature, sat_feature, mask)
    matches_row, sampled_matching_idx, sat_indices_sampled, grd_indices_sampled = sample_matches(
        matching_score,
        num_samples_matches,
    )
    sat_points, grd_points, weights = gather_sampled_matches(
        metric_coord_sat_batch,
        bev_coord_grd,
        matches_row,
        sampled_matching_idx,
        sat_indices_sampled,
        grd_indices_sampled,
    )
    return (
        metric_coord_sat_batch,
        bev_coord_grd,
        mask,
        sat_indices_sampled,
        grd_indices_sampled,
        sat_points,
        grd_points,
        weights,
        matching_score_original,
    )


def estimate_pose(sat_points, grd_points, weights):
    rotation, translation, scale, _ = weighted_procrustes_2d_with_scale(
        grd_points,
        sat_points,
        use_weights=True,
        use_mask=True,
        w=weights,
    )
    return rotation, translation, scale


def compute_valid_pose_mask(rotation, translation, scale):
    if translation is None:
        return None
    valid_pose = torch.isfinite(translation).all(dim=(1, 2)) & torch.isfinite(rotation).all(dim=(1, 2))
    if scale is not None:
        valid_pose &= torch.isfinite(scale).all(dim=(1, 2))
    return valid_pose


def filter_batch(valid_mask, *tensors):
    filtered = []
    for tensor in tensors:
        if tensor is None:
            filtered.append(None)
        else:
            filtered.append(tensor[valid_mask])
    return filtered


def train_epoch(model, feature_extractor, dataloader, optimizer, device, args, settings, metric_coord_sat, metric_coord4loss, u_grid, v_grid):
    model.train()

    for data in dataloader:
        sat, grd, depth, camera_k, tgt, rotation_gt = data
        sat_size = sat.shape[-1]

        sat = sat.to(device)
        grd = grd.to(device)
        depth = torch.clamp(depth.to(device), 0, args.max_depth)
        camera_k = camera_k.to(device)
        tgt = tgt.to(device).float()
        rotation_gt = rotation_gt.to(device).float()
        tgt_metric = (tgt / sat_size) * settings["grid_size_h"]

        grd_feature, sat_feature = extract_features(feature_extractor, grd, sat)
        (
            metric_coord_sat_batch,
            bev_coord_grd,
            mask,
            sat_indices_sampled,
            grd_indices_sampled,
            sat_points,
            grd_points,
            weights,
            matching_score_original,
        ) = build_match_batch(
            model,
            depth,
            camera_k,
            grd_feature,
            sat_feature,
            metric_coord_sat,
            u_grid,
            v_grid,
            args.max_depth,
            settings["num_samples_matches"],
        )

        rotation_pred, translation_pred, scale = estimate_pose(sat_points, grd_points, weights)
        valid_pose = compute_valid_pose_mask(rotation_pred, translation_pred, scale)
        if valid_pose is None or not valid_pose.any():
            print("Skipping batch: singular transformation matrix")
            continue

        num_invalid = int((~valid_pose).sum().item())
        if num_invalid:
            print(f"Skipping {num_invalid} samples: invalid transformation estimate")

        (
            rotation_gt,
            tgt_metric,
            rotation_pred,
            translation_pred,
            scale,
            sat_points,
            grd_points,
            sat_indices_sampled,
            grd_indices_sampled,
            matching_score_original,
            metric_coord_sat_batch,
            bev_coord_grd,
            mask,
        ) = filter_batch(
            valid_pose,
            rotation_gt,
            tgt_metric,
            rotation_pred,
            translation_pred,
            scale,
            sat_points,
            grd_points,
            sat_indices_sampled,
            grd_indices_sampled,
            matching_score_original,
            metric_coord_sat_batch,
            bev_coord_grd,
            mask,
        )

        distance_loss = torch.mean(
            loss_bev_space(metric_coord4loss, rotation_gt, tgt_metric, rotation_pred, translation_pred)
        )
        # Keep InfoNCE on the metric-aligned coordinates without feeding the estimated scale back into the loss.
        infonce_loss = torch.mean(
            compute_infonce_loss_match_all_with_scale_select_negatives(
                rotation_gt,
                tgt_metric,
                sat_points,
                grd_points,
                torch.ones_like(scale),
                sat_indices_sampled,
                grd_indices_sampled,
                matching_score_original,
                metric_coord_sat_batch,
                bev_coord_grd,
                mask,
                settings["grid_size_h"],
                negative_distance_threshold=args.negative_distance_threshold,
            )
        )
        loss = distance_loss + args.beta * infonce_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()


def compute_yaw_errors(rotation_gt, rotation_pred):
    rotation_gt_np = rotation_gt.cpu().numpy()
    rotation_pred_np = rotation_pred.cpu().numpy()
    yaw_errors = []
    for b in range(rotation_pred_np.shape[0]):
        yaw_pred = np.degrees(np.arctan2(rotation_pred_np[b, 1, 0], rotation_pred_np[b, 0, 0]))
        yaw_gt = np.degrees(np.arctan2(rotation_gt_np[b, 1, 0], rotation_gt_np[b, 0, 0]))
        diff = np.abs(yaw_pred - yaw_gt)
        yaw_errors.append(float(np.min([diff, 360 - diff])))
    return yaw_errors


def evaluate_split(name, model, feature_extractor, dataloader, device, args, settings, metric_coord_sat, u_grid, v_grid):
    model.eval()
    translation_error = []
    yaw_error = []
    scales = []

    with torch.no_grad():
        for data in dataloader:
            sat, grd, depth, camera_k, tgt, rotation_gt = data
            sat_size = sat.shape[-1]

            sat = sat.to(device)
            grd = grd.to(device)
            depth = torch.clamp(depth.to(device), 0, args.max_depth)
            camera_k = camera_k.to(device)
            tgt = tgt.to(device).float()
            rotation_gt = rotation_gt.to(device).float()

            grd_feature, sat_feature = extract_features(feature_extractor, grd, sat)
            (
                _,
                _,
                _,
                _,
                _,
                sat_points,
                grd_points,
                weights,
                _,
            ) = build_match_batch(
                model,
                depth,
                camera_k,
                grd_feature,
                sat_feature,
                metric_coord_sat,
                u_grid,
                v_grid,
                args.max_depth,
                settings["num_samples_matches"],
            )

            rotation_pred, translation_pred, scale = estimate_pose(sat_points, grd_points, weights)
            valid_pose = compute_valid_pose_mask(rotation_pred, translation_pred, scale)
            if valid_pose is None or not valid_pose.any():
                print("Skipping batch: singular transformation matrix")
                continue

            num_invalid = int((~valid_pose).sum().item())
            if num_invalid:
                print(f"Skipping {num_invalid} samples: invalid transformation estimate")

            rotation_gt = rotation_gt[valid_pose]
            tgt = tgt[valid_pose]
            rotation_pred = rotation_pred[valid_pose]
            translation_pred = translation_pred[valid_pose]
            if scale is not None:
                scale = scale[valid_pose]
                scales.extend(scale.view(-1).cpu().tolist())

            translation_pred = (translation_pred / settings["grid_size_h"]) * sat_size
            translation_error.extend(torch.norm(translation_pred - tgt, dim=-1).view(-1).cpu().tolist())
            yaw_error.extend(compute_yaw_errors(rotation_gt, rotation_pred))

    if not translation_error:
        raise RuntimeError(f"{name} produced no valid samples.")

    metrics = {
        "translation_mean": float(np.mean(translation_error)),
        "translation_median": float(np.median(translation_error)),
        "yaw_mean": float(np.mean(yaw_error)),
        "yaw_median": float(np.median(yaw_error)),
    }
    if scales:
        metrics["scale_mean"] = float(np.mean(scales))
        metrics["scale_median"] = float(np.median(scales))

    print(name)
    print(f"Mean Translation Error: {metrics['translation_mean']:.3f} pixels")
    print(f"Median Translation Error: {metrics['translation_median']:.3f} pixels")
    print(f"Mean Yaw Error: {metrics['yaw_mean']:.3f} deg")
    print(f"Median Yaw Error: {metrics['yaw_median']:.3f} deg")
    if "scale_mean" in metrics:
        print(f"Mean Scale: {metrics['scale_mean']:.3f}")
        print(f"Median Scale: {metrics['scale_median']:.3f}")
    return metrics


def append_metric(results_dir, filename, epoch, value, header):
    path = results_dir / filename
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{epoch}_{header} {value:.6f}\n")


def save_metrics(results_dir, epoch, split_name, metrics):
    prefix = split_name.replace(" ", "")
    append_metric(results_dir, "Mean_distance_error.txt", epoch, metrics["translation_mean"], f"{prefix}_mean_distance_error_in_pixels:")
    append_metric(results_dir, "Median_distance_error.txt", epoch, metrics["translation_median"], f"{prefix}_median_distance_error_in_pixels:")
    append_metric(results_dir, "Mean_orientation_error.txt", epoch, metrics["yaw_mean"], f"{prefix}_mean_yaw_error:")
    append_metric(results_dir, "Median_orientation_error.txt", epoch, metrics["yaw_median"], f"{prefix}_median_yaw_error:")
    if "scale_mean" in metrics:
        append_metric(results_dir, "Mean_scale.txt", epoch, metrics["scale_mean"], f"{prefix}_mean_scale:")
        append_metric(results_dir, "Median_scale.txt", epoch, metrics["scale_median"], f"{prefix}_median_scale:")


def main():
    args = parse_args()
    config = load_config()
    set_seeds(config.getint("RandomSeed", "seed"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    settings = resolve_runtime_settings(config)
    train_loader, test1_loader, test2_loader = build_dataloaders(
        settings,
        args.rotation_range,
        args.batch_size,
    )

    label = build_experiment_label(args, settings)
    print(f"Experiment label: {label}")
    torch.cuda.empty_cache()

    feature_extractor = DinoExtractor().to(device)
    feature_extractor.eval()
    model = KittiCrossViewMatcher(
        device,
        embed_dim=1024,
        temperature=args.temperature,
    ).to(device)

    if args.epoch_to_resume > 0:
        model_path = CHECKPOINT_ROOT / label / str(args.epoch_to_resume - 1) / "model.pt"
        model.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

    metric_coord_sat = create_metric_grid(settings["grid_size_h"], settings["sat_bev_res"], args.batch_size, device)
    metric_coord4loss = create_metric_grid(args.loss_grid_size, settings["num_virtual_point"], 1, device)
    u_grid, v_grid = create_image_grids(args.batch_size, device)

    results_dir = RESULTS_ROOT / label
    results_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epoch_to_resume, NUM_EPOCHS):
        train_epoch(
            model,
            feature_extractor,
            train_loader,
            optimizer,
            device,
            args,
            settings,
            metric_coord_sat,
            metric_coord4loss,
            u_grid,
            v_grid,
        )

        model_dir = CHECKPOINT_ROOT / label / str(epoch)
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f"save checkpoint at {model_dir}")
        torch.save(model.state_dict(), model_dir / "model.pt")

        print(f"Epoch {epoch} - Evaluating on Test 1...")
        test1_metrics = evaluate_split(
            "Test 1",
            model,
            feature_extractor,
            test1_loader,
            device,
            args,
            settings,
            metric_coord_sat,
            u_grid,
            v_grid,
        )
        save_metrics(results_dir, epoch, "Test1", test1_metrics)

        print(f"Epoch {epoch} - Evaluating on Test 2...")
        test2_metrics = evaluate_split(
            "Test 2",
            model,
            feature_extractor,
            test2_loader,
            device,
            args,
            settings,
            metric_coord_sat,
            u_grid,
            v_grid,
        )
        save_metrics(results_dir, epoch, "Test2", test2_metrics)


if __name__ == "__main__":
    main()
