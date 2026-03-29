import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import ast
import configparser
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, default_collate

from dataloaders.dataloader_vigor_with_depth import VIGORDataset, transform_grd, transform_sat
from models.modules import DinoExtractor
from models.utils import (
    e2eProbabilisticProcrustesSolver,
    weighted_procrustes_2d_with_scale,
)
from models.vigor_matcher import VigorCrossViewMatcher


CITY_METERS_PER_PIXEL = {
    "NewYork": 0.113248,
    "Seattle": 0.100817,
    "SanFrancisco": 0.118141,
    "Chicago": 0.111262,
}
NUM_WORKERS = 4
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.ini"
RESULTS_ROOT = PROJECT_ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--area", type=str, default="samearea", help="samearea or crossarea")
    parser.add_argument("-b", "--batch_size", type=int, default=80, help="batch size")
    parser.add_argument(
        "--random_orientation",
        type=float,
        default=0.0,
        help="random orientation range from -x degrees to x degrees",
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--max_depth", type=float, default=35.0)

    parser.add_argument("--ransac", choices=("True", "False"), default="False")
    parser.add_argument("--th_soft_inlier", type=float, default=5)
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


def safe_collate(batch):
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        return None
    return default_collate(batch)


def build_eval_dataset(dataset_root, area, random_orientation):
    return VIGORDataset(
        root=dataset_root,
        split=area,
        train=False,
        transform=(transform_grd, transform_sat),
        random_orientation=random_orientation,
    )


def resolve_model_path(args):
    if args.model_path is None:
        raise ValueError("Specify --model_path.")
    return Path(args.model_path)


def resolve_results_root(args):
    return Path(args.results_dir) if args.results_dir is not None else RESULTS_ROOT


def sanitize_filename_part(value):
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    return sanitized.strip("._") or "run"


def build_results_path(model_path, area, results_root):
    results_root.mkdir(parents=True, exist_ok=True)
    model_tag_parts = model_path.parts[-3:] if len(model_path.parts) >= 3 else (model_path.name,)
    model_tag = sanitize_filename_part("_".join(model_tag_parts))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return results_root / f"eval_vigor_{sanitize_filename_part(area)}_{model_tag}_{timestamp}.json"


def create_metric_grid(grid_size, resolution, batch_size, device):
    axis = torch.linspace(-grid_size / 2, grid_size / 2, resolution, device=device)
    metric_x, metric_y = torch.meshgrid(axis, axis, indexing="ij")
    metric_coord = torch.stack((metric_x.reshape(-1), metric_y.reshape(-1)), dim=-1)
    return metric_coord.unsqueeze(0).repeat(batch_size, 1, 1)


def create_city_coordinates(sat_bev_res, device):
    return {
        city: create_metric_grid(640 * meters_per_pixel, sat_bev_res, 1, device)
        for city, meters_per_pixel in CITY_METERS_PER_PIXEL.items()
    }


def create_spherical_grids(ground_image_size, batch_size, device):
    phi = torch.linspace(0, 2 * np.pi, int(ground_image_size[1] / 14), device=device)
    theta = torch.linspace(0, np.pi, int(ground_image_size[0] / 14), device=device)
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")
    theta = theta.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    phi = phi.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    return theta, phi


def build_pose_solver(args, metric_coord_sat, bev_coord_grd):
    return e2eProbabilisticProcrustesSolver(
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


def summarize_metrics(translation_errors, yaw_errors):
    if not translation_errors:
        raise RuntimeError("Evaluation produced no valid samples.")

    metrics = {
        "num_samples": len(translation_errors),
        "translation_mean_m": float(np.mean(translation_errors)),
        "translation_median_m": float(np.median(translation_errors)),
        "yaw_mean_deg": float(np.mean(yaw_errors)),
        "yaw_median_deg": float(np.median(yaw_errors)),
    }
    print(f"Mean Translation Error: {metrics['translation_mean_m']:.3f}")
    print(f"Median Translation Error: {metrics['translation_median_m']:.3f}")
    print(f"Mean Yaw Error: {metrics['yaw_mean_deg']:.3f}")
    print(f"Median Yaw Error: {metrics['yaw_median_deg']:.3f}")
    return metrics


def save_results(results_path, payload):
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def main():
    args = parse_args()
    config = load_config()
    set_seeds(config.getint("RandomSeed", "seed"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_root = config["VIGOR"]["scitas_dataset_root"]
    sat_bev_res = config.getint("Model", "sat_bev_res")
    num_samples_matches = config.getint("Matching", "num_samples_matches")
    ground_image_size = ast.literal_eval(config.get("VIGOR", "ground_image_size"))
    ransac = args.ransac == "True"
    test_dataset = build_eval_dataset(
        dataset_root,
        args.area,
        args.random_orientation,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=safe_collate,
    )

    model_path = resolve_model_path(args)
    print(f"Loading checkpoint: {model_path}")
    print(f"Evaluating {args.area} on test")

    torch.cuda.empty_cache()
    shared_feature_extractor = DinoExtractor().to(device)
    matcher_model = VigorCrossViewMatcher(device, sat_bev_res=sat_bev_res, embed_dim=1024)
    matcher_model.load_state_dict(torch.load(model_path, map_location=device))
    matcher_model.to(device)
    matcher_model.eval()

    city_coords = create_city_coordinates(sat_bev_res, device)
    theta, phi = create_spherical_grids(ground_image_size, args.batch_size, device)

    translation_errors = []
    yaw_errors = []

    with torch.no_grad():
        for data in test_dataloader:
            if data is None:
                continue

            grd, depth, sat, tgt, Rgt, cities, resolution = data
            metric_coord_sat = torch.cat([city_coords[city] for city in cities], dim=0)

            grd = grd.to(device)
            depth = depth.to(device)
            sat = sat.to(device)
            tgt = tgt.to(device)
            Rgt = Rgt.to(device)

            batch_size = sat.shape[0]
            grd_feature = shared_feature_extractor(grd)
            sat_feature = shared_feature_extractor(sat)

            depth = torch.clip(depth, 0, args.max_depth)
            depth_downsampled = F.interpolate(depth, size=grd_feature.shape[-2:], mode="nearest")
            mask = ~(depth_downsampled == depth.max()).flatten(1)

            grd_x = depth_downsampled * torch.sin(theta[:batch_size]) * torch.cos(phi[:batch_size])
            grd_y = depth_downsampled * torch.sin(theta[:batch_size]) * (-torch.sin(phi[:batch_size]))
            grd_z = depth_downsampled * torch.cos(theta[:batch_size])
            metric_coord_grd = torch.cat((grd_x.flatten(2), grd_y.flatten(2), grd_z.flatten(2)), 1).permute(0, 2, 1)
            bev_coord_grd = metric_coord_grd[:, :, :2]

            matching_score, _ = matcher_model(grd_feature, sat_feature, mask)
            _, _, num_kpts_grd = matching_score.shape

            matches_row = matching_score.flatten(1)
            batch_idx = torch.arange(batch_size, device=matches_row.device).view(batch_size, 1).expand(-1, num_samples_matches)
            sampled_matching_idx = torch.multinomial(matches_row, num_samples_matches)
            sat_indices_sampled = torch.div(sampled_matching_idx, num_kpts_grd, rounding_mode="trunc")
            grd_indices_sampled = sampled_matching_idx % num_kpts_grd

            X = metric_coord_sat[batch_idx, sat_indices_sampled, :]
            Y = bev_coord_grd[batch_idx, grd_indices_sampled, :]
            weights = matches_row[batch_idx, sampled_matching_idx]

            if ransac:
                pose_solver = build_pose_solver(args, metric_coord_sat, bev_coord_grd)
                R, t, scale, _, _ = pose_solver.estimate_pose(matching_score, return_inliers=False)
            else:
                R, t, scale, _ = weighted_procrustes_2d_with_scale(Y, X, use_weights=True, use_mask=True, w=weights)

            if t is None:
                print("Skipping batch: singular transformation matrix")
                continue

            valid_pose = torch.isfinite(t).all(dim=(1, 2)) & torch.isfinite(R).all(dim=(1, 2))
            if scale is not None:
                valid_pose &= torch.isfinite(scale).all(dim=(1, 2))

            if not valid_pose.any():
                print("Skipping batch: singular transformation matrix")
                continue

            num_invalid = int((~valid_pose).sum().item())
            if num_invalid:
                print(f"Skipping {num_invalid} samples: invalid transformation estimate")

            Rgt_np = Rgt.cpu().numpy()
            R_np = R.cpu().numpy()
            for b in range(batch_size):
                if not valid_pose[b]:
                    continue
                yaw = np.degrees(np.arctan2(R_np[b, 1, 0], R_np[b, 0, 0]))
                yaw_gt = np.degrees(np.arctan2(Rgt_np[b, 1, 0], Rgt_np[b, 0, 0]))
                diff = abs(yaw - yaw_gt)
                yaw_errors.append(float(min(diff, 360 - diff)))

                tgt_b = tgt[b] * resolution[b]
                translation_errors.append(float(torch.norm(t[b] - tgt_b, dim=-1).item()))

    metrics = summarize_metrics(translation_errors, yaw_errors)
    results_path = build_results_path(model_path, args.area, resolve_results_root(args))
    save_results(
        results_path,
        {
            "script": Path(__file__).name,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model_path": str(model_path),
            "args": {k: v for k, v in vars(args).items() if k != "results_dir"},
            "metrics": metrics,
        },
    )
    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
