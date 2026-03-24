import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import argparse
import configparser
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dataloaders.dataloader_kitti_with_depth import (
    GrdImg_H,
    GrdImg_W,
    SatGrdDatasetTest,
    get_meter_per_pixel,
    grdimage_transform,
    satmap_transform,
)
from models.kitti_matcher import KittiCrossViewMatcher
from models.modules import DinoExtractor
from models.utils import weighted_procrustes_2d_with_scale


NUM_WORKERS = 4
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.ini"
SPLITS_DIR = PROJECT_ROOT / "KITTI_splits"
RESULTS_ROOT = PROJECT_ROOT / "results"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=384)
    parser.add_argument('--rotation_range', type=float, default=10.0)
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--max_depth', type=float, default=40.0)
    return parser.parse_args()


def load_config():
    config = configparser.ConfigParser()
    if not config.read(CONFIG_PATH):
        raise FileNotFoundError(f'Could not read config file at {CONFIG_PATH}')
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


def resolve_results_root(args):
    return Path(args.results_dir) if args.results_dir is not None else RESULTS_ROOT


def sanitize_filename_part(value):
    sanitized = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value))
    return sanitized.strip('._') or 'run'


def build_results_path(model_path, results_root):
    results_root.mkdir(parents=True, exist_ok=True)
    model_tag_parts = model_path.parts[-3:] if len(model_path.parts) >= 3 else (model_path.name,)
    model_tag = sanitize_filename_part('_'.join(model_tag_parts))
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return results_root / f'eval_kitti_{model_tag}_{timestamp}.json'


def create_metric_grid(grid_size, resolution, batch_size, device):
    axis = torch.linspace(-grid_size / 2, grid_size / 2, resolution, device=device)
    metric_x, metric_y = torch.meshgrid(axis, axis, indexing='ij')
    metric_coord = torch.stack((metric_x.reshape(-1), metric_y.reshape(-1)), dim=-1)
    return metric_coord.unsqueeze(0).repeat(batch_size, 1, 1)


def create_image_grids(batch_size, device):
    u = torch.linspace(0, GrdImg_W / 14, int(GrdImg_W / 14), device=device)
    v = torch.linspace(0, GrdImg_H / 14, int(GrdImg_H / 14), device=device)
    v, u = torch.meshgrid(v, u, indexing='ij')
    v = v.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    u = u.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    return u, v


def build_test_dataset(config, split_path, rotation_range):
    dataset_root = config.get('KITTI', 'dataset_root')
    return SatGrdDatasetTest(
        root=dataset_root,
        file=split_path.as_posix(),
        transform=(satmap_transform, grdimage_transform),
        shift_range_lat=config.getfloat('KITTI', 'shift_range_lat'),
        shift_range_lon=config.getfloat('KITTI', 'shift_range_lon'),
        rotation_range=rotation_range,
    )


def build_eval_loader(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
        drop_last=False,
    )


def summarize_metrics(name, translation_errors, yaw_errors, longitudinal_errors, lateral_errors):
    metrics = {
        'num_samples': len(translation_errors),
        'translation_mean_m': float(np.mean(translation_errors)),
        'translation_median_m': float(np.median(translation_errors)),
        'yaw_mean_deg': float(np.mean(yaw_errors)),
        'yaw_median_deg': float(np.median(yaw_errors)),
        'lateral_recall_1m': float(np.mean(np.array(lateral_errors) < 1)),
        'lateral_recall_5m': float(np.mean(np.array(lateral_errors) < 5)),
        'longitudinal_recall_1m': float(np.mean(np.array(longitudinal_errors) < 1)),
        'longitudinal_recall_5m': float(np.mean(np.array(longitudinal_errors) < 5)),
        'yaw_recall_1deg': float(np.mean(np.array(yaw_errors) < 1)),
        'yaw_recall_5deg': float(np.mean(np.array(yaw_errors) < 5)),
    }
    print(f'=== {name} ===')
    print(f"Mean Translation Error: {metrics['translation_mean_m']:.3f} m")
    print(f"Median Translation Error: {metrics['translation_median_m']:.3f} m")
    print(f"Mean Yaw Error: {metrics['yaw_mean_deg']:.3f} deg")
    print(f"Median Yaw Error: {metrics['yaw_median_deg']:.3f} deg")
    print(f"Lateral Recall@1m: {metrics['lateral_recall_1m']:.5f}")
    print(f"Lateral Recall@5m: {metrics['lateral_recall_5m']:.5f}")
    print(f"Longitudinal Recall@1m: {metrics['longitudinal_recall_1m']:.5f}")
    print(f"Longitudinal Recall@5m: {metrics['longitudinal_recall_5m']:.5f}")
    print(f"Yaw Recall@1deg: {metrics['yaw_recall_1deg']:.5f}")
    print(f"Yaw Recall@5deg: {metrics['yaw_recall_5deg']:.5f}")
    return metrics


def save_results(results_path, payload):
    with open(results_path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
        handle.write('\n')


def evaluate_loader(
    name,
    dataloader,
    feature_extractor,
    matcher_model,
    args,
    grid_size_h,
    num_samples_matches,
    metric_coord_sat,
    u_grid,
    v_grid,
    meter_per_pixel,
    device,
):
    translation_errors = []
    yaw_errors = []
    longitudinal_errors = []
    lateral_errors = []

    with torch.no_grad():
        for data in dataloader:
            sat, grd, depth, camera_k, tgt, rotation_gt = data
            batch_size = sat.shape[0]
            sat_size = sat.shape[-1]

            sat = sat.to(device)
            grd = grd.to(device)
            depth = torch.clamp(depth.to(device), 0, args.max_depth)
            camera_k = camera_k.to(device).clone()
            tgt = tgt.to(device)
            rotation_gt = rotation_gt.to(device)

            camera_k[:, 0, :] = camera_k[:, 0, :] / 14
            camera_k[:, 1, :] = camera_k[:, 1, :] / 14

            grd_feature = feature_extractor(grd)
            sat_feature = feature_extractor(sat)

            depth_downsampled = F.interpolate(depth, size=grd_feature.shape[-2:], mode='nearest')
            mask = ~(depth_downsampled == args.max_depth).flatten(1)

            fx = camera_k[:, 0, 0].view(batch_size, 1, 1, 1)
            fy = camera_k[:, 1, 1].view(batch_size, 1, 1, 1)
            cx = camera_k[:, 0, 2].view(batch_size, 1, 1, 1)
            cy = camera_k[:, 1, 2].view(batch_size, 1, 1, 1)

            grd_x = -depth_downsampled
            grd_y = (u_grid[:batch_size] - cx) * depth_downsampled / fx
            grd_z = (v_grid[:batch_size] - cy) * depth_downsampled / fy
            metric_coord_grd = torch.cat((grd_x.flatten(2), grd_y.flatten(2), grd_z.flatten(2)), 1).permute(0, 2, 1)
            bev_coord_grd = metric_coord_grd[:, :, :2]

            matching_score, _ = matcher_model(grd_feature, sat_feature, mask)
            _, _, num_kpts_grd = matching_score.shape

            matches_row = matching_score.flatten(1)
            sampled_matching_idx = torch.multinomial(matches_row, num_samples_matches)
            sat_indices_sampled = torch.div(sampled_matching_idx, num_kpts_grd, rounding_mode='trunc')
            grd_indices_sampled = sampled_matching_idx % num_kpts_grd

            batch_idx = torch.arange(batch_size, device=device).view(batch_size, 1).expand(-1, num_samples_matches)
            metric_coord_sat_batch = metric_coord_sat[:batch_size]
            sat_points = metric_coord_sat_batch[batch_idx, sat_indices_sampled, :]
            grd_points = bev_coord_grd[batch_idx, grd_indices_sampled, :]
            weights = matches_row[batch_idx, sampled_matching_idx]

            rotation_pred, translation_pred, _, _ = weighted_procrustes_2d_with_scale(
                grd_points,
                sat_points,
                use_weights=True,
                use_mask=True,
                w=weights,
            )

            if translation_pred is None:
                print('Skipping batch: singular transformation matrix')
                continue

            translation_pred = (translation_pred / grid_size_h) * sat_size
            delta_pixels = translation_pred.squeeze(1) - tgt.squeeze(1)
            translation_batch = torch.linalg.norm(delta_pixels, dim=-1).cpu().numpy() * meter_per_pixel

            rotation_gt_np = rotation_gt.cpu().numpy()
            rotation_pred_np = rotation_pred.cpu().numpy()
            delta_pixels_np = delta_pixels.cpu().numpy()

            for b in range(batch_size):
                translation_errors.append(float(translation_batch[b]))

                yaw_pred = np.degrees(np.arctan2(rotation_pred_np[b, 1, 0], rotation_pred_np[b, 0, 0]))
                yaw_gt = np.degrees(np.arctan2(rotation_gt_np[b, 1, 0], rotation_gt_np[b, 0, 0]))
                yaw_diff = np.abs(yaw_pred - yaw_gt)
                yaw_errors.append(float(np.min([yaw_diff, 360 - yaw_diff])))

                heading_vec = np.array([np.sin(np.deg2rad(yaw_gt)), np.cos(np.deg2rad(yaw_gt))])
                dx, dy = delta_pixels_np[b]
                longitudinal_pix = dx * heading_vec[0] + dy * heading_vec[1]
                lateral_pix = heading_vec[0] * dy - heading_vec[1] * dx

                longitudinal_errors.append(float(np.abs(longitudinal_pix) * meter_per_pixel))
                lateral_errors.append(float(np.abs(lateral_pix) * meter_per_pixel))

    if not translation_errors:
        raise RuntimeError(f'{name} produced no valid samples.')

    return summarize_metrics(name, translation_errors, yaw_errors, longitudinal_errors, lateral_errors)


def main():
    args = parse_args()
    config = load_config()
    set_seeds(config.getint('RandomSeed', 'seed'))

    if args.model_path is None:
        raise ValueError('Specify --model_path.')

    test1_file = SPLITS_DIR / 'test1_files.txt'
    test2_file = SPLITS_DIR / 'test2_files.txt'
    if not test1_file.exists() or not test2_file.exists():
        raise FileNotFoundError(f'Missing KITTI split files in {SPLITS_DIR}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    grid_size_h = config.getfloat('KITTI', 'grid_size_h')
    sat_bev_res = config.getint('Model', 'sat_bev_res')
    num_samples_matches = config.getint('Matching', 'num_samples_matches')
    meter_per_pixel = get_meter_per_pixel()

    test1_dataset = build_test_dataset(config, test1_file, args.rotation_range)
    test2_dataset = build_test_dataset(config, test2_file, args.rotation_range)
    test1_loader = build_eval_loader(test1_dataset, args.batch_size)
    test2_loader = build_eval_loader(test2_dataset, args.batch_size)

    model_path = Path(args.model_path)
    print(f'Loading checkpoint: {model_path}')

    torch.cuda.empty_cache()
    feature_extractor = DinoExtractor().to(device)
    feature_extractor.eval()
    matcher_model = KittiCrossViewMatcher(device, embed_dim=1024).to(device)
    matcher_model.load_state_dict(torch.load(model_path, map_location=device))
    matcher_model.eval()

    metric_coord_sat = create_metric_grid(grid_size_h, sat_bev_res, args.batch_size, device)
    u_grid, v_grid = create_image_grids(args.batch_size, device)

    test1_metrics = evaluate_loader(
        'Test 1',
        test1_loader,
        feature_extractor,
        matcher_model,
        args,
        grid_size_h,
        num_samples_matches,
        metric_coord_sat,
        u_grid,
        v_grid,
        meter_per_pixel,
        device,
    )
    print('===========================')
    test2_metrics = evaluate_loader(
        'Test 2',
        test2_loader,
        feature_extractor,
        matcher_model,
        args,
        grid_size_h,
        num_samples_matches,
        metric_coord_sat,
        u_grid,
        v_grid,
        meter_per_pixel,
        device,
    )

    results_path = build_results_path(model_path, resolve_results_root(args))
    save_results(
        results_path,
        {
            'script': Path(__file__).name,
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'model_path': str(model_path),
            'args': {k: v for k, v in vars(args).items() if k != 'results_dir'},
            'splits': {
                'test1': test1_file.as_posix(),
                'test2': test2_file.as_posix(),
            },
            'metrics': {
                'test1': test1_metrics,
                'test2': test2_metrics,
            },
        },
    )
    print(f'Saved results to {results_path}')


if __name__ == '__main__':
    main()
