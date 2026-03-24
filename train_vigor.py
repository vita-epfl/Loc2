import argparse
import ast
import configparser
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, default_collate

from dataloaders.dataloader_vigor_with_depth import VIGORDataset, transform_grd, transform_sat
from models.loss import (
    compute_infonce_loss_match_all_with_scale_select_negatives,
    loss_bev_space,
)
from models.modules import DinoExtractor
from models.utils import weighted_procrustes_2d_with_scale
from models.vigor_matcher import VigorCrossViewMatcher

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.ini"
CHECKPOINT_ROOT = PROJECT_ROOT.parent / "checkpoints"
RESULTS_ROOT = PROJECT_ROOT.parent / "results"
CITY_METERS_PER_PIXEL = {
    "NewYork": 0.113248,
    "Seattle": 0.100817,
    "SanFrancisco": 0.118141,
    "Chicago": 0.111262,
}
NUM_WORKERS = 4
NUM_EPOCHS = 100
TRAIN_SPLIT = 0.8


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--area', type=str, default='samearea')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('-b', '--batch_size', type=int, default=80)
    parser.add_argument('--random_orientation', type=float, default=0)
    parser.add_argument('--loss_grid_size', type=float, default=5.0)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--epoch_to_resume', type=int, default=0)
    parser.add_argument('--max_depth', type=float, default=35)
    parser.add_argument('--temperature', type=float, default=0.1)
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


def resolve_runtime_settings(args, config):
    dataset_root = config.get('VIGOR', 'dataset_root', fallback=config.get('VIGOR', 'scitas_dataset_root'))

    settings = {
        'dataset_root': dataset_root,
        'grid_size_h': config.getint('VIGOR', 'grid_size_h'),
        'ground_image_size': ast.literal_eval(config.get('VIGOR', 'ground_image_size')),
        'sat_bev_res': config.getint('Model', 'sat_bev_res'),
        'num_samples_matches': config.getint('Matching', 'num_samples_matches'),
        'num_virtual_point': config.getint('Loss', 'num_virtual_point'),
    }
    return args, settings


def safe_collate(batch):
    batch = [sample for sample in batch if sample is not None]
    if not batch:
        return None
    return default_collate(batch)


def create_metric_grid(grid_size, resolution, batch_size, device):
    axis = torch.linspace(-grid_size / 2, grid_size / 2, resolution, device=device)
    metric_x, metric_y = torch.meshgrid(axis, axis, indexing='ij')
    metric_coord = torch.stack((metric_x.reshape(-1), metric_y.reshape(-1)), dim=-1)
    return metric_coord.unsqueeze(0).repeat(batch_size, 1, 1)


def create_city_coordinate_lookup(sat_bev_res, device):
    return {
        city: create_metric_grid(640 * meters_per_pixel, sat_bev_res, 1, device)
        for city, meters_per_pixel in CITY_METERS_PER_PIXEL.items()
    }


def create_spherical_grids(ground_image_size, batch_size, device):
    phi = torch.linspace(0, 2 * np.pi, int(ground_image_size[1] / 14), device=device)
    theta = torch.linspace(0, np.pi, int(ground_image_size[0] / 14), device=device)
    theta, phi = torch.meshgrid(theta, phi, indexing='ij')
    theta = theta.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    phi = phi.unsqueeze(0).unsqueeze(1).repeat(batch_size, 1, 1, 1)
    return theta, phi


def depth_to_metric_coordinates(depth_downsampled, theta, phi):
    grd_x = depth_downsampled * torch.sin(theta) * torch.cos(phi)
    grd_y = depth_downsampled * torch.sin(theta) * (-torch.sin(phi))
    grd_z = depth_downsampled * torch.cos(theta)
    metric_coord_grd = torch.cat((grd_x.flatten(2), grd_y.flatten(2), grd_z.flatten(2)), 1).permute(0, 2, 1)
    return metric_coord_grd, metric_coord_grd[:, :, :2]


def build_matching_inputs(depth, grd_feature_shape, cities, city_coords, theta, phi, max_depth):
    batch_size = depth.shape[0]
    metric_coord_sat = torch.cat([city_coords[city] for city in cities], dim=0)
    depth_downsampled = F.interpolate(depth, size=grd_feature_shape, mode='nearest')
    mask = ~(depth_downsampled == max_depth).flatten(1)
    _, bev_coord_grd = depth_to_metric_coordinates(
        depth_downsampled,
        theta[:batch_size],
        phi[:batch_size],
    )
    return metric_coord_sat, bev_coord_grd, mask


def sample_matches(matching_score, num_samples_matches):
    _, _, num_kpts_grd = matching_score.shape
    matches_row = matching_score.flatten(1)
    sampled_matching_idx = torch.multinomial(matches_row, num_samples_matches)
    sat_indices_sampled = torch.div(sampled_matching_idx, num_kpts_grd, rounding_mode='trunc')
    grd_indices_sampled = sampled_matching_idx % num_kpts_grd
    return matches_row, sampled_matching_idx, sat_indices_sampled, grd_indices_sampled


def gather_sampled_matches(
    metric_coord_sat,
    bev_coord_grd,
    matches_row,
    sampled_matching_idx,
    sat_indices_sampled,
    grd_indices_sampled,
):
    batch_size, num_samples = sampled_matching_idx.shape
    batch_idx = torch.arange(batch_size, device=metric_coord_sat.device).view(batch_size, 1).expand(-1, num_samples)
    sat_points = metric_coord_sat[batch_idx, sat_indices_sampled, :]
    grd_points = bev_coord_grd[batch_idx, grd_indices_sampled, :]
    weights = matches_row[batch_idx, sampled_matching_idx]
    return sat_points, grd_points, weights


def extract_features(feature_extractor, grd, sat):
    with torch.no_grad():
        grd_feature = feature_extractor(grd)
        sat_feature = feature_extractor(sat)
    return grd_feature, sat_feature


def build_match_batch(
    matcher,
    depth,
    grd_feature,
    sat_feature,
    cities,
    city_coords,
    theta,
    phi,
    max_depth,
    num_samples_matches,
):
    metric_coord_sat, bev_coord_grd, mask = build_matching_inputs(
        depth,
        grd_feature.shape[-2:],
        cities,
        city_coords,
        theta,
        phi,
        max_depth,
    )
    matching_score, matching_score_original = matcher(grd_feature, sat_feature, mask)
    matches_row, sampled_matching_idx, sat_indices_sampled, grd_indices_sampled = sample_matches(
        matching_score,
        num_samples_matches,
    )
    sat_points, grd_points, weights = gather_sampled_matches(
        metric_coord_sat,
        bev_coord_grd,
        matches_row,
        sampled_matching_idx,
        sat_indices_sampled,
        grd_indices_sampled,
    )
    return (
        metric_coord_sat,
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


def create_dataloaders(dataset_root, area, batch_size, random_orientation):
    dataset = VIGORDataset(
        root=dataset_root,
        split=area,
        train=True,
        transform=(transform_grd, transform_sat),
        random_orientation=random_orientation,
    )

    dataset_length = len(dataset)
    indices = np.arange(dataset_length)
    np.random.shuffle(indices)
    split_idx = int(dataset_length * TRAIN_SPLIT)
    train_indices, val_indices = indices[:split_idx], indices[split_idx:]
    training_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_dataloader = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=safe_collate,
    )
    val_dataloader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=safe_collate,
    )
    return train_dataloader, val_dataloader


def build_experiment_label(args, settings):
    return (
        f'vigor_{args.area}'
        f'_ori_{args.random_orientation}'
        f'_matches_{settings["num_samples_matches"]}'
        f'_depth_{args.max_depth}'
        f'_beta_{args.beta}'
        f'_bev_{settings["sat_bev_res"]}'
        f'_grid_{args.loss_grid_size}'
        f'_lr_{args.learning_rate}'
        f'_temp_{args.temperature}'
    )


def append_metric(results_dir, filename, epoch, value, header):
    with open(results_dir / filename, 'ab') as handle:
        np.savetxt(handle, [value], fmt='%4f', header=header, comments=f'{epoch}_')


def compute_yaw_errors(rotation_gt, rotation_pred):
    rotation_gt_np = rotation_gt.cpu().numpy()
    rotation_pred_np = rotation_pred.cpu().numpy()
    yaw_errors = []
    for batch_offset in range(rotation_pred.shape[0]):
        cos_pred = rotation_pred_np[batch_offset, 0, 0]
        sin_pred = rotation_pred_np[batch_offset, 1, 0]
        yaw_pred = np.degrees(np.arctan2(sin_pred, cos_pred))

        cos_gt = rotation_gt_np[batch_offset, 0, 0]
        sin_gt = rotation_gt_np[batch_offset, 1, 0]
        yaw_gt = np.degrees(np.arctan2(sin_gt, cos_gt))

        diff = np.abs(yaw_pred - yaw_gt)
        yaw_errors.append(np.min([diff, 360 - diff]))
    return yaw_errors


def train_epoch(
    model,
    feature_extractor,
    dataloader,
    optimizer,
    device,
    args,
    settings,
    city_coords,
    metric_coord4loss,
    theta,
    phi,
):
    model.train()

    for data in dataloader:
        if data is None:
            continue

        grd, depth, sat, tgt, rotation_gt, cities, _ = data
        sat_size = sat.shape[-1]

        grd = grd.to(device)
        depth = depth.to(device)
        sat = sat.to(device)
        tgt = tgt.to(device).float()
        rotation_gt = rotation_gt.to(device).float()
        tgt = (tgt / sat_size) * settings['grid_size_h']

        grd_feature, sat_feature = extract_features(feature_extractor, grd, sat)
        (
            metric_coord_sat,
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
            grd_feature,
            sat_feature,
            cities,
            city_coords,
            theta,
            phi,
            args.max_depth,
            settings['num_samples_matches'],
        )

        rotation_pred, translation_pred, scale, = estimate_pose(sat_points, grd_points, weights)
        if translation_pred is None:
            print('Skipping batch: singular transformation matrix')
            continue

        distance_loss = torch.mean(
            loss_bev_space(metric_coord4loss, rotation_gt, tgt, rotation_pred, translation_pred)
        )
        infonce_loss = torch.mean(
            compute_infonce_loss_match_all_with_scale_select_negatives(
                rotation_gt,
                tgt,
                sat_points,
                grd_points,
                torch.ones_like(scale),
                sat_indices_sampled,
                grd_indices_sampled,
                matching_score_original,
                metric_coord_sat,
                bev_coord_grd,
                mask,
                settings['grid_size_h'],
            )
        )
        loss = distance_loss + args.beta * infonce_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()



def evaluate(
    model,
    feature_extractor,
    dataloader,
    device,
    args,
    settings,
    city_coords,
    theta,
    phi,
):
    model.eval()
    translation_error = []
    yaw_error = []
    scales = []

    with torch.no_grad():
        for data in dataloader:
            if data is None:
                continue

            grd, depth, sat, tgt, rotation_gt, cities, _ = data
            sat_size = sat.shape[-1]

            grd = grd.to(device)
            depth = depth.to(device)
            sat = sat.to(device)
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
                grd_feature,
                sat_feature,
                cities,
                city_coords,
                theta,
                phi,
                args.max_depth,
                settings['num_samples_matches'],
            )

            rotation_pred, translation_pred, scale = estimate_pose(sat_points, grd_points, weights)
            if translation_pred is None:
                print('Skipping batch: singular transformation matrix')
                continue

            scales.extend(scale.view(-1).cpu().tolist())
            translation_pred = (translation_pred / settings['grid_size_h']) * sat_size
            translation_error.extend(torch.norm(translation_pred - tgt, dim=-1).view(-1).cpu().tolist())
            yaw_error.extend(compute_yaw_errors(rotation_gt, rotation_pred))

    if not translation_error:
        raise RuntimeError('Validation produced no valid samples.')

    return {
        'translation_mean': float(np.mean(translation_error)),
        'translation_median': float(np.median(translation_error)),
        'yaw_mean': float(np.mean(yaw_error)),
        'yaw_median': float(np.median(yaw_error)),
        'scale_mean': float(np.mean(scales)),
        'scale_median': float(np.median(scales)),
    }


def save_metrics(results_dir, epoch, metrics):
    append_metric(
        results_dir,
        'Mean_distance_error.txt',
        epoch,
        metrics['translation_mean'],
        'Validation_set_mean_distance_error_in_pixels:',
    )
    append_metric(
        results_dir,
        'Median_distance_error.txt',
        epoch,
        metrics['translation_median'],
        'Validation_set_median_distance_error_in_pixels:',
    )
    append_metric(
        results_dir,
        'Mean_orientation_error.txt',
        epoch,
        metrics['yaw_mean'],
        'Validation_set_mean_yaw_error:',
    )
    append_metric(
        results_dir,
        'Median_orientation_error.txt',
        epoch,
        metrics['yaw_median'],
        'Validation_set_median_yaw_error:',
    )


def main():
    args = parse_args()
    config = load_config()
    set_seeds(config.getint('RandomSeed', 'seed'))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    args, settings = resolve_runtime_settings(args, config)
    train_dataloader, val_dataloader = create_dataloaders(
        settings['dataset_root'],
        args.area,
        args.batch_size,
        args.random_orientation,
    )

    label = build_experiment_label(args, settings)
    print(f'Experiment label: {label}')
    torch.cuda.empty_cache()

    feature_extractor = DinoExtractor().to(device)
    feature_extractor.eval()
    model = VigorCrossViewMatcher(
        device,
        sat_bev_res=settings['sat_bev_res'],
        embed_dim=1024,
        temperature=args.temperature,
    ).to(device)

    if args.epoch_to_resume > 0:
        model_path = CHECKPOINT_ROOT / label / str(args.epoch_to_resume - 1) / 'model.pt'
        model.load_state_dict(torch.load(model_path, map_location=device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))

    city_coords = create_city_coordinate_lookup(settings['sat_bev_res'], device)
    metric_coord4loss = create_metric_grid(args.loss_grid_size, settings['num_virtual_point'], 1, device)
    theta, phi = create_spherical_grids(settings['ground_image_size'], args.batch_size, device)
    results_dir = RESULTS_ROOT / label
    results_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epoch_to_resume, NUM_EPOCHS):
        train_epoch(
            model,
            feature_extractor,
            train_dataloader,
            optimizer,
            device,
            args,
            settings,
            city_coords,
            metric_coord4loss,
            theta,
            phi,
        )

        model_dir = CHECKPOINT_ROOT / label / str(epoch)
        model_dir.mkdir(parents=True, exist_ok=True)
        print(f'save checkpoint at {model_dir}')
        torch.save(model.state_dict(), model_dir / 'model.pt')

        print(f'Epoch {epoch} - Evaluating on validation set...')
        metrics = evaluate(
            model,
            feature_extractor,
            val_dataloader,
            device,
            args,
            settings,
            city_coords,
            theta,
            phi,
        )

        print('epoch:', epoch)
        print(f"Mean Translation Error: {metrics['translation_mean']:.3f}")
        print(f"Median Translation Error: {metrics['translation_median']:.3f}")
        print(f"Mean Yaw Error: {metrics['yaw_mean']:.3f}")
        print(f"Median Yaw Error: {metrics['yaw_median']:.3f}")

        save_metrics(results_dir, epoch, metrics)



if __name__ == '__main__':
    main()
