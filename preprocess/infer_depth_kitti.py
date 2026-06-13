import argparse
import glob
import os
import sys

import cv2
import matplotlib
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from external.depthanythingv2.metric_depth.depth_anything_v2.dpt import DepthAnythingV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='external/depthanythingv2/metric_depth/checkpoints/depth_anything_v2_metric_vkitti_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=80)
    
    parser.add_argument('--save-depth', dest='save_depth', action='store_true', help='save the model output as png')
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    
    parser.add_argument('--save-visualization', dest='save_visualization', action='store_true', help='save the visualization result')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**{**model_configs[args.encoder], 'max_depth': args.max_depth})
    depth_anything.load_state_dict(torch.load(args.load_from, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.img_path):
        if args.img_path.endswith('txt'):
            with open(args.img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [args.img_path]
    else:
        filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        raw_image = cv2.imread(filename)
        if raw_image is None:
            print(f'Unreadable image: {filename}')
            continue

        stem = os.path.splitext(os.path.basename(filename))[0]
        
        depth = depth_anything.infer_image(raw_image, args.input_size)
        depth = depth.astype(np.float32)
        depth = np.clip(depth, 0.0, args.max_depth)
        
        if args.save_depth:
            metric_depth = np.round(depth * 256.0).astype(np.uint16)

            metric_output_path = os.path.join(args.outdir, stem + '.png')
            cv2.imwrite(metric_output_path, metric_depth)

        if args.save_numpy:
            numpy_output_path = os.path.join(args.outdir, stem + '.npy')
            np.save(numpy_output_path, depth)

        if args.save_visualization:
            depth_min = float(depth.min())
            depth_max = float(depth.max())

            if depth_max > depth_min:
                depth_vis = (depth - depth_min) / (depth_max - depth_min) * 255.0
            else:
                depth_vis = np.zeros_like(depth, dtype=np.float32)

            depth_vis = depth_vis.astype(np.uint8)

            if args.grayscale:
                depth_vis = np.repeat(depth_vis[..., np.newaxis], 3, axis=-1)
            else:
                depth_vis = (cmap(depth_vis)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

            vis_output_path = os.path.join(args.outdir, stem + '.png')

            if args.pred_only:
                cv2.imwrite(vis_output_path, depth_vis)
            else:
                split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
                combined_result = cv2.hconcat([raw_image, split_region, depth_vis])
                cv2.imwrite(vis_output_path, combined_result)