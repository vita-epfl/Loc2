import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from PIL import Image

from external.unik3d.unik3d.models import UniK3D
from external.unik3d.unik3d.utils.camera import (
    MEI,
    OPENCV,
    BatchCamera,
    Fisheye624,
    Pinhole,
    Spherical,
)
from external.unik3d.unik3d.utils.visualization import save_file_ply


def save(rgb, outputs, name, base_path, save_map=False, save_pointcloud=False):
    os.makedirs(base_path, exist_ok=True)

    points_torch = outputs["points"]  # [B, 3, H, W]

    points_np = points_torch.detach().cpu().numpy()
    depth_vigor = points_np[0, :, :, :]
    depth_vigor = np.linalg.norm(depth_vigor, axis=0)

    if save_map:
        depth_map = (np.clip(depth_vigor, 0.0, 100.0) * 500.0).astype(np.uint16)
        Image.fromarray(depth_map).save(os.path.join(base_path, f"{name}.png"))
        # np.save(os.path.join(base_path, f"{name}.npy"), depth_vigor)

    if save_pointcloud:
        predictions_3d = points_torch.permute(0, 2, 3, 1).reshape(-1, 3).detach().cpu().numpy()
        rgb_np = rgb.permute(1, 2, 0).reshape(-1, 3).detach().cpu().numpy()
        save_file_ply(predictions_3d, rgb_np, os.path.join(base_path, f"{name}.ply"))


def is_image_file(filename: str) -> bool:
    valid_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return Path(filename).suffix.lower() in valid_suffixes


def infer(model, args):
    camera = None
    camera_path = args.camera_path
    if camera_path is not None:
        with open(camera_path, "r") as f:
            camera_dict = json.load(f)

        params = torch.tensor(camera_dict["params"])
        camera_name = camera_dict["name"]
        assert camera_name in ["Fisheye624", "Spherical", "OPENCV", "Pinhole", "MEI"]
        camera = eval(camera_name)(params=params)

    if os.path.isfile(args.input):
        total_files = 1
        print(f"[1/{total_files}] Processing: {args.input}")

        rgb_path = args.input
        rgb = np.array(Image.open(rgb_path))
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)

        with torch.no_grad():
            outputs = model.infer(rgb=rgb_torch, camera=camera, normalize=True, rays=None)

        name = os.path.splitext(os.path.basename(rgb_path))[0]
        save(
            rgb_torch,
            outputs,
            name=name,
            base_path=args.output,
            save_map=args.save,
            save_pointcloud=args.save_ply,
        )

        print(f"[1/{total_files}] Done: {name}")

    elif os.path.isdir(args.input):
        rgb_fns = sorted(
            [fn for fn in os.listdir(args.input) if is_image_file(fn)]
        )
        total_files = len(rgb_fns)

        if total_files == 0:
            raise ValueError(f"No image files found in directory: {args.input}")

        for idx, rgb_fn in enumerate(rgb_fns, start=1):
            rgb_path = os.path.join(args.input, rgb_fn)
            print(f"[{idx}/{total_files}] Processing: {rgb_fn}")

            try:
                rgb = np.array(Image.open(rgb_path))
            except Exception as e:
                print(f"Error occurred while opening {rgb_path}: {e}")
                continue

            rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)

            with torch.no_grad():
                outputs = model.infer(rgb=rgb_torch, camera=camera, normalize=True, rays=None)

            name = os.path.splitext(rgb_fn)[0]
            save(
                rgb_torch,
                outputs,
                name=name,
                base_path=args.output,
                save_map=args.save,
                save_pointcloud=args.save_ply,
            )
        print(f"Finished processing {total_files}/{total_files} files.")

    else:
        raise ValueError("Input path is not valid.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script", conflict_handler="resolve")
    parser.add_argument("--input", type=str, required=True, help="Path to input image/folder.")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory.")
    parser.add_argument(
        "--config-file",
        type=str,
        default="./external/unik3d/configs/eval/vitl.json",
        help="Path to config file.",
    )
    parser.add_argument(
        "--camera-path",
        type=str,
        default="./external/unik3d/assets/demo/equirectangular.json",
        help="Path to camera parameters json file.",
    )
    parser.add_argument("--save", action="store_true", help="Save outputs as .npy.")
    parser.add_argument("--save-ply", action="store_true", help="Save pointcloud as ply.")
    parser.add_argument(
        "--resolution-level",
        type=int,
        default=9,
        help="Resolution level in [0,10).",
        choices=list(range(10)),
    )
    parser.add_argument(
        "--interpolation-mode",
        type=str,
        default="bilinear",
        help="Output interpolation.",
        choices=["nearest", "nearest-exact", "bilinear"],
    )
    args = parser.parse_args()

    print("Torch version:", torch.__version__)
    version = args.config_file.split("/")[-1].split(".")[0]
    name = f"unik3d-{version}"
    model = UniK3D.from_pretrained(f"lpiccinelli/{name}")

    model.resolution_level = args.resolution_level
    model.interpolation_mode = args.interpolation_mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    infer(model, args)