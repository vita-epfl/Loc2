import os
from pathlib import Path
import ast
import configparser

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms


_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.ini"
config = configparser.ConfigParser()
config.read(_CONFIG_PATH)

ImageFile.LOAD_TRUNCATED_IMAGES = True

MAX_DEPTH_METERS = 35.0
CITY_RESOLUTIONS = {
    "NewYork": 0.113248,
    "Seattle": 0.100817,
    "SanFrancisco": 0.118141,
    "Chicago": 0.111262,
}


GROUND_IMAGE_SIZE = ast.literal_eval(config.get("VIGOR", "ground_image_size"))
SATELLITE_IMAGE_SIZE = ast.literal_eval(config.get("VIGOR", "satellite_image_size"))

transform_grd = transforms.Compose([
    transforms.Resize(GROUND_IMAGE_SIZE),
    transforms.ToTensor(),
])

transform_sat = transforms.Compose([
    transforms.Resize(SATELLITE_IMAGE_SIZE),
    transforms.ToTensor(),
])


class VIGORDataset(Dataset):
    def __init__(
        self,
        root='/home/ziminxia/Work/datasets/VIGOR',
        label_root='splits_new',
        split='samearea',
        train=True,
        random_orientation=0,
        transform=None,
    ):
        self.root = root
        self.label_root = label_root
        self.split = split
        self.train = train
        self.random_orientation = random_orientation
        self.grdimage_transform, self.satimage_transform = (
            transform if transform is not None else (transform_grd, transform_sat)
        )

        self.city_list = self._get_city_list()
        self.sat_list, self.sat_index_dict = self._load_satellite_data()
        self.grd_list, self.label, self.delta = self._load_ground_data()
        self.data_size = len(self.grd_list)

    def _get_city_list(self):
        if self.split == 'samearea':
            return ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        if self.split == 'crossarea':
            return ['NewYork', 'Seattle'] if self.train else ['SanFrancisco', 'Chicago']
        raise ValueError(f'Unsupported split: {self.split}')

    def _load_satellite_data(self):
        sat_list = []
        sat_index_dict = {}
        idx = 0

        for city in self.city_list:
            sat_list_fname = os.path.join(self.root, self.label_root, city, 'satellite_list.txt')
            with open(sat_list_fname, 'r') as file:
                for line in file:
                    sat_path = os.path.join(self.root, city, 'satellite', line.strip())
                    sat_list.append(sat_path)
                    sat_index_dict[line.strip()] = idx
                    idx += 1
            print(f'Loaded {sat_list_fname}, {idx} entries')

        return np.array(sat_list), sat_index_dict

    def _load_ground_data(self):
        grd_list, label_list, delta_list = [], [], []
        idx = 0

        for city in self.city_list:
            label_fname = self._get_label_file(city)

            with open(label_fname, 'r') as file:
                for line in file:
                    data = np.array(line.split())
                    label = np.array([self.sat_index_dict[data[i]] for i in [1, 4, 7, 10]]).astype(int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(np.float32)

                    grd_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    label_list.append(label)
                    delta_list.append(delta)
                    idx += 1

            print(f'Loaded {label_fname}, {idx} entries')

        return grd_list, np.array(label_list), np.array(delta_list)

    def _get_label_file(self, city):
        if self.split == 'samearea':
            return os.path.join(
                self.root,
                self.label_root,
                city,
                'same_area_balanced_train.txt' if self.train else 'same_area_balanced_test.txt',
            )
        return os.path.join(self.root, self.label_root, city, 'pano_label_balanced.txt')

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        try:
            grd = self._load_image(self.grd_list[idx])
            if grd is None:
                raise ValueError('Ground image is None')
            grd = self.grdimage_transform(grd)

            depth_path = os.path.splitext(self.grd_list[idx].replace('panorama', 'unik3d_depth'))[0] + '.npy'
            depth = self._load_metric_depth(depth_path)
            if depth is None:
                raise ValueError('Depth image is None')
            depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
            depth = F.interpolate(depth, size=GROUND_IMAGE_SIZE, mode='nearest').squeeze(0)

            rotation = np.random.uniform(-self.random_orientation / 360, self.random_orientation / 360)
            grd_rolled = torch.roll(grd, int(round(rotation * grd.size(2))), dims=2)
            depth_rolled = torch.roll(depth, int(round(rotation * depth.size(2))), dims=2)

            yaw = -rotation * 360 * (np.pi / 180)

            sat, row_offset, col_offset, raw_height = self._load_satellite_image(idx)
            if sat is None:
                raise ValueError('Satellite image is None')

            gt_loc = torch.tensor([[-row_offset, col_offset]])
            gt_loc = -gt_loc
            yaw = -yaw
            rotation_matrix = torch.tensor(
                [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
                dtype=torch.float32,
            )

            city, resolution = self._get_city_info(self.grd_list[idx], raw_height)
            return grd_rolled, depth_rolled, sat, gt_loc, rotation_matrix, city, resolution

        except Exception as e:
            print(f"[Warning] Skipping sample {idx} due to error: {e}")
            return None

    def _load_image(self, path):
        try:
            with Image.open(path) as image:
                return image.convert('RGB')
        except Exception as e:
            print(f'Unreadable image: {path} ({e})')
            return None

    def _load_metric_depth(self, path, max_depth=MAX_DEPTH_METERS):
        try:
            depth = np.load(path)
            depth = np.clip(depth, 0, max_depth)
            if np.all(depth == max_depth):
                print('all depth are larger than defined max depth')
                return None
            return depth
        except Exception as e:
            print(f'Unreadable depth image: {path} ({e})')
            return None

    def _load_satellite_image(self, idx):
        try:
            pos_index = 0
            row_offset, col_offset = self.delta[idx, pos_index]
            sat = self._load_image(self.sat_list[self.label[idx][pos_index]])
            raw_width, raw_height = sat.size
            sat = self.satimage_transform(sat)

            row_offset = row_offset / raw_height * sat.size(1)
            col_offset = col_offset / raw_width * sat.size(2)
            return sat, row_offset, col_offset, raw_height
        except Exception as e:
            print(f'[Warning] Failed to load satellite image at index {idx}: {e}')
            return None, None, None, None

    def _get_city_info(self, path, raw_height):
        for city, resolution in CITY_RESOLUTIONS.items():
            if city in path:
                return city, resolution * raw_height / SATELLITE_IMAGE_SIZE[0]
        return 'Unknown', 1.0
