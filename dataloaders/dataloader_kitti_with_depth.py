import os
import random

import cv2 as cv
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _set_deterministic_seed(seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(mode=True, warn_only=True)


_set_deterministic_seed()

Default_lat = 49.015
Satmap_zoom = 18

GrdImg_H = 364  
GrdImg_W = 1232  
GrdOriImg_H = 375
GrdOriImg_W = 1242
SatMap_original_sidelength = 512 
SatMap_process_sidelength = 504 

satmap_dir = 'satmap'
grdimage_dir = 'raw_data'
oxts_dir = 'oxts/data'  
left_color_camera_dir = 'image_02/data'
CameraGPS_shift_left = [1.08, 0.26]

satmap_transform = transforms.Compose([
    transforms.Resize(size=[SatMap_process_sidelength, SatMap_process_sidelength]),
    transforms.ToTensor()
])

grdimage_transform = transforms.Compose([
    transforms.Resize(size=[GrdImg_H, GrdImg_W]),
    transforms.ToTensor()
])

MAX_DEPTH_METERS = 40.0


def get_meter_per_pixel(lat=Default_lat, zoom=Satmap_zoom, scale=SatMap_process_sidelength / SatMap_original_sidelength):
    meter_per_pixel = 156543.03392 * np.cos(lat * np.pi / 180.0) / (2**zoom)
    meter_per_pixel /= 2
    meter_per_pixel /= scale
    return meter_per_pixel


def _load_metric_depth_file(path, max_depth=MAX_DEPTH_METERS):
    try:
        metric_depth = cv.imread(path, cv.IMREAD_UNCHANGED)
        if metric_depth is None:
            raise ValueError('cv2.imread returned None')
        metric_depth = metric_depth.astype(np.float32) / 256.0
        return np.clip(metric_depth, 0, max_depth)
    except Exception as e:
        print(f'Unreadable depth image: {path} ({e})')
        return None


def _read_left_camera_k(root, day_dir):
    calib_file_name = os.path.join(root, grdimage_dir, day_dir, 'calib_cam_to_cam.txt')
    with open(calib_file_name, 'r') as f:
        for line in f:
            if 'P_rect_02' not in line:
                continue
            items = line.split(':')
            values = items[1].strip().split(' ')
            fx = float(values[0]) * GrdImg_W / GrdOriImg_W
            cx = float(values[2]) * GrdImg_W / GrdOriImg_W
            fy = float(values[5]) * GrdImg_H / GrdOriImg_H
            cy = float(values[6]) * GrdImg_H / GrdOriImg_H
            left_camera_k = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            return torch.from_numpy(np.asarray(left_camera_k, dtype=np.float32))
    raise ValueError(f'P_rect_02 not found in {calib_file_name}')


def _load_ground_image(path, transform):
    with Image.open(path, 'r') as grd_image:
        ground_image = grd_image.convert('RGB')
        if transform is not None:
            ground_image = transform(ground_image)
    return ground_image


def _rotation_matrix_from_yaw(yaw):
    return torch.tensor(
        [[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]],
        dtype=torch.float32,
    )

class SatGrdDataset(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
        self.root = root

        self.meter_per_pixel = get_meter_per_pixel(scale=1)
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of pixels
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of pixels

        self.rotation_range = rotation_range  # in terms of degree

        if transform is not None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = 'raw_data'

        self.satmap_dir = satmap_dir

        with open(file, 'r') as f:
            file_name = f.readlines()

        self.file_name = [line.rstrip('\n') for line in file_name]

    def __len__(self):
        return len(self.file_name)


    def __getitem__(self, idx):
        # read cemera k matrix from camera calibration files, day_dir is first 10 chat of file name

        file_name = self.file_name[idx]
        day_dir = file_name[:10]
        drive_dir = file_name[:38]#2011_09_26/2011_09_26_drive_0002_sync/
        image_no = file_name[38:]

        # =================== read camera intrinsice for left and right cameras ====================
        left_camera_k = _read_left_camera_k(self.root, day_dir)
        
        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
            heading = float(content[5])

        left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                     image_no.lower())
        metric_depth_name = left_img_name.replace('raw_data', 'depth_anythingv2_depth').replace('image_02/data', 'depth')

        grd_img_left = _load_ground_image(left_img_name, self.grdimage_transform)
        depth = _load_metric_depth_file(metric_depth_name)
        depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        depth = F.interpolate(depth, size=(GrdImg_H, GrdImg_W), mode='nearest').squeeze(0)

        
        sat_rot = sat_map.rotate((-heading) / np.pi * 180) # make the east direction the vehicle heading
        sat_align_cam = sat_rot.transform(sat_rot.size, Image.AFFINE,
                                          (1, 0, CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=Image.BILINEAR) 
        # randomly generate shift
        gt_shift_x = np.random.uniform(-1, 1)  # --> right as positive, parallel to the heading direction
        gt_shift_y = np.random.uniform(-1, 1)  # --> up as positive, vertical to the heading direction
        
        sat_rand_shift = \
            sat_align_cam.transform(
                sat_align_cam.size, Image.AFFINE,
                (1, 0, gt_shift_x * self.shift_range_pixels_lon,
                 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                resample=Image.BILINEAR)
        
        # randomly generate roation
        random_ori = np.random.uniform(-1, 1) * self.rotation_range # 0 means the arrow in aerial image heading Easting, counter-clockwise increasing for satellite rotation
        
        sat_rand_shift_rand_rot = sat_rand_shift.rotate(random_ori)
        
        sat_map = TF.center_crop(sat_rand_shift_rand_rot, SatMap_original_sidelength)
        
        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        # location gt
        x_offset = int(gt_shift_x*self.shift_range_pixels_lon*np.cos(random_ori/180*np.pi) - gt_shift_y*self.shift_range_pixels_lat*np.sin(random_ori/180*np.pi)) # horizontal direction
        y_offset = int(-gt_shift_y*self.shift_range_pixels_lat*np.cos(random_ori/180*np.pi) - gt_shift_x*self.shift_range_pixels_lon*np.sin(random_ori/180*np.pi)) # vertical direction
        
        x_offset = x_offset/SatMap_original_sidelength*SatMap_process_sidelength
        y_offset = y_offset/SatMap_original_sidelength*SatMap_process_sidelength
        
        # orientation gt
        orientation_angle = 90 - random_ori # from ground ori (assuming heading north) to up in satellite clockwise increasing

        yaw = -orientation_angle * (np.pi/180)
        
        gt_loc = torch.tensor([[y_offset, x_offset]]) 
        gt_loc = -gt_loc # make it from grd to sat

        r = _rotation_matrix_from_yaw(yaw)
        
        return sat_map, grd_img_left, depth, left_camera_k, gt_loc, r
               
class SatGrdDatasetTest(Dataset):
    def __init__(self, root, file,
                 transform=None, shift_range_lat=20, shift_range_lon=20, rotation_range=10):
        self.root = root

        self.meter_per_pixel = get_meter_per_pixel(scale=1)
        self.shift_range_pixels_lat = shift_range_lat / self.meter_per_pixel  # shift range is in terms of meters
        self.shift_range_pixels_lon = shift_range_lon / self.meter_per_pixel  # shift range is in terms of meters

        self.rotation_range = rotation_range  # in terms of degree

        if transform is not None:
            self.satmap_transform = transform[0]
            self.grdimage_transform = transform[1]

        self.pro_grdimage_dir = 'raw_data'

        self.satmap_dir = satmap_dir

        with open(file, 'r') as f:
            file_name = f.readlines()

        self.file_name = [line.rstrip('\n') for line in file_name]
       

    def __len__(self):
        return len(self.file_name)


    def __getitem__(self, idx):

        line = self.file_name[idx]
        file_name, gt_shift_x, gt_shift_y, theta = line.split(' ')
        day_dir = file_name[:10]
        drive_dir = file_name[:38]
        image_no = file_name[38:]

        # =================== read camera intrinsice for left and right cameras ====================
        left_camera_k = _read_left_camera_k(self.root, day_dir)
        
        
        # =================== read satellite map ===================================
        SatMap_name = os.path.join(self.root, self.satmap_dir, file_name)
        with Image.open(SatMap_name, 'r') as SatMap:
            sat_map = SatMap.convert('RGB')

        oxts_file_name = os.path.join(self.root, grdimage_dir, drive_dir, oxts_dir,
                                      image_no.lower().replace('.png', '.txt'))
        with open(oxts_file_name, 'r') as f:
            content = f.readline().split(' ')
            heading = float(content[5])

        left_img_name = os.path.join(self.root, self.pro_grdimage_dir, drive_dir, left_color_camera_dir,
                                     image_no.lower())
        metric_depth_name = left_img_name.replace('raw_data', 'depth_anythingv2_depth').replace('image_02/data', 'depth')

        grd_img_left = _load_ground_image(left_img_name, self.grdimage_transform)
        depth = _load_metric_depth_file(metric_depth_name)
        depth = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        depth = F.interpolate(depth, size=(GrdImg_H, GrdImg_W), mode='nearest').squeeze(0)

        
        sat_rot = sat_map.rotate(-heading / np.pi * 180)
        
        sat_align_cam = sat_rot.transform(sat_rot.size, Image.AFFINE,
                                          (1, 0, CameraGPS_shift_left[0] / self.meter_per_pixel,
                                           0, 1, CameraGPS_shift_left[1] / self.meter_per_pixel),
                                          resample=Image.BILINEAR)
        
        # load the shifts 
        gt_shift_x = -float(gt_shift_x)  # --> right as positive, parallel to the heading direction
        gt_shift_y = -float(gt_shift_y)  # --> up as positive, vertical to the heading direction

        sat_rand_shift = \
            sat_align_cam.transform(
                sat_align_cam.size, Image.AFFINE,
                (1, 0, gt_shift_x * self.shift_range_pixels_lon,
                 0, 1, -gt_shift_y * self.shift_range_pixels_lat),
                resample=Image.BILINEAR)
        random_ori = float(theta) * self.rotation_range # degree

        sat_rand_shift_rand_rot = sat_rand_shift.rotate(random_ori)
        
        sat_map = TF.center_crop(sat_rand_shift_rand_rot, SatMap_original_sidelength)


        # transform
        if self.satmap_transform is not None:
            sat_map = self.satmap_transform(sat_map)

        # location gt
        x_offset = (gt_shift_x*self.shift_range_pixels_lon*np.cos(random_ori/180*np.pi) - gt_shift_y*self.shift_range_pixels_lat*np.sin(random_ori/180*np.pi)) # horizontal direction
        y_offset = (-gt_shift_y*self.shift_range_pixels_lat*np.cos(random_ori/180*np.pi) - gt_shift_x*self.shift_range_pixels_lon*np.sin(random_ori/180*np.pi)) # vertical direction
        
        x_offset = x_offset/SatMap_original_sidelength*SatMap_process_sidelength
        y_offset = y_offset/SatMap_original_sidelength*SatMap_process_sidelength

        gt_loc = torch.tensor([[y_offset, x_offset]], dtype=torch.float32)
        gt_loc = -gt_loc # make it from grd to sat

        # orientation gt
        orientation_angle = 90 - random_ori # from ground ori (assuming heading north) to satellite ori clockwise increasing

        yaw = -orientation_angle * (np.pi/180) 
        

        r = _rotation_matrix_from_yaw(yaw)
        
        return sat_map, grd_img_left, depth, left_camera_k, gt_loc, r