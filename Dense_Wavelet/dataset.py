import os
import numpy as np
import cv2
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
# import noise_generator
import utils
import matplotlib
import time
import scipy.io
import math

class RandomCrop(object):
    def __init__(self, image_size, crop_size):
        self.ch, self.cw = crop_size
        ih, iw = image_size

        self.h1 = random.randint(0, ih - self.ch)
        self.w1 = random.randint(0, iw - self.cw)

        self.h2 = self.h1 + self.ch
        self.w2 = self.w1 + self.cw

    def __call__(self, img):
        if len(img.shape) == 3:
            return img[self.h1: self.h2, self.w1: self.w2, :]
        else:
            return img[self.h1: self.h2, self.w1: self.w2]


class Noise2CleanDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.in_root = opt.in_root
        self.gt_root = opt.gt_root

        # randomly ignore 4 images from each directory to speed up training
        ignore = 4

        imgs = []
        target = []

        # Get sorted lists to ensure correspondence (assuming paired data)
        in_dirs = sorted([os.path.join(self.in_root, fname) for fname in os.listdir(self.in_root) if os.path.isdir(os.path.join(self.in_root, fname))])
        in_files = []
        for i, in_dir in enumerate(in_dirs):
            files = [os.path.join(in_dir, fname) for fname in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, fname))]
            files = sorted(files)
            if len(files) > ignore:
                files_to_keep = files.copy()
                files_to_ignore = random.sample(files_to_keep, ignore)
                files = [f for f in files_to_keep if f not in files_to_ignore]
            in_files.extend(files)
        # Attempt to align in_files with gt_files by filename match
        if in_files == []:
            print("files not stored in subdirectories")
            in_files = sorted([os.path.join(self.in_root, fname) for fname in os.listdir(self.in_root) if os.path.isfile(os.path.join(self.in_root, fname))])

        in_basenames = [os.path.basename(f) for f in in_files]
        gt_files_unsorted = [os.path.join(self.gt_root, fname) for fname in os.listdir(self.gt_root) if os.path.isfile(os.path.join(self.gt_root, fname))]

        # For each in_file, try to find corresponding gt_file by matching filename (basename)
        gt_dict = {os.path.basename(f): f for f in gt_files_unsorted}
        matched_in_files = []
        matched_gt_files = []

        for in_file, base_name in zip(in_files, in_basenames):
            if base_name in gt_dict:
                matched_in_files.append(in_file)
                matched_gt_files.append(gt_dict[base_name])
            else:
                print(f"Warning: No matching GT file found for {in_file} ({base_name})")

        print(len(matched_in_files), len(matched_gt_files))

        in_files = matched_in_files
        gt_files = matched_gt_files

        # Ensure same number of input and gt images
        # assert len(in_files) == len(gt_files), "Mismatch between in_root and gt_root file counts"

        imgs.extend(in_files)
        target.extend(gt_files)

        self.imgs = imgs
        self.target = target
        self.ignore = ignore

        # generate random number
    def random_crop_start(self, h, w, crop_size, min_divide):
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w

    def img_aug(self, noise, clean):
        # random rotate
        if self.opt.angle_aug:
            # rotate
            rotate = random.randint(0, 3)
            if rotate != 0:
                noise = np.rot90(noise, rotate)
                clean = np.rot90(clean, rotate)
            # horizontal flip
            if np.random.random() >= 0.5:
                noise = cv2.flip(noise, flipCode = 1)
                clean = cv2.flip(clean, flipCode = 1)
        return noise, clean
    def NormMinandMax(self, npdarr, min=-1, max=1):

        arr = npdarr.flatten()
        Ymax = np.max(arr)  
        Ymin = np.min(arr)  
        k = (max - min) / (Ymax - Ymin)
        last = min + k * (npdarr - Ymin)

        return last
    def img_sharpen(self, img):

        kernel_sharpen = np.array([
                [-1,-1,-1],
                [-1,9,-1],
                [-1,-1,-1]])

        # kernel_sharpen = np.array([
        #         [-1,-1,-1,-1,-1],
        #         [-1,2,2,2,-1],
        #         [-1,2,8,2,-1],
        #         [-1,2,2,2,-1], 
        #         [-1,-1,-1,-1,-1]])/8.0

        output = cv2.filter2D(img,-1,kernel_sharpen)
        return output

    # def __getitem__(self, index):
    #     # Define path
    #     noise_path= self.imgs[index]
    #     clean_path=self.target[index]
    #     # Read images
    #     # input
        
    #     noise_r = Image.open(noise_path).convert('RGB')
    #     noise_r = np.array(noise_r).astype(np.float32)
    #     h, w = noise_r.shape[:2]
    #     # print(h,w)
    #     rand_h, rand_w = self.random_crop_start(h, w, self.opt.crop_size, 4)
    #     noise_r = noise_r[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size]
         
    #     # noise_r = self.NormMinandMax(noise_r, -1, 1)
    #     noise_r = (noise_r - 128) / 128
    #     # noise_r = (noise_r) / 255.0
    #     # output
    #     clean = Image.open(clean_path).convert('RGB')
    #     clean = np.array(clean).astype(np.float32)
    #     clean = clean[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size]
    #     # clean = self.img_sharpen(clean)
    #     # clean = self.NormMinandMax(clean, -1, 1)
    #     clean = (clean - 128) / 128
    #     # clean = (clean) / 255.0
    #     # noise_s, noise_level_map = noise_generator.Poisson_Gaussian_random(clean)
    #     # noise_s = np.array(noise_s).astype(np.float32)
    #     # noise_level_map = np.array(noise_level_map).astype(np.float32)
    #     # noise_s = np.clip(noise_s, -1, 1)
    #     # noise_level_map = np.clip(noise_level_map, -1, 1)
    #     # noise_level_map = self.NormMinandMax(noise_level_map, -1, 1)
    #     noise_r = torch.from_numpy(noise_r.transpose(2, 0, 1).astype(np.float32)).contiguous()
    #     # noise_s = torch.from_numpy(noise_s.transpose(2, 0, 1).astype(np.float32)).contiguous()
    #     clean = torch.from_numpy(clean.transpose(2, 0, 1).astype(np.float32)).contiguous()
    #     # noise_level_map = torch.from_numpy(noise_level_map.transpose(2, 0, 1).astype(np.float32)).contiguous()
    #     return noise_r, clean# noise_s, noise_level_map

    def __getitem__(self, index):
        # Define path
        noise_path= self.imgs[index]
        if len(self.target) < len(self.imgs):
            clean_path = self.target[math.floor(index/(12-self.ignore))]
        else:
            clean_path = self.target[index]
        
        noise_r = Image.open(noise_path)
        noise_r_np = np.array(noise_r).astype(np.float32)
        # Normalize to 0-255 range for display
        min_val, max_val = noise_r_np.min(), noise_r_np.max()
        noise_r = ((noise_r_np - min_val) / (max_val - min_val))
        noise_r = np.array(noise_r).astype(np.float32)

        h, w = noise_r.shape[:2]

        rand_h, rand_w = self.random_crop_start(h, w, self.opt.crop_size, 4)
        noise_r = noise_r[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size]
         
        clean = Image.open(clean_path)
        clean_r_np = np.array(clean).astype(np.float32)
        # Normalize to 0-255 range for display
        min_val, max_val = clean_r_np.min(), clean_r_np.max()
        clean_r = ((clean_r_np - min_val) / (max_val - min_val))
        clean = np.array(clean_r).astype(np.float32)
        clean = clean[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size]

        noise_r = torch.from_numpy(noise_r.astype(np.float32)).contiguous().unsqueeze(0)
        clean = torch.from_numpy(clean.astype(np.float32)).contiguous().unsqueeze(0)

        return noise_r, clean
    
    def __len__(self):
        return len(self.imgs)

class ValDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.in_root = opt.val_root
        self.gt_root = opt.gtval_root

        # randomly ignore 4 images from each directory to speed up training
        ignore = 4

        imgs = []
        target = []

        # Get sorted lists to ensure correspondence (assuming paired data)
        in_dirs = sorted([os.path.join(self.in_root, fname) for fname in os.listdir(self.in_root) if os.path.isdir(os.path.join(self.in_root, fname))])
        in_files = []
        for i, in_dir in enumerate(in_dirs):
            files = [os.path.join(in_dir, fname) for fname in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, fname))]
            files = sorted(files)
            if len(files) > ignore:
                files_to_keep = files.copy()
                files_to_ignore = random.sample(files_to_keep, ignore)
                files = [f for f in files_to_keep if f not in files_to_ignore]
            in_files.extend(files)
        gt_files = sorted([os.path.join(self.gt_root, fname) for fname in os.listdir(self.gt_root) if os.path.isfile(os.path.join(self.gt_root, fname))])

        # Ensure same number of input and gt images
        # assert len(in_files) == len(gt_files), "Mismatch between in_root and gt_root file counts"

        imgs.extend(in_files)
        target.extend(gt_files)

        self.imgs = imgs
        self.target = target
        self.ignore = ignore

        # generate random number
    def random_crop_start(self, h, w, crop_size, min_divide):
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w

    def img_aug(self, noise, clean):
        # random rotate
        if self.opt.angle_aug:
            # rotate
            rotate = random.randint(0, 3)
            if rotate != 0:
                noise = np.rot90(noise, rotate)
                clean = np.rot90(clean, rotate)
            # horizontal flip
            if np.random.random() >= 0.5:
                noise = cv2.flip(noise, flipCode = 1)
                clean = cv2.flip(clean, flipCode = 1)
        return noise, clean
    
    def NormMinandMax(self, npdarr, min=-1, max=1):

        arr = npdarr.flatten()
        Ymax = np.max(arr)  
        Ymin = np.min(arr)  
        k = (max - min) / (Ymax - Ymin)
        last = min + k * (npdarr - Ymin)

        return last

    def img_sharpen(self, img):

        kernel_sharpen = np.array([
                [-1,-1,-1],
                [-1,9,-1],
                [-1,-1,-1]])

        # kernel_sharpen = np.array([
        #         [-1,-1,-1,-1,-1],
        #         [-1,2,2,2,-1],
        #         [-1,2,8,2,-1],
        #         [-1,2,2,2,-1], 
        #         [-1,-1,-1,-1,-1]])/8.0

        output = cv2.filter2D(img,-1,kernel_sharpen)
        return output

    def __getitem__(self, index):
        # Define path
        noise_path= self.imgs[index]
        clean_path = self.target[math.floor(index/(12-self.ignore))]
        
        noise_r = Image.open(noise_path)
        noise_r_np = np.array(noise_r).astype(np.float32)
        # Normalize to 0-255 range for display
        min_val, max_val = noise_r_np.min(), noise_r_np.max()
        noise_r = ((noise_r_np - min_val) / (max_val - min_val))
        noise_r = np.array(noise_r).astype(np.float32)

        h, w = noise_r.shape[:2]

        rand_h, rand_w = self.random_crop_start(h, w, self.opt.crop_size, 4)
        noise_r = noise_r[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size]
         
        clean = Image.open(clean_path)
        clean_r_np = np.array(clean).astype(np.float32)
        # Normalize to 0-255 range for display
        min_val, max_val = clean_r_np.min(), clean_r_np.max()
        clean_r = ((clean_r_np - min_val) / (max_val - min_val))
        clean = np.array(clean_r).astype(np.float32)
        clean = clean[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size]

        noise_r = torch.from_numpy(noise_r.astype(np.float32)).contiguous().unsqueeze(0)
        clean = torch.from_numpy(clean.astype(np.float32)).contiguous().unsqueeze(0)
        return noise_r, clean
    
    def __len__(self):
        return len(self.imgs)

class TestDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.in_root = opt.test_root
        self.gt_root = opt.gttest_root

        imgs = []
        target = []

        # Get sorted lists to ensure correspondence (assuming paired data)
        in_files = sorted([os.path.join(self.in_root, fname) for fname in os.listdir(self.in_root) if os.path.isfile(os.path.join(self.in_root, fname))])
        gt_files = sorted([os.path.join(self.gt_root, fname) for fname in os.listdir(self.gt_root) if os.path.isfile(os.path.join(self.gt_root, fname))])

        # Ensure same number of input and gt images
        assert len(in_files) == len(gt_files), "Mismatch between in_root and gt_root file counts"

        imgs.extend(in_files)
        target.extend(gt_files)

        self.imgs = imgs
        self.target = target

        # generate random number
    def random_crop_start(self, h, w, crop_size, min_divide):
        rand_h = random.randint(0, h - crop_size)
        rand_w = random.randint(0, w - crop_size)
        rand_h = (rand_h // min_divide) * min_divide
        rand_w = (rand_w // min_divide) * min_divide
        return rand_h, rand_w

    def img_aug(self, noise, clean):
        # random rotate
        if self.opt.angle_aug:
            # rotate
            rotate = random.randint(0, 3)
            if rotate != 0:
                noise = np.rot90(noise, rotate)
                clean = np.rot90(clean, rotate)
            # horizontal flip
            if np.random.random() >= 0.5:
                noise = cv2.flip(noise, flipCode = 1)
                clean = cv2.flip(clean, flipCode = 1)
        return noise, clean
    
    def NormMinandMax(self, npdarr, min=-1, max=1):

        arr = npdarr.flatten()
        Ymax = np.max(arr)  
        Ymin = np.min(arr)  
        k = (max - min) / (Ymax - Ymin)
        last = min + k * (npdarr - Ymin)

        return last

    def img_sharpen(self, img):

        kernel_sharpen = np.array([
                [-1,-1,-1],
                [-1,9,-1],
                [-1,-1,-1]])

        # kernel_sharpen = np.array([
        #         [-1,-1,-1,-1,-1],
        #         [-1,2,2,2,-1],
        #         [-1,2,8,2,-1],
        #         [-1,2,2,2,-1], 
        #         [-1,-1,-1,-1,-1]])/8.0

        output = cv2.filter2D(img,-1,kernel_sharpen)
        return output

    def __getitem__(self, index):
        # Define path
        noise_path= self.imgs[index]
        clean_path=self.target[index]
        
        noise_r = Image.open(noise_path)
        noise_r_np = np.array(noise_r).astype(np.float32)
        # Normalize to 0-255 range for display
        min_val, max_val = noise_r_np.min(), noise_r_np.max()
        noise_r = ((noise_r_np - min_val) / (max_val - min_val))
        noise_r = np.array(noise_r).astype(np.float32)

        h, w = noise_r.shape[:2]

        rand_h, rand_w = self.random_crop_start(h, w, self.opt.crop_size, 4)
        noise_r = noise_r[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size]
         
        clean = Image.open(clean_path)
        clean_r_np = np.array(clean).astype(np.float32)
        # Normalize to 0-255 range for display
        min_val, max_val = clean_r_np.min(), clean_r_np.max()
        clean_r = ((clean_r_np - min_val) / (max_val - min_val))
        clean = np.array(clean_r).astype(np.float32)
        clean = clean[rand_h:rand_h+self.opt.crop_size, rand_w:rand_w+self.opt.crop_size]

        noise_r = torch.from_numpy(noise_r.astype(np.float32)).contiguous().unsqueeze(0)
        clean = torch.from_numpy(clean.astype(np.float32)).contiguous().unsqueeze(0)
        return noise_r, clean
    
    def __len__(self):
        return len(self.imgs)