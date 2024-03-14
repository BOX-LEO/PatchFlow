import cv2
import skimage.exposure
import numpy as np
from numpy.random import default_rng
from einops import rearrange
from typing import List, Tuple
import imgaug.augmenters as iaa
from PIL import Image
import torch
import math
import random
import os


def lerp_np(x, y, w):
    fin_out = (y - x) * w + x
    return fin_out


# generate perlin noise given shape and resolution
def rand_perlin_2d_np(shape, res, fade=lambda t: 6 * t ** 5 - 15 * t ** 4 + 10 * t ** 3):
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1

    angles = 2 * math.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
    tt = np.repeat(np.repeat(gradients, d[0], axis=0), d[1], axis=1)

    tile_grads = lambda slice1, slice2: np.repeat(
        np.repeat(gradients[slice1[0]:slice1[1], slice2[0]:slice2[1]], d[0], axis=0), d[1], axis=1)
    dot = lambda grad, shift: (
            np.stack((grid[:shape[0], :shape[1], 0] + shift[0], grid[:shape[0], :shape[1], 1] + shift[1]),
                     axis=-1) * grad[:shape[0], :shape[1]]).sum(axis=-1)

    n00 = dot(tile_grads([0, -1], [0, -1]), [0, 0])
    n10 = dot(tile_grads([1, None], [0, -1]), [-1, 0])
    n01 = dot(tile_grads([0, -1], [1, None]), [0, -1])
    n11 = dot(tile_grads([1, None], [1, None]), [-1, -1])
    t = fade(grid[:shape[0], :shape[1]])
    return math.sqrt(2) * lerp_np(lerp_np(n00, n10, t[..., 0]), lerp_np(n01, n11, t[..., 0]), t[..., 1])


def rand_augment():
    augmenters = [
        iaa.GammaContrast((0.5, 2.0), per_channel=True),
        iaa.MultiplyAndAddToBrightness(mul=(0.8, 1.2), add=(-30, 30)),
        iaa.pillike.EnhanceSharpness(),
        iaa.AddToHueAndSaturation((-50, 50), per_channel=True),
        iaa.Solarize(0.5, threshold=(32, 128)),
        iaa.Posterize(),
        iaa.Invert(),
        iaa.pillike.Autocontrast(),
        iaa.pillike.Equalize(),
        iaa.Affine(rotate=(-45, 45))
    ]

    aug_idx = np.random.choice(np.arange(len(augmenters)), 3, replace=False)
    aug = iaa.Sequential([
        augmenters[aug_idx[0]],
        augmenters[aug_idx[1]],
        augmenters[aug_idx[2]]
    ])

    return aug


class AddAnomaly():
    def __init__(
            self, threshold: int = 50,
            resize: Tuple[int, int] = (768, 768),
            perlin_scale: int = 6,
            structure_grid_size: str = 8,
            min_perlin_scale: int = 0,
            perlin_noise_threshold: float = 0.5,
            transparency_range: List[float] = [0.15, 1.]):

        # synthetic anomaly
        self.transparency_range = transparency_range
        self.structure_grid_size = structure_grid_size
        self.perlin_scale = perlin_scale
        self.min_perlin_scale = min_perlin_scale
        self.perlin_noise_threshold = perlin_noise_threshold
        # transform ndarray into tensor
        self.resize = resize
        self.threshold = threshold

    def add_defect(self, image: np.ndarray):
        image = cv2.resize(image, dsize=self.resize)
        try:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            gray = True
        except:
            gray = False

        # step 1. generate mask

        # a. generate target foreground mask
        target_foreground_mask = self.get_foreground_mask(image=image)
        # cv2.imshow('target_foreground_mask',target_foreground_mask)
        # b. generate defect mask
        defect_mask = self.gen_defect_mask()
        # cv2.imshow('defect mask',defect_mask)
        # c. bitwise and both mask
        mask = cv2.bitwise_and(defect_mask, target_foreground_mask)

        # step 2. generate texture or structure anomaly

        # anomaly source
        anomaly_source_img = self.structure_source(image=image)
        # print('anomaly source image',anomaly_source_img.shape)
        # cv2.imshow('ano', anomaly_source_img)
        # # mask anomaly parts
        factor = np.random.uniform(*self.transparency_range, size=1)[0]
        # anomaly_source_img = factor * (mask_expanded * anomaly_source_img) + (1 - factor) * (mask_expanded * image)
        #
        # # step 3. blending image and anomaly source
        # anomaly_source_img = ((- mask_expanded + 1) * image) + anomaly_source_img

        anomaly_source_img = factor * cv2.bitwise_and(anomaly_source_img, anomaly_source_img, mask=mask) \
                             + (1 - factor) * cv2.bitwise_and(image, image, mask=mask)

        # step 3. blending image and anomaly source
        anomaly_source_img = cv2.bitwise_and(image, image, mask=cv2.bitwise_not(mask)) + anomaly_source_img
        if gray:
            anomaly_source_img = cv2.cvtColor(anomaly_source_img.astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # if gray:
        #     anomaly_source_img = cv2.cvtColor(anomaly_source_img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        return anomaly_source_img.astype(np.uint8), mask

    # define random seed to change the pattern
    def gen_defect_mask(self):
        # self.resize[0]and self.resize[1] are height and width of input image
        if np.random.randint(2) > 0:
            # Gaussian random
            seed_val = np.random.randint(10000)
            rng = default_rng(seed=seed_val)

            # create random noise image
            noise = rng.integers(0, 255, (self.resize[0], self.resize[1]), np.uint8, True)

            # blur the noise image to control the size
            blur = cv2.GaussianBlur(noise, (0, 0), sigmaX=15, sigmaY=15, borderType=cv2.BORDER_DEFAULT)

            # stretch the blurred image to full dynamic range
            stretch = skimage.exposure.rescale_intensity(blur, in_range='image', out_range=(0, 255)).astype(np.uint8)

            # threshold stretched image to control the size
            threshold = np.random.randint(195, 200)
            # print(threshold)
            thresh = cv2.threshold(stretch, threshold, 255, cv2.THRESH_BINARY)[1]

            # apply morphology open and close to smooth out and make 3 channels
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # _, mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY_INV)
            # print('gaussian', type(mask))
            # print('gaussian', np.shape(mask))
            # print(mask)
            # mask = mask.astype(bool).astype(int)
        else:
            # define perlin noise scale
            perlin_scalex = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(self.min_perlin_scale, self.perlin_scale, (1,)).numpy()[0])

            # generate perlin noise
            perlin_noise = rand_perlin_2d_np((self.resize[0], self.resize[1]), (perlin_scalex, perlin_scaley))

            # apply affine transform
            rot = iaa.Affine(rotate=(-90, 90))
            perlin_noise = rot(image=perlin_noise)

            # make a mask by applying threshold
            threshold = random.uniform(0.6, 0.7)
            # print(threshold)
            mask = np.where(
                perlin_noise > threshold,
                np.ones_like(perlin_noise),
                np.zeros_like(perlin_noise)
            )
            mask = mask * 255
            mask = mask.astype(np.uint8)
            # cv2.imshow('perlin mask',mask)
            # print('perlin', np.shape(mask))
            # print(mask)
        return mask

    def get_foreground_mask(self, image: np.ndarray):
        # convert RGB into GRAY scale
        img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if self.threshold >= 0:
            # generate binary mask of gray scale image
            _, target_foreground_mask = cv2.threshold(img_gray, self.threshold, 255, cv2.THRESH_BINARY)
        else:
            # generate binary mask of gray scale image
            _, target_foreground_mask = cv2.threshold(img_gray, -self.threshold, 255, cv2.THRESH_BINARY_INV)
        return target_foreground_mask

    def structure_source(self, image: np.ndarray) -> np.ndarray:
        structure_source_img = rand_augment()(image=image)
        W, H, _ = structure_source_img.shape
        # random crop 10 patches and swap the patches randomly
        patch_size = 400
        num_patches = 25
        rng = default_rng()
        patches = []
        position = []
        for _ in range(num_patches):
            x = rng.integers(0, W - patch_size)
            y = rng.integers(0, H - patch_size)
            position.append((x, y))
            patches.append(structure_source_img[x:x + patch_size, y:y + patch_size, :])
        # random shuffle the patches
        for i in range(num_patches):
            structure_source_img[position[i][0]:position[i][0] + patch_size,
                                 position[i][1]:position[i][1] + patch_size,:] \
                                 = patches[(i + 1) % num_patches]
        return structure_source_img


if __name__ == '__main__':
    addAnomaly = AddAnomaly()
    datapath = '/home/dao2/defect_detection/mvtec_anomaly_detection_v'
    categories = ['screw','bottle','metal_nut', 'pill', 'toothbrush','transistor','zipper','carpet','grid','leather','tile','wood','cable','capsule','hazelnut']

    # datapath = '/home/dao2/defect_detection/VisA/visa_pytorch'
    # categories = os.listdir(datapath)

    for category in categories:
        print(category)
        validation_path = os.path.join(datapath, category, 'val')
        if not os.path.exists(validation_path):
            os.makedirs(validation_path)
        output_iamge_path = os.path.join(validation_path, 'images')
        if not os.path.exists(output_iamge_path):
            os.makedirs(output_iamge_path)
        output_iamge_path = os.path.join(output_iamge_path, 'good')
        if not os.path.exists(output_iamge_path):
            os.makedirs(output_iamge_path)
        output_mask_path = os.path.join(validation_path, 'masks')
        if not os.path.exists(output_mask_path):
            os.makedirs(output_mask_path)
        output_mask_path = os.path.join(output_mask_path, 'good')
        if not os.path.exists(output_mask_path):
            os.makedirs(output_mask_path)


        if category in ['bottle', 'capsule', 'screw', 'zipper']:
            addAnomaly.threshold = -150
        elif category in ['carpet', 'grid', 'leather', 'tile', 'transister', 'wood']:
            addAnomaly.threshold = 0
        else:
            addAnomaly.threshold = 50


        # if category in ['candles', 'cashew', 'chewing_gum', 'fryum', 'macaroni2','pcb1']:
        #     addAnomaly.threshold = 150
        # elif category in ['macaroni1', 'pcb2', 'pcb3', 'pipe_fryum']:
        #     addAnomaly.threshold = 100
        # elif category in ['capsules']:
        #     addAnomaly.threshold = -150
        # elif category == 'pcb4':
        #     addAnomaly.threshold = 70


        train_path = os.path.join(datapath, category, 'train', 'good')
        # randomly select 20% of the training data as validation data
        train_images = os.listdir(train_path)
        # set the random seed
        random.seed(0)
        validation_images = random.sample(train_images, k=int(0.2 * len(train_images)))
        for image in validation_images:
            image_path = os.path.join(train_path, image)
            img = cv2.imread(image_path)
            H, W, _ = img.shape
            mask = np.zeros((H, W))
            while np.sum(mask) < 0.1 * H * W or np.sum(mask) > 0.5 * H * W:
                anomaly_img, mask = addAnomaly.add_defect(img)
            # show the image
            # cv2.imshow('anomaly_img', anomaly_img)
            cv2.imwrite(os.path.join(output_iamge_path, image), anomaly_img)
            cv2.imwrite(os.path.join(output_mask_path, image), mask)


