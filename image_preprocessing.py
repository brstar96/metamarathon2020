import cv2
from skimage.morphology import skeletonize, erosion, disk
import numpy as np
import os

"""
2020-03-05 sunyong 
"""

CROP_SIZE = 512
DATASET_PATH = 'dataset/'


class StaticVariable:
    counter = 0


def mask_image(src):
    img = src
    img_h = len(img)

    img_mask = np.zeros_like(img)
    img_mask[img_h-200:, :700] = True
    img_mask[img_h-180:, 1200:2000] = True

    background = np.full(img.shape, 255, dtype=np.uint8)
    img_masked = cv2.bitwise_xor(img, background, mask=img_mask)
    img_masked = cv2.bitwise_or(img, img_masked)

    return img_masked


def image_contrast_stretching(src):
    pixel_min = np.min(src)
    pixel_max = np.max(src)
    src = (src - pixel_min) / (pixel_max - pixel_min) * 255

    return np.array(src, dtype=np.uint8)


def image_pre_process(src, mode='normal'):
    img = cv2.medianBlur(src, 3)
    img = image_contrast_stretching(img)
    img = mask_image(img)

    if mode is 'pre':
        selem = disk(1)
        pre_img = erosion(img, selem)
        pre_img = cv2.threshold(pre_img, 230, 1, cv2.THRESH_BINARY_INV)[1]
        pre_img = skeletonize(pre_img).astype(np.uint8)
        pre_img = cv2.threshold(pre_img, 0, 255, cv2.THRESH_BINARY_INV)[1]
        img = pre_img

    return img


def mk_crop_image(src, output_dir, crop_iteration=1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = src

    img_w = len(img[0])
    img_h = len(img)

    for i in range(crop_iteration):
        ratio_x_max = 1 - (CROP_SIZE / img_w)
        ratio_y_max = 1 - (CROP_SIZE / img_h)

        while True:
            random_crop_area_x = np.random.random()
            random_crop_area_y = np.random.random()

            if random_crop_area_x < ratio_x_max and random_crop_area_y < ratio_y_max:
                break

        crop_x_start = int(img_w * random_crop_area_x)
        crop_x_end = int(img_w * random_crop_area_x + CROP_SIZE)
        crop_y_start = int(img_h * random_crop_area_y)
        crop_y_end = int(img_h * random_crop_area_y + CROP_SIZE)

        cropped_img = img[crop_y_start: crop_y_end, crop_x_start: crop_x_end]

        save_img = cropped_img

        file_name = str(StaticVariable.counter).zfill(5) + '.jpg'
        cv2.imwrite(output_dir + file_name, save_img)

        print(file_name + '\t successfully saved!')
        StaticVariable.counter += 1


def main():
    file_list = os.listdir(DATASET_PATH)

    for idx, fn in enumerate(file_list):
        img = cv2.imread(DATASET_PATH + fn, cv2.IMREAD_GRAYSCALE)
        img = image_pre_process(img, mode='normal')
        mk_crop_image(img, output_dir='cropped image/', crop_iteration=2)


if __name__ == '__main__':
    main()
