import cv2
from skimage.morphology import skeletonize, erosion, disk
import numpy as np
import os

"""
2020-03-05 sunyong 
"""

CROP_SIZE = 512


def image_contrast_stretching(src):
    pixel_min = np.min(src)
    pixel_max = np.max(src)
    src = (src - pixel_min) / (pixel_max - pixel_min) * 255

    return np.array(src, dtype=np.uint8)


def mk_crop_image(input_dir, output_dir, crop_iteration=1, mode='normal'):
    file_list = os.listdir(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    counter = 0
    for idx, fn in enumerate(file_list):
        img = cv2.imread(input_dir + fn, cv2.IMREAD_GRAYSCALE)
        img_w = len(img[0])
        img_h = len(img)

        img = cv2.medianBlur(img, 3)
        img = image_contrast_stretching(img)

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

            if mode is 'pre':
                selem = disk(1)
                pre_img = erosion(cropped_img, selem)
                pre_img = cv2.threshold(pre_img, 230, 1, cv2.THRESH_BINARY_INV)[1]
                pre_img = skeletonize(pre_img).astype(np.uint8)
                pre_img = cv2.threshold(pre_img, 0, 255, cv2.THRESH_BINARY_INV)[1]
                save_img = pre_img

            file_name = str(counter).zfill(5) + '.jpg'
            cv2.imwrite(output_dir + file_name, save_img)

            print(file_name + '\t successfully saved!')
            counter += 1


def main():
    mk_crop_image(input_dir='dataset/',
                  output_dir='cropped image/',
                  crop_iteration=2,
                  mode='normal')


if __name__ == '__main__':
    main()
