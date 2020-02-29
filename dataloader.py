import os
from PIL import Image
import torchvision as vision

def custom_transforms(args):
    data_transforms = {
        'train': vision.transforms.Compose([
            vision.transforms.Resize((args.img_size, args.img_size)),
            # vision.transforms.RandomHorizontalFlip(),
            # vision.transforms.RandomRotation(10),
            vision.transforms.ToTensor(),
            vision.transforms.Normalize(
                # ImageNet default parameter:
                # [0.485, 0.456, 0.406],
                # [0.485, 0.456, 0.406],
                [0.,],
                [1.,])
        ]),
        # 'test'는 reference와 query 이미지에 대해 수행됩니다.
        'test': vision.transforms.Compose([
            vision.transforms.Resize((args.img_size, args.img_size)),
            vision.transforms.ToTensor(),
            vision.transforms.Normalize(
                [0.,],
                [1.,])
        ]),
    }
    return data_transforms

# def preprocessing(data, args):
#     resized_arr_list = []
#
#     # resize array to 1024 X 1024
#     for array in data:
#         resized_arr_list.append(cv2.resize(array, dsize=(args.img_size, args.img_size), interpolation=cv2.INTER_AREA))
#     resized_X = np.array(resized_arr_list)
#     resized_X = np.expand_dims(resized_X, axis=3)
#
#     # minmax scaling
#     print('Start minmax scaling...')
#     minmax_scaled_X = (resized_X - resized_X.min(axis=0)) / (resized_X.max(axis=0) - resized_X.min(axis=0)).astype(np.float32)
#
#     print('Preprocessing complete...')
#
#     return minmax_scaled_X

class metamatathonDataset():
    def __init__(self, args, transforms=None):
        self.train_data_path = args.DATASET_PATH
        self.transforms = transforms
        self.img_list = []

        for img_path in os.listdir(self.train_data_path):
            if '.png' in img_path:
                self.img_list.append(self.train_data_path+img_path)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        image = Image.open(self.img_list[idx]).convert("L")
        if self.transforms:
            image = self.transforms(image)
            return image
        else:
            return image