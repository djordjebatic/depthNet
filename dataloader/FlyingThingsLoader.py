from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from .python_pfm import *
from torchvision import transforms


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) 

def load_disparity(file_path):
    return readPFM(file_path)

def load_image(image_path):
    return Image.open(image_path).convert('RGB')

def preprocess_data(image, augment=False):

    if augment:
        print('TODO Augmentation')

    else:
        data_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])

    return data_transforms(image)

def FlyingThingsDataloader(Dataset):

    def __init__(self, left_images, right_images, left_disparities, train):
        self.left_images = left_images
        self.right_images = right_images
        self.left_disparities = left_disparities
        self.train = train

    def __getitem__(self, index):

        left_image_path = self.left_images[index]
        right_image_path = self.right_images[index]
        disparity_path = self.left_disparities[index]

        left_img = load_image(left_image_path)/256
        right_img = load_image(right_image_path)/256
        data, _ = readPFM(disparity_path)
        data = np.ascontiguousarray(data, dtype=np.float32)/256

        if self.train:

            left_img = preprocess_data(left_img)
            right_img = preprocess_data(right_img)

            return left_img, right_img, data
        else:

            left_img = preprocess_data(left_img)
            right_img = preprocess_data(right_img)

            return left_img, right_img, data

    def __len__(self):
        return len(self.left_image)


if __name__ == '__main__':

    im = load_image('sample_dataset/RGB_cleanpass/left/0006.png')
    print(im.size)
    data, scale = load_disparity("sample_dataset/disparity/0006.pfm")
    data = np.asarray(data)
    print(data.shape)

