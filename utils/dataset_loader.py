import os
import os.path
import filetype


def check_image(filename):
    return filetype.is_image(filename)

def load_data():
    
    root = 'FlyingThings3D_subset'

    left_images_train = []
    right_images_train = []
    left_disps_train = []
    left_images_test = []
    right_images_test = []
    left_disps_test = []

    for i in ['train', 'val']:
        img_l = os.listdir(root + '/' + i + '/image_clean/left/')

        for img in img_l:
            if check_image(root + '/' + i + '/image_clean/left/' + img):
                if i == 'train':
                    left_images_train.append(root + '/' + i + '/image_clean/left/' + img)
                    left_disps_train.append(root + '/' + i + '/disparity/left/' + img.split(".")[0] + '.pfm')
                else:
                    left_images_test.append(root + '/' + i + '/image_clean/left/' + img)
                    left_disps_test.append(root + '/' + i + '/disparity/left/' + img.split(".")[0] + '.pfm')
            if check_image(root + '/' + i + '/image_clean/right/' + img):
                if i == 'train':
                    right_images_train.append(root + '/' + i + '/image_clean/right/' + img)
                else:
                    right_images_test.append(root + '/' + i + '/image_clean/right/' + img)
            

    return left_images_train, right_images_train, left_disps_train, left_images_test, right_images_test, left_disps_test


if __name__ == '__main__':
    lt, rt, ld, lte, rte, ldte = load_data()
    print(len(lt), len(rt), len(ld), len(lte), len(rte), len(ldte))