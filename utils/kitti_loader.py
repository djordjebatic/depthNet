import os
import os.path
import filetype


def check_image(filename):
    return filetype.is_image(filename)


def kitti_loader(file_path, mode):

    left_folder  = mode + '/image_2/'
    right_folder = mode + '/image_3/'
    disp_left = mode + '/disp_occ_0/'

    images = [img for img in os.listdir(file_path+ '/' + left_folder) if check_image(file_path+ '/' + left_folder + '/' + img)]

    training_images = images[:300]
    val_images = images[300:]

    left_images_train = [file_path+ '/' + left_folder + image for image in training_images]
    right_images_train = [file_path+ '/' + right_folder + image for image in training_images]
    left_disps_train = [file_path+ '/' + disp_left + image for image in training_images]
    left_images_test = [file_path+ '/' + left_folder + image for image in val_images]
    right_images_test = [file_path+ '/' + right_folder + image for image in val_images]
    left_disps_test = [file_path+ '/' + disp_left + image for image in val_images]

    return left_images_train, right_images_train, left_disps_train, left_images_test, right_images_test, left_disps_test

if __name__ == '__main__':
    kitti_loader('data_scene_flow', 'training')





