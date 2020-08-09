import os
import os.path
import filetype
import numpy as np

def check_image(filename):
    return filetype.is_image(filename)

def load_data(root = 'FlyingThings3D_subset' ):


    left_images_train = []
    right_images_train = []
    left_disps_train = []
    left_images_test = []
    right_images_test = []
    left_disps_test = []

    if root =='FlyingThings3D_subset':
        for count, i in enumerate(['train', 'val']):

            img_l = os.listdir(root + os.sep + i + os.sep + 'image_clean' + os.sep + 'left' + os.sep)

            for img in img_l:
                if check_image(root + os.sep + i + os.sep + 'image_clean' + os.sep + 'left' + os.sep + img):
                    if i == 'train':
                        left_images_train.append(root + os.sep + i + os.sep + 'image_clean' + os.sep + 'left' + os.sep + img)
                        left_disps_train.append(root + os.sep + i + os.sep +'disparity' + os.sep + 'left' + os.sep + img.split(".")[0] + '.pfm')
                    else:
                        left_images_test.append(root + os.sep + i + os.sep + 'image_clean' + os.sep + 'left' + os.sep + img)
                        left_disps_test.append(root + os.sep + i + os.sep + 'disparity' + os.sep + 'left' + os.sep + img.split(".")[0] + '.pfm')
                if check_image(root + os.sep + i + os.sep + 'image_clean' + os.sep + 'right' + os.sep + img):
                    if i == 'train':
                        right_images_train.append(root + os.sep + i + os.sep + 'image_clean' + os.sep + 'right' + os.sep + img)
                    else:
                        right_images_test.append(root + os.sep + i + os.sep + 'image_clean' + os.sep + 'right' + os.sep + img)

            count += 1

    elif root == 'KITTI':
        left_folder  = 'image_2/'
        right_folder = 'image_3/'
        disp_left = 'disp_occ_0/'

        images = [img for img in os.listdir('data_scene_flow/training/' + left_folder) if img.find('_10.png') > 0]

        np.random.shuffle(images)
        training_images = images[:160]
        val_images = images[160:]

        left_images_train = ['data_scene_flow/training/' + left_folder + image for image in training_images]
        right_images_train = ['data_scene_flow/training/' + right_folder + image for image in training_images]
        left_disps_train = ['data_scene_flow/training/' + disp_left + image for image in training_images]
        left_images_test = ['data_scene_flow/training/' + left_folder + image for image in val_images]
        right_images_test = ['data_scene_flow/training/' + right_folder + image for image in val_images]
        left_disps_test = ['data_scene_flow/training/' + disp_left + image for image in val_images]
        
    elif root == 'driving':

        left_folder  = 'Driving_dataset/frames_cleanpass/15mm_focallength/scene_backwards/fast/left/'
        right_folder = 'Driving_dataset/frames_cleanpass/15mm_focallength/scene_backwards/fast/right/'
        disp_left = 'Driving_dataset/disparity/15mm_focallength/scene_backwards/fast/left/'

        images = [img for img in os.listdir(left_folder) if check_image(left_folder + img)]

        np.random.shuffle(images)
        training_images = images[:284]
        val_images = images[284:]

        left_images_train = [left_folder + image for image in training_images]
        right_images_train = [right_folder + image for image in training_images]
        left_disps_train = [disp_left + image.split(".")[0] + '.pfm' for image in training_images]
        left_images_test = [left_folder + image for image in val_images]
        right_images_test = [right_folder + image for image in val_images]
        left_disps_test = [disp_left + image.split(".")[0] + '.pfm' for image in val_images]
    else:
        return "WRONG ROOT FOLDER NAME"

    return left_images_train, right_images_train, left_disps_train, left_images_test, right_images_test, left_disps_test


if __name__ == '__main__':
    load_data('driving')