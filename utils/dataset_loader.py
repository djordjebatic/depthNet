import os
import os.path
import filetype


def check_image(filename):
    return filetype.is_image(filename)

def load_data(root = 'FlyingThings3D_subset' ):


    left_images_train = []
    right_images_train = []
    left_disps_train = []
    left_images_test = []
    right_images_test = []
    left_disps_test = []

    for count, i in enumerate(['train', 'val']):
        if count > 150:
            break

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


    return left_images_train, right_images_train, left_disps_train, left_images_test, right_images_test, left_disps_test


if __name__ == '__main__':
    root = 'sample_dataset'
    lt, rt, ld, lte, rte, ldte = load_data(root)
    print(len(lt), len(rt), len(ld), len(lte), len(rte), len(ldte))
    for img_set in (lt, rt, ld, lte, rte, ldte):
        print(img_set)