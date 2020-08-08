import numpy as np
from python_pfm import readPFM
import matplotlib.pyplot as plt
# import PyPNG

def disp_to_depth_norm(disp):
    ''' Converts disparity to depth image.

    Normalized to [0,255].
    '''
    depth = 1 / disp
    min_depth = np.min(depth)
    max_depth = np.max(depth)
    depth_range = max_depth - min_depth
    depth_norm = (depth - min_depth)/depth_range
    depth_img = depth_norm * 256
    depth_img = depth_img.astype(dtype='uint8')
    return depth_img

def disp_to_depth_real(disp, focal_length = 35, baseline = 1):
    ''' Converts disparity to depth image.

    Default focal length and baseline are for SceneFlow dataset.
    '''
    depth = (focal_length * baseline) / np.abs(disp)
    return depth


if __name__ == '__main__':
    print('Testing: disp_to_depth() ...')
    path = 'SAMPLE_BATCH/val/disparity/left/0000000.pfm'
    disp, scale = readPFM(path)
    depth_img = disp_to_depth_norm(disp)
    plt.imshow(depth_img)
    plt.show()