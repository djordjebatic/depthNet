import numpy as np
from python_pfm import readPFM
import matplotlib.pyplot as plt
# import PyPNG

def disp_to_depth(disp):
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


if __name__ == '__main__':
    print('Testing: disp_to_depth() ...')
    path = 'SAMPLE_BATCH/val/disparity/left/0000000.pfm'
    disp, scale = readPFM(path)
    depth_img = disp_to_depth(disp)
    plt.imshow(depth_img)
    plt.show()