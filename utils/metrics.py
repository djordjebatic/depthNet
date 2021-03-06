import numpy as np
from utils.disp2depth import *
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from PIL import Image

K_default = np.array([
    [1050, 0, 479.5],
    [0, 1050, 269.5],
    [0,    0,   1]], dtype = float)

K_driving = np.array([[450, 0, 479.5],
                    [0, 450, 269.5],
                    [0, 0, 1]], dtype = float)

def get_list_of_3d_points(depth, K):
    K_inv = np.linalg.inv(K)
    h,w = depth.shape
    list_3d = np.zeros((h*w, 3))
    i = 0
    for y_d in range(h):
        for x_d in range(w):
            x_p_2d = np.array([[x_d, y_d, 1]]).transpose()
            x_p_3d = depth[y_d, x_d] * K_inv @ x_p_2d
            list_3d[i, :] = x_p_3d[:,0]
            i += 1
    return list_3d


def get_GT_explained(depth, depth_true, K):
    'A new, experimental metric.'

    predicted_depth_3d = get_list_of_3d_points(depth, K)
    true_depth_3d = get_list_of_3d_points(depth_true, K)

    predicted_depth_tree = KDTree(predicted_depth_3d)

    distance_list = np.zeros(len(predicted_depth_3d))
    for i, x in enumerate(true_depth_3d):
        dist, idx = predicted_depth_tree.query(x.reshape(1,-1), k=1)

        distance_list[i] = dist[0,0]

    N_all = len(distance_list)

    GT_explained = []
    D_list = []
    D = 1e-4
    explained = False
    while not explained:
        new_GT_exp = np.sum(distance_list < D) / N_all
        GT_explained.append(new_GT_exp)
        D_list.append(D)
        if new_GT_exp < (1 - 1e-3):
            D *= 1.2
        else:
            explained = True

    return GT_explained, D_list

def root_mean_square_error(output, target):
    output_sq = np.ravel(output)
    target_sq = np.ravel(target)

    mse = np.mean(np.square(output_sq - target_sq))
    return np.sqrt(mse)

def relative_absolute_error(output, target):
    output_sq = np.ravel(output)
    target_sq = np.ravel(target)

    return np.mean(np.abs(output_sq - target_sq) / np.abs(output_sq))


if __name__ == '__main__':
    path_gt = '../metrics_data/0000799_gt.pfm'
    path_our = '../metrics_data/0000799_our_8_9_57.pfm'

    disp_gt, scale = readPFM(path_gt)
    disp_our, scale = readPFM(path_our)

    depth_gt = disp_to_depth_real(disp_gt)
    depth_our = disp_to_depth_real(disp_our)

    for metric, metric_name in zip([root_mean_square_error, relative_absolute_error], ['RMSE', 'MRAE']):
        m = metric(depth_our, depth_gt)
        print(metric_name, '=', m)

    # values = [disp_our, disp_gt, depth_our, depth_gt]
    # names = ['Estimated Disparity', 'GT Disparity', 'Estimated Depth', 'GT Depth']
    plt.figure()
    values = [disp_our, disp_gt]
    names = ['Estimated Disparity', 'GT Disparity']
    for i, d, name in zip([0,2],values, names):
        plt.subplot(2,2,i+1)
        plt.imshow(d)
        plt.title(name)

        plt.subplot(2,2,i+2)
        d_flat = np.ravel(d)
        plt.hist(d_flat, bins=100)
        plt.title(name + ' histogram')

    plt.show()


    print('GT explained:')
    gt_exp, d_list = get_GT_explained(depth_our, depth_gt, K_default)
    plt.figure()
    plt.plot(d_list, gt_exp)
    plt.title('GT explained')
    plt.xscale('log')
    plt.show()
