import numpy as np
from disp2depth import *
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree

K_default = np.array([
    [1050, 0, 479.5],
    [0, 1050, 269.5],
    [0,    0,   1]], dtype = float)

K_driving = np.array([[450, 0, 479.5],
                    [0, 450, 269.5],
                    [0, 0, 1]], dtype = float)

def get_list_of_3d_points(depth, K):
    h,w = depth.shape
    list_3d = np.zeros((h*w, 3))
    i = 0
    for y_d in range(h):
        for x_d in range(w):
            x_p_2d = np.array([[x_d, y_d, 1]]).transpose()
            x_p_3d = depth[y_d, x_d] * K @ x_p_2d
            list_3d[i, :] = x_p_3d[:,0]
            i += 1
    return list_3d


def get_GT_explained(depth, depth_true, K):
    'A new, experimental metric.'

    predicted_depth_3d = get_list_of_3d_points(depth, K)
    true_depth_3d = get_list_of_3d_points(depth_true, K)

    predicted_depth_tree = KDTree(predicted_depth_3d)

    distance_list = [None] * len(predicted_depth_3d)
    for i, x in enumerate(true_depth_3d):
        dist, idx = predicted_depth_tree.query(x.reshape(1,-1), k=1)

        distance_list[i] = dist[0,0]

    N_all = len(distance_list)

    n = 100
    D_list = np.logspace(0.025, 10, n)
    GT_explained = [None] * n
    for i, D in enumerate(D_list):
        GT_explained[i] = np.sum(distance_list < D) / N_all

    return GT_explained, D_list

def root_mean_square_error(output, target):
    output_sq = np.squeeze(output)
    target_sq = np.squeeze(target)

    mse = np.mean(np.square(output_sq - target_sq))
    return np.sqrt(mse)

def relative_absolute_error(output, target):
    output_sq = np.squeeze(output)
    target_sq = np.squeeze(target)

    return np.mean(np.abs(output_sq - target_sq) / output_sq)

if __name__ == '__main__':
    path = 'SAMPLE_BATCH/val/disparity/left/0000000.pfm'
    disp, scale = readPFM(path)
    depth = disp_to_depth_real(disp)

    for metric in [root_mean_square_error, relative_absolute_error]:
        m = metric(depth, depth)
        assert abs(m) <= 1e-4

    gt_exp, d_list = get_GT_explained(depth, depth, K_default)
    plt.plot(d_list, gt_exp)
    plt.show()
