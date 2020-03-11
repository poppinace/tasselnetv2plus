"""
@author: hao
"""

import scipy
import numpy as np
import math
import cv2 as cv
from scipy.ndimage import gaussian_filter, morphology
from skimage.measure import label, regionprops
from sklearn import linear_model
import matplotlib.pyplot as plt


def compute_mae(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    mae = np.mean(np.abs(diff))
    return mae


def compute_mse(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    mse = np.sqrt(np.mean((diff ** 2)))
    return mse

def compute_relerr(pd, gt):
    pd, gt = np.array(pd), np.array(gt)
    diff = pd - gt
    diff = diff[gt > 0]
    gt = gt[gt > 0]
    if (diff is not None) and (gt is not None):
        rmae = np.mean(np.abs(diff) / gt) * 100
        rmse = np.sqrt(np.mean(diff**2 / gt**2)) * 100
    else:
        rmae = 0
        rmse = 0
    return rmae, rmse


def rsquared(pd, gt):
    """ Return R^2 where x and y are array-like."""
    pd, gt = np.array(pd), np.array(gt)
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(pd, gt)
    return r_value**2


def dense_sample2d(x, sx, stride):
    (h,w) = x.shape[:2]
    #idx_img = np.array([i for i in range(h*w)]).reshape(h,w)
    idx_img = np.zeros((h,w),dtype=float)
    
    th = [i for i in range(0, h-sx+1, stride)]
    tw = [j for j in range(0, w-sx+1, stride)]
    norm_vec = np.zeros(len(th)*len(tw))

    for i in th:
        for j in tw:
            idx_img[i:i+sx,j:j+sx] = idx_img[i:i+sx,j:j+sx]+1

    # # plot redundancy map
    # import os
    # import matplotlib.pyplot as plt
    # cmap = plt.cm.get_cmap('hot')
    # idx_img = idx_img / (idx_img.max())
    # idx_img = cmap(idx_img) * 255.
    # plt.figure()
    # plt.imshow(idx_img.astype(np.uint8))
    # plt.axis('off')
    # plt.savefig(os.path.join('redundancy_map.pdf'), bbox_inches='tight', dpi = 300)
    # plt.close()
   
    idx_img = 1/idx_img
    idx_img = idx_img/sx/sx
    #line order
    idx = 0
    for i in th:
        for j in tw:
            norm_vec[idx] =idx_img[i:i+sx,j:j+sx].sum()
            idx+=1
    
    return norm_vec


def recover_countmap(pred, image, patch_sz, stride):
    pred = pred.reshape(-1)
    imH, imW = image.shape[2:4]
    cntMap = np.zeros((imH, imW), dtype=float)
    norMap = np.zeros((imH, imW), dtype=float)
    
    H = np.arange(0, imH - patch_sz + 1, stride)
    W = np.arange(0, imW - patch_sz + 1, stride)
    cnt = 0
    for h in H:
        for w in W:
            pixel_cnt = pred[cnt] / patch_sz / patch_sz
            cntMap[h:h+patch_sz, w:w+patch_sz] += pixel_cnt
            norMap[h:h+patch_sz, w:w+patch_sz] += np.ones((patch_sz,patch_sz))
            cnt += 1
    return cntMap / (norMap + 1e-12)
