MIN_PERCENTILE = 0.17
MAX_PERCENTILE = 99.9#99.99

# CRC01
#full_img = "/n/pfister_lab2/Lab/vcg_biology/ORION/ORION-IF-1/P37_S34_A24_C59kX_E15@20220107_202112_212579.ome.tiff"
# CRC04
full_img = "/n/pfister_lab2/Lab/vcg_biology/ORION/ORION-IF-1/P37_S32_A24_C59kX_E15@20220106_014630_553652.ome.tiff"
root = "/net/coxfs01/srv/export/coxfs01/pfister_lab2/share_root/Lab/scajas/DATASETS/DATASET_pix2pix/train"

# save directory
save_dir = "/net/coxfs01/srv/export/coxfs01/pfister_lab2/share_root/Lab/scajas/pytorch-CycleGAN-and-pix2pix_master/CRC01_distributions_title_fixed_CRC04"

import os
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

os.listdir(root)
import copy
import os
import torchvision
import torchvision.transforms as T
import skimage.exposure
import torch
from PIL import Image

import torch
import torch.utils.data
import torchvision


#import transforms as T
import torchvision.transforms as visionT
import pdb
import numpy as np
import cv2

from skimage import io

import glob

import random
import tifffile
import pickle
import time
import matplotlib.pyplot as plt




def plot_imgs(imgs, titles):
    """
    Generate visualization of list of arrays
    :param imgs: list of arrays, each numpy array is an image of size (width, height)
    :param titles: list of titles [string]
    """
    # create figure
    fig = plt.figure(figsize=(50, 50))
    # loop over images
    for i in range(len(imgs)):
        fig.add_subplot(4, 4, i + 1)
        plt.imshow(imgs[i])
        plt.title(str(titles[i]))
        plt.axis("off")

import torchvision
import torchvision.transforms as T
#import transforms as T # Custom version 
import PIL
import skimage.exposure


from tifffile import imread

marker = ['Hoechst','AF1','CD31','CD45','CD68','Blank','CD4',
          'FOXP3','CD8a','CD45RO','CD20','PD-L1','CD3e','CD163',
          'E-Cadherin','PD-1','Ki-67','Pan-CK','SMA'
         ]

# format plot grid
numRows = 4
numColumns = 6
grid_dims = (numRows, numColumns)
# initialize figure canvas
fig_orig = plt.figure(figsize=(12, 8.5))
fig_log = plt.figure(figsize=(12, 8.5))
fig_clip = plt.figure(figsize=(12, 8.5))

cutoffs = {}
for e, marker_name in enumerate(marker):
    # add channel cutoffs to dict
    #cutoffs[marker_name]=(lower_cutoff_log[e], upper_cutoff_log[e])
    # scale 0.17th and 99.99th percentile between 0 and 1
    # Note: this will cause outlier pixels below the 0.17th percentile and above
    # the 99.99th to take values <0 and >1, respectively
    # Read channel
    img = imread(full_img, key=e)
    #img =img_original[:,:,e] 
    print("Mean IF img_original over all dataloader: ", img.shape, "Min value: ",  img.min(),"Max Value: ", img.max())
    # log-transform image
    log_img = np.log10(img, where=(img != 0))
    print("Mean IF log_img over all dataloader: ", log_img.shape, "Min value: ",  log_img.min(),"Max Value: ", log_img.max())
    # specify lower and upper percentile cutoffs
    lower_cutoff_log = np.percentile(log_img.ravel(), 0.17)
    upper_cutoff_log = np.percentile(log_img.ravel(), 99.99)
    
    # add channel cutoffs to dict
    cutoffs[marker_name] = (lower_cutoff_log, upper_cutoff_log)
    
    # scale 0.17th and 99.99th percentile between 0 and 1
    # Note: this will cause outlier pixels below the 0.17th percentile and above
    # the 99.99th to take values <0 and >1, respectively
    rescaled_log_img = (
        (((1-0)*(log_img.ravel()-lower_cutoff_log)) /
         (upper_cutoff_log-lower_cutoff_log)
         ) + 0).reshape(log_img.shape)

    #rescaled_log_img = (
    #    (((1-0)*(log_img.ravel()-lower_cutoff_log[e])) /
    #     (upper_cutoff_log[e]-lower_cutoff_log[e])
    #     ) + 0).reshape(log_img.shape)
    
    # clip outliers to lower and upper percentile cutoffs (i.e., 0-1)
    clip_rescaled_log_img = np.clip(a=rescaled_log_img, a_min=0, a_max=1)
    
    # add channel subplot to figures
    ax_orig = fig_orig.add_subplot(grid_dims[0], grid_dims[1], e + 1)
    ax_log = fig_log.add_subplot(grid_dims[0], grid_dims[1], e + 1)
    ax_clip = fig_clip.add_subplot(grid_dims[0], grid_dims[1], e + 1)
    fig_orig.tight_layout() #**ADDING**: vertical padding
    fig_log.tight_layout() #**ADDING**: vertical padding

    # plot original channel histogram
    vals, bins, patches = ax_orig.hist(
        img.ravel(), bins=60, color='tab:blue', alpha=0.7, rwidth=0.85
        )
    ax_orig.title.set_text(marker_name)

    # plot log-transformed channel histogram
    vals, bins, patches = ax_log.hist(
        log_img.ravel(), bins=60, color='tab:blue', alpha=0.7, rwidth=0.85
        )
    ax_log.vlines(
        x=[np.percentile(log_img.ravel(), 0.17),
           np.percentile(log_img.ravel(), 99.99)],
        ymin=0, ymax=vals.max(), color='tab:red'
           )
    ax_log.title.set_text(marker_name)

    # plot normalized channel histogram
    vals, bins, patches = ax_clip.hist(
        clip_rescaled_log_img.ravel(), bins=60,
        color='tab:blue', alpha=0.7, rwidth=0.85
        )
    ax_clip.title.set_text(marker_name)

plt.xticks(fontsize=7)
plt.yticks(fontsize=7)
plt.subplots_adjust(bottom=0.01, top=0.99, left=0.01, right=0.99, hspace=0.2)
plt.tight_layout()
fig_orig.savefig(os.path.join(save_dir, 'log_hists_orig.pdf'))
fig_log.savefig(os.path.join(save_dir, 'log_hists_log.pdf'))
fig_clip.savefig(os.path.join(save_dir, 'log_hists_clip.pdf'))
plt.close('all')

# save cutoffs to disk
with open(os.path.join(save_dir, 'cutoffs.pkl'), 'wb') as handle:
    pickle.dump(cutoffs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
