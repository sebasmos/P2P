import torchvision
import torchvision.transforms as T
import PIL
import skimage.exposure
import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import torchvision
import torch
import numpy as np
import torchvision.transforms as T
from skimage import io
import tifffile

class logDataset(BaseDataset):
    """A dataset class for paired image dataset.
    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """
    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.transforms_he =  get_transform(train= opt.train, size=256, HE_IF = "he")
        self.transforms_if = get_transform(train=opt.train, size=256 , HE_IF = "if")
        print(f"training? {opt.train}")
        self.imgs = list(sorted([logo_name for i, logo_name in enumerate(os.listdir(os.path.join(opt.dataroot, "img_he"))) if ".tif" in logo_name]  
))# HE
        self.targets = list(sorted([logo_name for i, logo_name in enumerate(os.listdir(os.path.join(opt.dataroot, "img_if"))) if ".tif" in logo_name]  
))# IF  
    def __getitem__(self, index):
        img_path = os.path.join(self.root, "img_he", self.imgs[index])        
        targets_path = os.path.join(self.root, "img_if", self.targets[index])
        img = tifffile.imread(img_path)
        target = tifffile.imread(targets_path)
        target = skimage.util.img_as_float32(target)#**ADDED**
        img = np.moveaxis(img, 0, 2)
        # Normalize images
        CHANNELS = range(19)
        log_img = np.log10(target, where=(target != 0))
        #print("3: (max of log_img in float32) ", log_img.max(), log_img.shape)
            
        lower_cutoff_log = np.percentile(log_img.ravel(), 0.17)
        upper_cutoff_log = np.percentile(log_img.ravel(), 99.99)
        rescaled_log_img = (
                (((1-0)*(log_img.ravel()-lower_cutoff_log)) /
                (upper_cutoff_log-lower_cutoff_log)
                ) + 0).reshape(log_img.shape)
        #print("4: (max of rescaled_log_img in float32) ", rescaled_log_img.max(), rescaled_log_img.shape)
            
        target = np.clip(a=rescaled_log_img, a_min=0, a_max=1)
        target = np.moveaxis(target, 0, 2)
        #print("5: (max of target in float32) ", target.max(), target.shape)
        
        img, target = self.transforms_he(img), self.transforms_if(target)
        #print("6: (max of target in float32) ", target.max(), target.shape)
        #print(f"TARGET SIZE IN DATALOADER: ", target.shape, img.shape)
        #return {'A': img, 'B': target, 'A_paths': img_path, 'B_paths': targets_path}
        return {'A': target, 'B': img, 'A_paths': targets_path, 'B_paths': img_path}
    
    def __len__(self):
        return len(self.imgs)
    
def get_transform(train, size=256, HE_IF = "he"):
    transforms = []
    transforms.append(T.ToTensor())

    if train==1:
        print("training data!!")
        transforms.append(T.Resize((size,size)))
        # Adding transformation to both inputs
        transforms.append(T.RandomHorizontalFlip(0.5))
    else:
        print("testing data!!")
        transforms.append(T.Resize((size,size)))
        
    return T.Compose(transforms)
