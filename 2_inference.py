#!/usr/bin/env python
# coding: utf-8

# ### 2 Main Pipeline for Inference
# 
import numpy as np
import cv2
from skimage import measure
from tqdm import tqdm
import os
import pandas as pd
import argparse
import glob, os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import morphology, measure
from skimage.color import hsv2rgb
from scipy import stats, ndimage
from tqdm import tqdm
from PIL import Image
import argparse

import os
import glob
import sys
sys.path.append("/cache/plaquebox-paper/")
# sys.path.append("/cache/plaquebox-paper/utils")
# sys.path.append("/cache/plaquebox-paper/")

import torch
torch.manual_seed(123456789)
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from tqdm import tqdm
import argparse
import time
import gc

class HeatmapDataset(Dataset):
    def __init__(self, tile_dir, row, col, normalize, stride=1):
        """
        Args:
            tile_dir (string): path to the folder where tiles are
            row (int): row index of the tile being operated
            col (int): column index of the tile being operated
            stride: stride of sliding 
        """
        self.tile_size = 256
        self.img_size = 1536
        self.stride = stride
        padding = 128
        large_img = torch.ones(3, 3*self.img_size, 3*self.img_size)
        
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                img_path = tile_dir+'/'+str(row+i)+'/'+str(col+j)+'.jpg'
                try:
                    img = Image.open(img_path)
                    img = transforms.ToTensor()(img) 
                except:
                    img = torch.ones(3,self.img_size, self.img_size)
                
                large_img[:, (i+1)*self.img_size:(i+2)*self.img_size,(j+1)*self.img_size:(j+2)*self.img_size] = img
        
        large_img = normalize(large_img)
        
        self.padding_img = large_img[:,self.img_size-padding:2*self.img_size+padding, self.img_size-padding:2*self.img_size+padding]
        self.len = (self.img_size//self.stride)**2
        
    def __getitem__(self, index):

        row = (index*self.stride // self.img_size)*self.stride
        col = (index*self.stride % self.img_size)

        img = self.padding_img[:, row:row+self.tile_size, col:col+self.tile_size]        
    
        return img

    def __len__(self):
        return self.len

class Net(nn.Module):

    def __init__(self, fc_nodes=512, num_classes=3, dropout=0.5):
        super(Net, self).__init__()
        
    def forward(self, x):
 
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

def saveBrainSegImage(nums, save_dir) :
    """
    Converts 2D array with {0,1,2} into RGB
     to determine different segmentation areas
     and saves image at given directory
    
    Input:
       nums: 2D-NumPy Array containing classification
       save_dir: string indicating save location
    """ 
    
    nums = np.repeat(nums[:,:, np.newaxis], 3, axis=2)
    
    # nums[:,:,0] = RED, nums[:,:,1] = Green, nums[:,:,2] = Blue
    idx_1 = np.where(nums[:,:,0] == 1)  # Index of label 1 (WM)
    idx_2 = np.where(nums[:,:,0] == 2)  # Index of label 2 (GM)

    # For label 0, leave as black color
    # For label 1, set to yellow color: R255G255B0 (WM)
    nums[:,:,0].flat[np.ravel_multi_index(idx_1, nums[:,:,0].shape)] = 255
    nums[:,:,1].flat[np.ravel_multi_index(idx_1, nums[:,:,1].shape)] = 255
    nums[:,:,2].flat[np.ravel_multi_index(idx_1, nums[:,:,2].shape)] = 0
    # For label 2, set to cyan color: R0G255B255 (GM)
    nums[:,:,0].flat[np.ravel_multi_index(idx_2, nums[:,:,0].shape)] = 0
    nums[:,:,1].flat[np.ravel_multi_index(idx_2, nums[:,:,1].shape)] = 255
    nums[:,:,2].flat[np.ravel_multi_index(idx_2, nums[:,:,2].shape)] = 255

    nums = nums.astype(np.uint8) # PIL save only accepts uint8 {0,..,255}
    save_img = Image.fromarray(nums, 'RGB')
    save_img.save(save_dir)
    print("Saved at: " + save_dir)

def count_blobs_vizz(mask, size_threshold=1,which_label=0):
    #are there too many labels? check how many to see if this can be optimized
    mask = np.where((mask == 0) | (mask == which_label), 0, mask)
    labels = measure.label(mask, connectivity=2, background=0)
    
    new_mask = np.zeros(mask.shape, dtype='uint8')
    labeled_mask = np.zeros(mask.shape, dtype='uint16')
    sizes = []
    print(np.unique(mask,return_counts=True))
    
    
    for label in tqdm(np.unique(labels)):
        # print(label,len(np.unique(labels)))
        
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels 
        labelMask = np.zeros(mask.shape, dtype="uint8")
        if which_label==1:#paint label 2
            labelMask[labels == label] = 255
        elif which_label==2:#paintr label 1
            labelMask[labels == label] = 200
        numPixels = cv2.countNonZero(labelMask)
        
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "large blobs"

        if numPixels > size_threshold:
            sizes.append(numPixels)
            new_mask = cv2.add(new_mask, labelMask)

            # Save confirmed unique location of plaque
            
            
            labeled_mask[labels==label] = label
    # exit()
    num_labels = len(np.unique(labels))
    
    return sizes, new_mask, num_labels,labeled_mask


def classify_blobs(labeled_mask, seg_area) :
    """
    Classifies each certain plaques according to each
    Segmentation Area and gives each count
    
    Input:
        labeled_mask (NumPy Array): 
            contains plaque information 
            Note: See count_blobs()'s 
            labeled_mask output for more info
        
        seg_area (NumPy Array):
            contains segmentation information
            based on BrainSeg's classification
            
    Output:
        count_dict (Dictionary):
            contains number of plaques at each
            segmentaion area
            
        Other Variables:
            - Background Count
            - WM Count
            - GM Count
            - Unclassified Count
    """
    
    # 0: Background, 1: WM, 2: GM
    count_dict = {0: 0, 1: 0, 2: 0, "uncounted": 0}
    print("oinside classify nlons",np.unique(labeled_mask),len(np.unique(labeled_mask)))
    # Loop over unique components
    for label in np.unique(labeled_mask) :
        
        if label == 0:
            continue
            
        plaque_loc = np.where(labeled_mask == label)
        
        plaque_area = seg_area[plaque_loc]
        
        indexes, counts = np.unique(plaque_area, return_counts=True)
        # print(label,counts)
        class_idx = indexes[np.where(counts == np.amax(counts))]
        
        try:
            class_idx = class_idx.item()
            count_dict[class_idx] += 1
                
        except:
            count_dict["uncounted"] += 1
    # exit()
            
    return count_dict, count_dict[0], count_dict[1], count_dict[2], count_dict["uncounted"]




def inference(IMG_DIR, MODEL_PLAQ, SAVE_PLAQ_DIR, MODEL_SEG, SAVE_IMG_DIR, SAVE_NP_DIR,SAVE_NFT_IMG_DIR,SAVE_NFT_NP_DIR):
    img_size = 1536
    stride = 32
    batch_size = 48 
    num_workers = 16

    # norm = np.load('utils/normalization.npy', allow_pickle=True).item() # brainseg
    norm = np.load('/cache/plaquebox-paper/utils/normalization.npy', allow_pickle=True).item()
    normalize = transforms.Normalize(norm['mean'], norm['std'])
    
    to_tensor = transforms.ToTensor()
    
    # Retrieve Files
    filenames = glob.glob(IMG_DIR + '*')
    filenames = [filename.split('/')[-1] for filename in filenames]
    filenames = sorted(filenames)
    print(filenames)

    # Check GPU:
    use_gpu = torch.cuda.is_available()
    
    # instatiate the model
    # plaq_model = torch.load(MODEL_PLAQ, map_location=lambda storage, loc: storage)
    seg_model = torch.load(MODEL_SEG, map_location=lambda storage, loc: storage)
    
    if use_gpu:
        seg_model = seg_model.cuda() # Segmentation
        # plaq_model = plaq_model.module.cuda() # Plaquebox-paper
    else:
        seg_model = seg_model
        # plaq_model = plaq_model.module

    # Inference Loop:
    import pandas as pd

    # Sample data: Replace this with your actual DataFrame
    data_nftcounts_dict = {
        'WSI_NAME':[],
        'prenft_count_wm': [],
        'prenft_count_gm': [],
        'prenft_count_bg': [],
        'prenft_count_total': [],
        'inft_count_wm': [],
        'inft_count_gm': [],
        'inft_count_bg': [],
        'inft_count_total': [],
    }


    for filename in filenames[:]:
        print("Now processing: ", filename)
    

        
        # Retrieve Files
        TILE_DIR = IMG_DIR+'{}/0/'.format(filename)
        if os.path.isdir(TILE_DIR) is False:
            print(TILE_DIR, " not a directory")
            continue
        imgs = []
        for target in sorted(os.listdir(TILE_DIR)):
            d = os.path.join(TILE_DIR, target)
            if not os.path.isdir(d):
                continue

            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    if fname.endswith('.jpg'):
                        path = os.path.join(root, fname)
                        imgs.append(path)
        
        # Imports
        import sys
        sys.path.append('/cache/Shivam/yolo-braak-stage/')  # mamke nft_helpers module available
        sys.path.append('/cache/Shivam/yolo-braak-stage/nft_helpers/')
        import matplotlib.pyplot as plt
        import cv2 as cv
        import large_image

        # Parameters
        # ----------
        # Choose weights to load on YOLO model to predict. The best.pt file in the repo is the best MAL weights from
        # project.
        weights = '/cache/Shivam/yolo-braak-stage/best.pt'

        # Select an ROI image to test. Note that some ROIs have no NFTs to predict so might return no predictions.
        # roi_fp = '/cache/Shivam/BrainSec-py/data_ONCE/norm_tiles/1-271-Temporal_AT8.czi/0/10/13.jpg'


        # For WSI we speed tiling by using parallel processing in Python 
        # This parameter sets the number of processes to use, don't set this higher than the number
        # cores your machine otherwise it will be detrimental.
        nproc = 10

        device = None  # options, "cpu", None (all GPUs available), or specific ids (e.g. "0,1")

        # Thresholds:
        iou_thr = 0.4  # non-max-suppresion IoU
        contained_thr = 0.7  # removing small boxes mostly in larger ones
        conf_thr = 0.5  # remove predictions of low confidence
        mask_thr = 0.25  # ignore tiles that are mostly not in ROI or tissue mask

        # from nft_helpers import extract_tissue_mask
        from nft_helpers.utils import imread
        from nft_helpers.yolov5 import roi_inference
        from nft_helpers import extract_tissue_mask

        save_fp = 'results_jc/sample-roi-label.txt'


        rows = [int(image.split('/')[-2]) for image in imgs]
        row_nums = max(rows) + 1
        cols = [int(image.split('/')[-1].split('.')[0]) for image in imgs]
        col_nums = max(cols) +1    
        
        # Initialize outputs accordingly:
        heatmap_res = img_size // stride
        # print(heatmap_res,"heatmao size",row_nums,col_nums)

        def parse_annotation(annotation, stride_large=1536):
            """
            Parse the annotation string and convert normalized coordinates to pixel values.
            
            Args:
                annotation (str): Annotation string (class x_center y_center width height confidence).
                stride_large (int): The size of the sub-image that the bounding box is relative to (960 in this case).

            Returns:
                (int, int, int, int): The coordinates of the bounding box in pixel values (x1, y1, x2, y2).
            """
            class_, x_center, y_center, width, height, conf_ = map(float, annotation.split())
            class_=int(class_)
            
            # Convert normalized coordinates to pixel values based on stride_large (960x960)
            x_center *= stride_large
            y_center *= stride_large
            width *= stride_large
            height *= stride_large
            
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)
            
            return class_,x1, y1, x2, y2,conf_

        def update_heatmap(heatmap, x1, y1, x2, y2, row, col, offset_x, offset_y, class_,stride=1):
            """
            Update the heatmap based on bounding box coordinates.

            Args:
                heatmap (np.ndarray): The heatmap array to update.
                x1, y1, x2, y2 (int): Bounding box coordinates.
                row, col (int): Row and column indices of the current sub-image in the grid.
                offset_x, offset_y (int): Offsets to apply based on the position of the sub-image within the full image.
                stride (int): The stride of the first pipeline's heatmap (default is 32).
            """
            # Adjust coordinates to the heatmap's resolution
            x1 = (x1 + offset_x) // stride
            y1 = (y1 + offset_y) // stride
            x2 = (x2 + offset_x) // stride
            y2 = (y2 + offset_y) // stride
            
            # Calculate the offset in the heatmap for this particular image grid
            row_offset = row * (img_size // stride)
            col_offset = col * (img_size // stride)
            
            # Update the global heatmap
            # print(class_,row_offset + y1, row_offset + y2,col_offset + x1, col_offset + x2)
            small_nft_output[class_,  y1: y2,x1:   x2] = 1
            nft_output[class_,row_offset + y1: row_offset + y2,
                    col_offset + x1: col_offset + x2] = 1
                
        nft_output = np.zeros((2, img_size*row_nums, img_size*col_nums))
        print(nft_output.shape)
        # exit()
        # plaque_output = np.zeros((3, heatmap_res*row_nums, heatmap_res*col_nums))
        seg_output = np.zeros((heatmap_res*row_nums, heatmap_res*col_nums), dtype=np.uint8)
        # print(seg_output.shape)
        # exit()
        seg_model.train(False)  # Set model to evaluate mode
        # plaq_model.train(False)
        
        start_time = time.perf_counter() # To evaluate Time taken per inference

        final_prenft_count_wm = 0
        final_prenft_count_gm = 0
        final_prenft_count_bg = 0
        final_inft_count_wm = 0
        final_inft_count_gm = 0
        final_inft_count_bg = 0
        for row in tqdm(range(row_nums)):
            for col in range(col_nums):
                
                # row=0
                # col=12
                final_prenft_count=0
                final_inft_count=0
                image_datasets = HeatmapDataset(TILE_DIR, row, col, normalize, stride=stride)
                dataloader = torch.utils.data.DataLoader(image_datasets, batch_size=batch_size,
                                                    shuffle=False, num_workers=num_workers)
                
                # # From Plaque-Detection:
                running_plaques = torch.Tensor(0)
                small_nft_output = np.zeros((2, img_size, img_size), dtype=np.uint8)
                # For Stride 32 (BrainSeg):
                running_seg = torch.zeros((32), dtype=torch.uint8)
                output_class = np.zeros((heatmap_res, heatmap_res), dtype=np.uint8)

                roi_fp = TILE_DIR+'/'+str(row)+'/'+str(col)+'.jpg'

                
                
                save_fp = 'results_jc/sample-roi-label.txt'
                temp_dir="/cache/Shivam/BrainSec-py/"+str(IMG_DIR.split("/")[0])+'/vizz_nft_outputs_stride1536/{}/{}/{}'.format(filename,row,col)
                print(TILE_DIR,IMG_DIR,temp_dir)

                # exit()
                print("ROW, COLUMNS",row,col)
                print("STARTINF NFT PREDICTION","#########",sep="\n")
                pred_df = roi_inference(
                    roi_fp, weights, tile_size = 1536, temp_dir=temp_dir,save_fp=save_fp,yolodir='/cache/Shivam/yolo-braak-stage/nft-detection-yolov5/', device=device, 
                    iou_thr=iou_thr, contained_thr=contained_thr, conf_thr=conf_thr, 
                    boundary_thr=mask_thr
                )
                # for txtfile in os.listdir(os.path.join(temp_dir,"predictions","labels"))
                annotation_files = [
                    f'{col}-x0y0.txt', 
                    f'{col}-x0y960.txt', 
                    f'{col}-x960y0.txt'
                ]
                offsets = [(0, 0), (0, 960), (960, 0)]
                
                
                for annotation_file, (offset_x, offset_y) in zip(annotation_files, offsets):
                    if os.path.exists(os.path.join(temp_dir,"predictions","labels",annotation_file)):
                        annotation_file = os.path.join(temp_dir,"predictions","labels",annotation_file)
                        with open(annotation_file, 'r') as file:
                            annotations = file.readlines()  # Read all annotations from the text file
                        
                        if annotations:  # Check if there are any annotations
                            # Update the heatmap based on each annotation in the current file
                            print("NFT row col",row,col)
                            for annotation in annotations:
                                print("number of 1s before",np.count_nonzero(nft_output==1))
                                class_,x1, y1, x2, y2,conf_ = parse_annotation(annotation.strip())
                                update_heatmap(nft_output, x1, y1, x2, y2, row, col, offset_x, offset_y,class_)
                                print("number of 1s after",np.count_nonzero(nft_output==1))
                    else:
                        print(f"Annotation file {annotation_file} does not exist, skipping.")
                
                
                
                
                print("finishing NFT PREDICTION","#########",sep="\n")
                print("MFT shape is",nft_output.shape,small_nft_output.shape,np.count_nonzero(small_nft_output==1))
                # exit()

                print("STARTING WMGM PREDICTION","#########",sep="\n")

                for idx, data in enumerate(dataloader):
                    print("lenght of data for this WMGM batch",len(data),image_datasets.len,len(dataloader))
                    # get the inputs
                    inputs = data
                    # wrap them in Variable
                    if use_gpu:
                        inputs = Variable(inputs.cuda(), volatile=True)
                        
                        # # forward (Plaque Detection) :
                        # outputs = plaq_model(inputs)
                        # preds = F.sigmoid(outputs) # Posibility for each class = [0,1]
                        # preds = preds.data.cpu()
                        # running_plaques = torch.cat([running_plaques, preds])
                        
                        # exit()
                        
                        
                        # forward (BrainSeg) :
                        predict = seg_model(inputs)
                        _, indices = torch.max(predict.data, 1) # indices = 0:Background, 1:WM, 2:GM
                        indices = indices.type(torch.uint8)
                        
                        running_seg =  indices.data.cpu()
                        # print(heatmap_res,batch_size,img_size,stride,"error")
                        print("tile classifn",row,col,running_seg,running_seg.shape)
                        # exit()
                        # For Stride 32 (BrainSeg) :
                        i = (idx // (heatmap_res//batch_size))
                        j = (idx % (heatmap_res//batch_size))
                        
                        output_class[i,j*batch_size:(j+1)*batch_size] = running_seg
                saveBrainSegImage(output_class, \
                        SAVE_IMG_DIR + filename+'_'+str(row)+'_'+str(col) + '_tempROI2.png')
                segRA = np.repeat(np.repeat(output_class, 32, axis=0), stride, axis=1)
                saveBrainSegImage(segRA, \
                        SAVE_IMG_DIR + filename+'_'+str(row)+'_'+str(col) + '_tempROIExtended2.png')
                # exit()
                print("for this nft output is",row,col,nft_output.shape,small_nft_output.shape,torch.max(running_seg))
                # print("large roi classifn",row,col,output_class.shape,running_seg.shape)
                # print(np.unique(output_class,return_counts=True))

                # larger_roi = np.repeat(np.repeat(output_class,stride,axis=0),stride,axis=1)
                # print("large roi shape",larger_roi.shape,np.unique(larger_roi,return_counts=True))

                
                # calculating the nft for the small ROI
                # new_nft_output = np.zeros((small_nft_output.shape[1], small_nft_output.shape[2]), dtype=int)
                # print("zero arr created")
                # # Set locations for preNFT (class 0) to 1
                # new_nft_output[small_nft_output[0, :, :] == 1] = 1

                # # Set locations for iNFT (class 1) to 2
                # new_nft_output[small_nft_output[1, :, :] == 1] = 2

                # Now result_array contains 1 for preNFT locations and 2 for iNFT locations
                # print(new_nft_output)
                # unique_elements, counts = np.unique(new_nft_output, return_counts=True)
                # print("locations added",small_nft_output.shape)
                # PreNFTlabels, PreNFTnew_mask, PreNFTnum_labels,PreNFTlabeled_mask = count_blobs_vizz(small_nft_output,size_threshold=1,which_label=0)
                # # counts, bg, wm, gm, unknowns = classify_blobs(PreNFTlabeled_mask, larger_roi)
                # print("PreNFT ",PreNFTlabels,PreNFTnum_labels)                
                # if len(PreNFTlabels)>0:
                #     print(print("PreNFT found",len(PreNFTlabels)))
                #     final_prenft_count +=len(PreNFTlabels)


                # iNFTlabels, iNFTnew_mask, iNFTnum_labels,iNFTlabeled_mask = count_blobs_vizz(small_nft_output,size_threshold=1,which_label=1)
                # # counts, bg, wm, gm, unknowns = classify_blobs(iNFTlabeled_mask, larger_roi)
                # print("iNFT",iNFTlabels,iNFTnum_labels )
                # if len(iNFTlabels)>1:
                #     print(print("iNFT found",len(iNFTlabels)))
                #     final_inft_count+=len(iNFTlabels)
                # # exit()

                # if np.max(output_class)==0:
                #     final_prenft_count_bg += final_prenft_count
                #     final_inft_count_bg += final_inft_count
                    
                # elif np.max(output_class)==1:
                #     final_prenft_count_wm += final_prenft_count
                #     final_inft_count_wm += final_inft_count

                # elif np.max(output_class)==2:
                #     final_prenft_count_gm +=final_prenft_count
                #     final_inft_count_gm += final_inft_count
                # else:
                #     print("unknown class")
                # print(final_prenft_count,final_inft_count,type(final_inft_count),data_nftcounts_dict)
                # exit()
                                
                print(output_class,"row",row,col,output_class.shape)
                seg_output[row*heatmap_res:(row+1)*heatmap_res, col*heatmap_res:(col+1)*heatmap_res] = output_class
                # exit()
                # Final Outputs of Plaque Detection:
                # cored = np.asarray(running_plaques[:,0]).reshape(img_size//stride,img_size//stride)
                # diffuse = np.asarray(running_plaques[:,1]).reshape(img_size//stride,img_size//stride)
                # caa = np.asarray(running_plaques[:,2]).reshape(img_size//stride,img_size//stride)
                
                # plaque_output[0, row*heatmap_res:(row+1)*heatmap_res, col*heatmap_res:(col+1)*heatmap_res] = cored
                # plaque_output[1, row*heatmap_res:(row+1)*heatmap_res, col*heatmap_res:(col+1)*heatmap_res] = diffuse
                # plaque_output[2, row*heatmap_res:(row+1)*heatmap_res, col*heatmap_res:(col+1)*heatmap_res] = caa

                seg_output[row*heatmap_res:(row+1)*heatmap_res, col*heatmap_res:(col+1)*heatmap_res] = output_class
                # break

        # # Saving Confidence=[0,1] for Plaque Detection
        # np.save(SAVE_PLAQ_DIR+filename, plaque_output)
        # data_nftcounts_dict['WSI_NAME'].append(filename)
        # data_nftcounts_dict['prenft_count_wm'].append(final_prenft_count_wm)
        # data_nftcounts_dict['prenft_count_gm'].append(final_prenft_count_gm)
        # data_nftcounts_dict['prenft_count_bg'].append(final_prenft_count_bg)
        # data_nftcounts_dict['prenft_count_total'].append(final_prenft_count_wm+final_prenft_count_gm)
        # data_nftcounts_dict['inft_count_wm'].append(final_inft_count_wm)
        # data_nftcounts_dict['inft_count_gm'].append(final_inft_count_gm)
        # data_nftcounts_dict['inft_count_bg'].append(final_inft_count_bg)
        # data_nftcounts_dict['inft_count_total'].append(final_inft_count_wm+final_inft_count_gm)

        # df =pd.DataFrame(data_nftcounts_dict)
        # df.to_csv("/cache/Shivam/BrainSec-py/"+str(IMG_DIR.split("/")[0])+'/vizz_nft_outputs_stride1536_2/NFT_scores_ROILevelMaxRuleCorrespondence2.csv',index=False) 

        # # Saving Confidence=[0,1] for NFT Detection
        np.save("/cache/Shivam/BrainSec-py/"+str(IMG_DIR.split("/")[0])+'/vizz_nft_outputs_stride1536_2/'+filename, nft_output)
        
        # Saving BrainSeg Classification={0,1,2}
        # exit()
        np.save(SAVE_NP_DIR+filename, seg_output)
        saveBrainSegImage(seg_output, \
                        SAVE_IMG_DIR + filename + '.png')
        
        # Time Statistics for Inference
        end_time = time.perf_counter()
        print("Time to process " \
            + filename \
            + ": ", end_time-start_time, "sec")
        
        
def plot_heatmap(final_output) :
    """
    Plots Confidence Heatmap of Plaques = [0,1]
    
    Inputs:
        final_output (NumPy array of 
        3*img_height*height_width) :
            Contains Plaque Confidence with each axis
            representing different types of plaque
            
    Outputs:
        Subplots containing Plaque Confidences
    """
    fig = plt.figure(figsize=(45,15))

    ax = fig.add_subplot(311)

    im = ax.imshow(final_output[0], cmap=plt.cm.get_cmap('viridis', 20), vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])

    ax = fig.add_subplot(312)

    im = ax.imshow(final_output[1], cmap=plt.cm.get_cmap('viridis', 20), vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])

    ax = fig.add_subplot(313)

    im = ax.imshow(final_output[2], cmap=plt.cm.get_cmap('viridis', 20), vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax, ticks=[0.0, 0.25, 0.5, 0.75, 1.0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, default='data/norm_tiles/', help="Directory to retrieve the patches for inference")
    # Plaque Detection
    parser.add_argument("--model_plaq", type=str, default='models/CNN_model_parameters.pkl', help="Saved model for plaque detection")
    parser.add_argument("--save_plaq_dir", type=str, default='data/outputs/heatmaps/', help="Directory to save heatmaps")
    # Brainseg
    parser.add_argument("--model_seg", type=str, default='models/ResNet18_19.pkl', help="Saved model for segmentation")
    parser.add_argument("--save_img_dir", type=str, default='data/brainseg/images/', help="Directory to save image masks")
    parser.add_argument("--save_np_dir", type=str, default='data/brainseg/numpy/', help="Directory to save numpy masks")

    args = parser.parse_args()

    print(f"Tiled Image Directory: {args.img_dir}")
    print(f"Plaque detection model path: {args.model_plaq}")
    print(f"Heatmap save path: {args.save_plaq_dir}")
    print(f"Segmentation model path: {args.model_seg}")
    print(f"Image mask save path: {args.save_img_dir}")
    print(f"Numpy mask save path: {args.save_np_dir}")


    IMG_DIR = args.img_dir              #'data_1_40/norm_tiles/'
    MODEL_PLAQ = args.model_plaq        #'models/CNN_model_parameters.pkl'
    SAVE_PLAQ_DIR = args.save_plaq_dir  #'data_1_40/outputs/heatmaps/'
    MODEL_SEG = args.model_seg          #'models/ResNet18_19.pkl'
    SAVE_IMG_DIR = str(IMG_DIR.split("/")[0])+"/brainseg/images/" #args.save_img_dir    #'data_1_40/brainseg/images/'
    SAVE_NP_DIR = str(IMG_DIR.split("/")[0])+"/brainseg/numpy/"  #args.save_np_dir      #'data_1_40/brainseg/numpy/'

    SAVE_NFT_IMG_DIR = str(IMG_DIR.split("/")[0])+"/outputs/masked_nft/images/" #args.save_img_dir    #'data_1_40/brainseg/images/'
    SAVE_NFT_NP_DIR = str(IMG_DIR.split("/")[0])+"/outputs/masked_nft/numpy/" 

    
    if not os.path.exists(IMG_DIR):
        print("Tiled image folder does not exist, script should stop now")
    elif not os.path.exists(MODEL_PLAQ):
        print("Plaque detection model does not exist, script should stop now")
    elif not os.path.exists(MODEL_SEG):
        print("Segmentation model does not exist, script should stop now")
    else:
        if not os.path.exists(SAVE_PLAQ_DIR):
            os.makedirs(SAVE_PLAQ_DIR)
        if not os.path.exists(SAVE_IMG_DIR):
            os.makedirs(SAVE_IMG_DIR)
        if not os.path.exists(SAVE_NP_DIR):
            os.makedirs(SAVE_NP_DIR)
        if not os.path.exists(SAVE_NFT_IMG_DIR):
            os.makedirs(SAVE_NFT_IMG_DIR)
        if not os.path.exists(SAVE_NFT_NP_DIR):
            os.makedirs(SAVE_NFT_NP_DIR)

        print("Found tiled image folder... ")
        img_files = os.listdir(IMG_DIR)
        filenames = sorted(img_files)
        print("All files in img_dir: ")
        print(filenames)

    #----------------------------------------------------------

    # nft_output = np.load("/cache/Shivam/BrainSec-py/data_ONCE_uncorrupted/vizz_nft_outputs/1-102-Temporal_AT8.czi.npy")
    # new_nft_output = np.zeros((nft_output.shape[1], nft_output.shape[2]), dtype=int)
    # print("zero arr created")
    # # Set locations for preNFT (class 0) to 1
    # new_nft_output[nft_output[0, :, :] == 1] = 1

    # # Set locations for iNFT (class 1) to 2
    # new_nft_output[nft_output[1, :, :] == 1] = 2
    
    # print( count_blobs_vizz(new_nft_output,size_threshold=1,which_label=1))
    # print( count_blobs_vizz(new_nft_output,size_threshold=1,which_label=2))
    # exit()
    inference(IMG_DIR, MODEL_PLAQ, SAVE_PLAQ_DIR, MODEL_SEG, SAVE_IMG_DIR, SAVE_NP_DIR,SAVE_NFT_IMG_DIR,SAVE_NFT_NP_DIR)
    print("____________________________________________")
    print("Segmentation masks and heatmaps generated")

if __name__ == "__main__":
    main()