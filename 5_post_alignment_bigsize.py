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

def saveBrainSegImage(nums, save_dir) :
    """
    Converts 2D array with {0,1,2} into RGB
     to determine different segmentation areas
     and saves image at given directory
    
    Input:
       nums: 2D-NumPy Array containing classification
       save_dir: string indicating save location
    """ 
    # print(nums,nums.shape)
    # unique_values, counts = np.unique(nums, return_counts=True)

    # Display the results
    # for value, count in zip(unique_values, counts):
    #     print(f"Value: {value}, Count: {count}")
    # exit()
    nums = np.repeat(nums[:,:, np.newaxis], 3, axis=2)
    
    # nums[:,:,0] = RED, nums[:,:,1] = Green, nums[:,:,2] = Blue
    idx_0 = np.where(nums[:,:,0] == 0)  # Index of BG
    
    idx_1 = np.where(nums[:,:,0] == 1)  # Index of label 1 (WM)
    idx_2 = np.where(nums[:,:,0] == 2)  # Index of label 2 (GM)
    print("idx1",idx_1,"idx2",idx_2,"lengths",len(idx_1[0]),len(idx_1[1]),len(idx_2[0]),len(idx_2[1]),len(idx_0[0]),len(idx_0[1]),sep="\n\n\n\n")
    unique_values, counts = np.unique(nums, return_counts=True)
    WM_cnt = len(idx_1[0])
    GM_cnt = len(idx_2[0])
    # Display the results
    # for value, count in zip(unique_values, counts):
    #     print(f"Value: {value}, Count: {count}")
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
    return nums,WM_cnt,GM_cnt


# Example shapes (adjust based on actual data)
# Assuming brainseg is already defined with its shape

import numpy as np


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
    # print("oinside classify nlons",np.unique(labeled_mask),len(np.unique(labeled_mask)))
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

def get_filenames(BRAINSEG_NP_PRE_DIR):
    filenames = sorted(os.listdir(BRAINSEG_NP_PRE_DIR))
    filenames = [os.path.splitext(file)[0] for file in filenames]
    return filenames
#for sina pipeline which_label=0 only since its only 1 class
def count_blobs_vizz(mask, size_threshold=1,which_label=0):
    #are there too many labels? check how many to see if this can be optimized
    mask = np.where((mask == 0) | (mask == which_label), 0, mask)
    labels = measure.label(mask, connectivity=2, background=0)
    
    new_mask = np.zeros(mask.shape, dtype='uint8')
    labeled_mask = np.zeros(mask.shape, dtype='uint16')
    sizes = []
    # print(np.unique(mask,return_counts=True))
    
    
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

import numpy as np

def color_mask_by_label(maskArray, color):
    """
    Assign a single color to all non-zero label values in a mask array,
    based on the value of 'color' (which can be either 0 or 1).

    Parameters:
        maskArray: 2D NumPy array with label values (e.g., 0 for background, 1, 2, etc. for different regions).
        color: Integer value (0 or 1) determining which color to use.
               - If color == 0, use red ([255, 0, 0]).
               - If color == 1, use green ([0, 255, 0]).

    Returns:
        rgb_maskArray: 3D RGB NumPy array with the specified color assigned to all non-zero labels.
    """
    # Define the color mapping based on the value of 'color'
    if color == 1:
        rgb_color = np.array([255, 0, 0], dtype=np.uint8)  # Red
    elif color == 2:
        rgb_color = np.array([0, 255, 0], dtype=np.uint8)  # Green
    else:
        raise ValueError("Invalid color value. 'color' must be 1 or 2.")
    
    # Create an empty RGB array (height x width x 3)
    rgb_maskArray = np.zeros((*maskArray.shape, 3), dtype=np.uint8)
    
    # Create a boolean mask where maskArray is not zero
    non_zero_mask = maskArray != 0
    
    # Assign the specified color to all positions where the mask is non-zero
    rgb_maskArray[non_zero_mask] = rgb_color  # Vectorized assignment
    
    print(f"Colored array with color: {rgb_color.tolist()}")
    return rgb_maskArray

    
def saveUniqueMaskImage(maskArray, save_dir) :
    '''
    Plots post-processed detected 
    with the diversity of Colour distingushing
    the density of Plaques
    
    ie. More Diversity of Colour
    == More Plaque Count for that certain Plaque type
    
    Inputs:
        maskArray = Numpy Array containing Unique plaque
        save_dir  = String for Save Directory
    '''
    # print(maskArray.shape,np.unique(maskArray),np.count_nonzero(maskArray))
    # exit()
    max_val = np.amax(np.unique(maskArray))
#     print("Maximum Value = ", max_val)
    maskArray = np.asarray(maskArray, dtype=np.float64)
    maskArray = np.repeat(maskArray[:,:, np.newaxis], 3, axis=2)

    for label in np.unique(maskArray) :

        # For label 0, leave as black color (BG)
        if label == 0:
            continue

        idx = np.where(maskArray[:,:,0] == label) 

        # For label, create HSV space based on unique labels
        maskArray[:,:,0].flat[np.ravel_multi_index(idx, maskArray[:,:,0].shape)] = label / max_val
        maskArray[:,:,1].flat[np.ravel_multi_index(idx, maskArray[:,:,1].shape)] = label % max_val
        maskArray[:,:,2].flat[np.ravel_multi_index(idx, maskArray[:,:,2].shape)] = 1

    rgb_maskArray = hsv2rgb(maskArray)
    rgb_maskArray = rgb_maskArray * 255
    rgb_maskArray = rgb_maskArray.astype(np.uint8) # PIL save only accepts uint8 {0,..,255}
    
    save_img = Image.fromarray(rgb_maskArray, 'RGB')
    save_img.save(save_dir)
    print("Saved at: " + save_dir)

import numpy as np
import cv2
import matplotlib.pyplot as plt

def apply_transparency(rgb_maskArray):
    """
    Apply transparency to the heatmap, making black pixels fully transparent.
    
    Parameters:
        rgb_maskArray: 3D NumPy array (heatmap with RGB values).
    
    Returns:
        RGBA image with transparency applied to black pixels.
    """
    # Create an alpha channel where black pixels are fully transparent
    if len(rgb_maskArray.shape) == 2:
        rgb_maskArray = np.stack([rgb_maskArray] * 3, axis=-1)
    alpha_channel = ~np.all(rgb_maskArray[:, :, :3] == [0, 0, 0], axis=2)
    alpha_channel = alpha_channel.astype(np.uint8) * 255  # Convert to 0 (transparent) or 255 (opaque)
    
    # Create an RGBA image by adding the alpha channel to the RGB heatmap
    rgba_maskArray = np.dstack((rgb_maskArray, alpha_channel))
    
    return rgba_maskArray

def overlay_images_with_transparency(nums, mask1, mask2,base_filename, num_dilations=3, dilation_kernel_size=5, save_path="/cache/Shivam/BrainSec-py/data_ONCE_uncorrupted/outputs/masked_nft_stride1536_2/"):
    """
    Overlay two mask arrays (mask1 and mask2) on a third image (nums) with transparency for black pixels.
    
    Parameters:
        nums: 3D NumPy array (original or segmented image).
        mask1: 3D NumPy array (first heatmap to overlay).
        mask2: 3D NumPy array (second heatmap to overlay).
        num_dilations: int, number of dilation iterations for heatmap emphasis.
        dilation_kernel_size: int, size of the kernel for dilation.
    
    Returns:
        Blended image with transparent heatmap overlay from both masks.
    """
    # Apply dilation to both heatmaps to emphasize small regions
    base_filename = base_filename
    kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    dilated_mask1 = cv2.dilate(mask1.astype(np.uint8), kernel, iterations=num_dilations)
    dilated_mask2 = cv2.dilate(mask2.astype(np.uint8), kernel, iterations=num_dilations)
    print("dilated both masks, shapes are",dilated_mask1.shape,dilated_mask2.shape)
    # Save dilated_mask1 as {filename}_ift.png
    inft_filename = save_path+f"{base_filename}_inft.png"
    Image.fromarray(dilated_mask1).save(inft_filename)
    print(f"Saved {inft_filename}")
    
    # Save dilated_mask2 as {filename}_prenft.png
    prenft_filename = save_path+f"{base_filename}_prenft.png"
    Image.fromarray(dilated_mask2).save(prenft_filename)
    print(f"Saved {prenft_filename}")
    
    # exit()

    
    # Apply transparency to the dilated heatmaps, making black pixels transparent
    rgba_mask1 = apply_transparency(dilated_mask1)
    rgba_mask2 = apply_transparency(dilated_mask2)
    print("both makss made transparent, shapes are",rgba_mask1.shape,rgba_mask2.shape)
    
    # Combine both masks by adding them up (if they overlap, their transparency will also overlap)
    combined_rgba_mask = np.maximum(rgba_mask1, rgba_mask2)
    print("Shape of combined_rgba mask is",combined_rgba_mask.shape)
    # Convert nums to a 4-channel RGBA image by adding an opaque alpha channel
    nums_rgba = np.dstack((nums, np.full((nums.shape[0], nums.shape[1]), 255, dtype=np.uint8)))
    print("Stacking of channels done, nums_rgba shape is",nums_rgba.shape)
    # Overlay the combined transparent heatmap on top of the nums image
    overlay = np.where(combined_rgba_mask[:, :, 3:] == 255, combined_rgba_mask, nums_rgba)
    print("overlay done")
    # Convert the overlay to a PIL Image and save it
    overlay_image = Image.fromarray(overlay.astype(np.uint8), 'RGBA')
    overlay_image.save(save_path+f"{base_filename}_blendedimg_stride1536_nooverlap.png")
    
    print(f"Image saved at {save_path}")
    return overlay
# Example usage with nums and rgb_maskArray
# nums: your segmented image (3D RGB image)
# rgb_maskArray: your heatmap (3D RGB image)

# Overlay the images with transparency for black pixels


def saveJointImage(masks, save_dir,seg,base_filename):
    nums,WM_cnt,GM_cnt = saveBrainSegImage(seg,save_dir)
    
    # print( np.unique(maskArray, return_counts=True))
    # exit()

    # Color both masks based on their labels
    colored_mask1 = color_mask_by_label(masks[0],1)#iNFTs
    colored_mask2 = color_mask_by_label(masks[1],2)#preNFTs
    
    # Overlay the two colored masks with transparency for black pixels onto the segmented brain image
    blended_image = overlay_images_with_transparency(nums, colored_mask1, colored_mask2,base_filename, num_dilations=3, dilation_kernel_size=5)
    
    return blended_image,WM_cnt,GM_cnt


nft_npy_output_folder  = os.listdir("/cache/Shivam/BrainSec-py/data_ONCE_uncorrupted/vizz_nft_outputs_stride1536_2")
nft_npy_output_folder = [ i for i in nft_npy_output_folder if i[-3:]=="npy" ]

# To create CSV containing WSI names for
# plaque counting at different regions
CSV_FILE ="/cache/Shivam/BrainSec-py/data_ONCE_uncorrupted/outputs/NFTscore/NFT_Density_stride1536_2vizz.csv"
file = pd.DataFrame({"WSI_ID": nft_npy_output_folder})
file.to_csv(CSV_FILE, index=False)
print('Index CSV:', CSV_FILE)

# Using existing CSV
file = pd.read_csv(CSV_FILE)
# filenames = list(file['WSI_ID'])

#imp next comment
img_class = ['iNFT','preNFT']
#switched this to iNFT and preNFT because whichlabel arg of count_blobs masks whatever label we pass to it, so instead of preNFT it will count iNFT at index=0 and viceversaa for index 1

# two hyperparameters (For Plaque-Counting)
# confidence_thresholds = [0.1, 0.95, 0.9]
# pixel_thresholds = [100, 1, 200]

new_file = file
new_file['WM_pixel_count']=0
new_file['GM_pixel_count']=0

for idx,(filepath_nft,filepath_brainseg) in enumerate(zip(nft_npy_output_folder,os.listdir("/cache/Shivam/BrainSec-py/data_ONCE_uncorrupted/brainseg/numpy"))):
    base_filename = filepath_nft[:5]
    # if base_filename!="1-573":
    #     continue
    filepath_brainseg = os.path.join("/cache/Shivam/BrainSec-py/data_ONCE_uncorrupted/brainseg/numpy",filepath_brainseg)
    filepath_nft = os.path.join("/cache/Shivam/BrainSec-py/data_ONCE_uncorrupted/vizz_nft_outputs_stride1536_2",filepath_nft)
    seg = np.load(filepath_brainseg)
    print("Paths",filepath_brainseg,filepath_nft)
    print(seg.shape)
    # exit()
    nft_output = np.load(filepath_nft)
    print("NFT output shape",nft_output.shape)
    print(nft_output.shape,nft_output[0].shape,nft_output[0][0][1])

    preNFT_indices = np.argwhere(nft_output[0, :, :] == 1)

    # Finding the indices where class 1 (iNFT) is 1
    iNFT_indices = np.argwhere(nft_output[1, :, :] == 1)

    print("Locations of preNFT (Class 0):", preNFT_indices,len(preNFT_indices))
    print("Locations of iNFT (Class 1):", iNFT_indices,len(iNFT_indices))
    # Check where preNFT and iNFT overlap
    overlap_indices = np.argwhere((nft_output[0, :, :] == 1) & (nft_output[1, :, :] == 1))

    print("Number of overlapping indices (both preNFT and iNFT):", len(overlap_indices))
    # exit()
    new_nft_output = np.zeros((nft_output.shape[1], nft_output.shape[2]), dtype=int)

    # Set locations for preNFT (class 0) to 1
    new_nft_output[nft_output[0, :, :] == 1] = 1

    # Set locations for iNFT (class 1) to 2
    new_nft_output[nft_output[1, :, :] == 1] = 2

    # Now result_array contains 1 for preNFT locations and 2 for iNFT locations
    # print(new_nft_output)
    stride = 32
    seg = np.repeat(np.repeat(seg, stride, axis=0), stride, axis=1)
    mask = []
    for index in [0,1]:        

        # Now, expanded_outputs has the shape (26112, 26112)
        print("Brain sec output shape",seg.shape)
        print("shape of expanded heatmap",seg.shape)  # Should print (26112, 26112)
            # saveBrainSegImage(expanded_outputs,"tempbrainseg.png")

        # mask = h[index] > confidence_threshold
        # mask = mask.astype(np.float32)

        # kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))

        # # Apply morphological closing, then opening operations 
        # opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        # closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        #for sina pipeline this needs to be just index==0
        print(index)
        save_img = "/cache/Shivam/BrainSec-py/data_ONCE_uncorrupted/outputs/masked_nft_stride1536_2/images/"+base_filename+"_"+img_class[index] + "_NFT.png"
        save_np = "/cache/Shivam/BrainSec-py/data_ONCE_uncorrupted/outputs/masked_nft_stride1536_2/numpy/"+base_filename+"_" +img_class[index] + "_NFT.npy"
        
        labels, new_mask, num_labels,labeled_mask = count_blobs_vizz(new_nft_output,size_threshold=1,which_label=index+1)
        # labeled_mask = np.load(save_np)
        counts, bg, wm, gm, unknowns = classify_blobs(labeled_mask, seg)
        print(new_mask.shape,num_labels,counts, bg, wm, gm, unknowns)
        # # continue

        
        np.save(save_np, labeled_mask)
        mask.append(labeled_mask)
        print("masks appended")
        # saveUniqueMaskImage(labeled_mask, save_img)# To show Colored Result
        preds = len(labels)
        
        # print(confidence_threshold, pixel_threshold)
        
        
        new_file.loc[new_file['WSI_ID'] == nft_npy_output_folder[idx], 'CNN_{}_count'.format(img_class[index])] = preds
        new_file.loc[new_file['WSI_ID'] == nft_npy_output_folder[idx], 'BG_{}_count'.format(img_class[index])] = bg
        new_file.loc[new_file['WSI_ID'] == nft_npy_output_folder[idx], 'GM_{}_count'.format(img_class[index])] = gm
        new_file.loc[new_file['WSI_ID'] == nft_npy_output_folder[idx], 'WM_{}_count'.format(img_class[index])] = wm
        new_file.loc[new_file['WSI_ID'] == nft_npy_output_folder[idx], '{}_no-count'.format(img_class[index])] = unknowns

    blended_image,WM_cnt,GM_cnt=saveJointImage(mask, save_img,seg,base_filename)
    new_file.loc[new_file['WSI_ID'] == nft_npy_output_folder[idx], 'WM_pixel_count']=WM_cnt
    new_file.loc[new_file['WSI_ID'] == nft_npy_output_folder[idx], 'GM_pixel_count']=GM_cnt

    new_file.to_csv("/cache/Shivam/BrainSec-py/data_ONCE_uncorrupted/outputs/masked_nft_stride1536_2/"+'vizcarra_nft_bigWMGMsize.csv', index=False)
    print("CSV file saved at","/cache/Shivam/BrainSec-py/data_ONCE_uncorrupted/outputs/masked_nft_stride1536_2/"+'vizcarra_nft_bigWMGMsize.csv')
#         saveMask(img_mask, save_img)  # To show Classification Result
    

    
new_file.to_csv("/cache/Shivam/BrainSec-py/data_ONCE_uncorrupted/outputs/masked_nft_stride1536_2/"+'vizcarra_nft_bigWMGMsize.csv', index=False)
print('CSVs saved ')

















def compare():
    # Save this script as read_image.py
    import czifile


    # exit()
    import zarr
    example_path ='/cache/Shivam/nft/raw_data/wsis/zarr/scale_1.0/1-102-Temporal_AT8.zarr'
    #'/cache/Shivam/nft/wsi_heatmaps2/NFTDetector/stride_1024/1-102-Temporal_AT8'
    # z_store = zarr.open(example_path, mode='r')

    import imageio
    import numpy as np

    # Path to the pickle file containing annotations
    import pickle
    from pathlib import Path
    import os
    import sys
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    import seaborn as sns
    import skimage
    import os, sys
    import zarr

    import cv2
    #parent directories
    import pickle
    nft_scratch = Path('/cache/Shivam/nft/')
    wsi_path = Path(nft_scratch, 'raw_data/wsis/zarr/scale_0.1')
    wsi_heatmap_path = Path(nft_scratch, 'wsi_heatmaps')

    #variables
    model_name = 'NFTDetector'#'e782o7db_NFT_trial-epoch=34-valid_loss=0.14.ckpt'
    # # wsi_name = '1-473-Temporal_AT8' #NOT INCLUDED IN STUDY
    wsi_name = '1-102-Temporal_AT8'
    


    #target paths
    wsi_target_path = Path(wsi_path, wsi_name).with_suffix('.zarr')
    wsi_heatmap_target_path =Path(wsi_heatmap_path, model_name, 'stride_1024', wsi_name)

    # z_wsi_arr = zarr.open(wsi_target_path, mode = 'r')
    # z_wsi_arr.info
    # # print("Attempting to open:", wsi_heatmap_target_path.resolve())

    # z_heatmap_arr = zarr.open(wsi_heatmap_target_path, mode = 'r')
    # z_heatmap_arr.info
    # center_x, center_y = z_wsi_arr.shape[0] // 2, z_wsi_arr.shape[1] // 2
    # half_width, half_height = 500, 500  # Define the size of the slice to view

    #create custom color map that makes <threshold vals fully transparent, and all else linearly increasing transparency
    threshold = 0.5
    colors = [(0, (0, 0, 0, 0)), (threshold, (0, 1.0, 0, 0.0)), (1.0, (0, 1, 0, 0.5))]
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)
    import gc
    def apply_transparency2(rgb):
        # Create an alpha channel where black pixels are fully transparent
        if len(rgb.shape) == 2:
            rgb = np.stack([rgb] * 3, axis=-1)
        alpha_channel = ~np.all(rgb[:, :, :3] == [0, 0, 0], axis=2)

        # Convert the alpha channel to 8-bit (0 or 255) for RGBA format
        alpha_channel = alpha_channel.astype(np.uint8) * 255

        # Create an RGBA image by adding the alpha channel
        rgba = np.dstack((rgb, alpha_channel))
        return rgba

    def overlay_wsi(wsi_arr, heatmap_arr,wsi_name, downsample_rate = 40, num_dilations = 3):
        wsi = wsi_arr[::downsample_rate, ::downsample_rate, :]
        imageio.imwrite(wsi_name+'_org.png', wsi)
        wsi = apply_transparency(wsi)

        heatmap = heatmap_arr[::downsample_rate, ::downsample_rate, :]

        print(heatmap.shape,"heatmap")
        non_zero_count = np.count_nonzero(heatmap)

        print(f'The number of non-zero elements is: {non_zero_count}')

        heatmap = cv2.resize(heatmap, (brainseg.shape[1], brainseg.shape[0]), interpolation=cv2.INTER_NEAREST)
        print(heatmap.shape,"heatmap")
        non_zero_count = np.count_nonzero(heatmap)

        print(f'The number of non-zero elements is: {non_zero_count}')
        if num_dilations > 0:
            kernel = np.ones((int(5), int(5)), np.uint8)
            mask = cv2.dilate(heatmap.astype(np.uint8), kernel, iterations=num_dilations)
        plt.imshow(wsi)
        del heatmap,wsi
        gc.collect()
        plt.imshow(mask, cmap = cmap)
        del mask
        gc.collect
        plt.axis('off')
        plt.savefig(wsi_name+'.png')
        # saveJointImage(mask,  data_ONCE/"+img_class[index] + "_sinaNFT.png",brainseg)

    def overlay_images_with_transparency2(nums, mask1, num_dilations=3, dilation_kernel_size=5, save_path="data_ONCE/"):
        """
        Overlay two mask arrays (mask1 and mask2) on a third image (nums) with transparency for black pixels.
        
        Parameters:
            nums: 3D NumPy array (original or segmented image).
            mask1: 3D NumPy array (first heatmap to overlay).
            mask2: 3D NumPy array (second heatmap to overlay).
            num_dilations: int, number of dilation iterations for heatmap emphasis.
            dilation_kernel_size: int, size of the kernel for dilation.
        
        Returns:
            Blended image with transparent heatmap overlay from both masks.
        """
        # Apply dilation to both heatmaps to emphasize small regions
        base_filename = "1-102"
        kernel = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
        dilated_mask1 = cv2.dilate(mask1.astype(np.uint8), kernel, iterations=num_dilations)
        
        # Save dilated_mask1 as {filename}_prenft.png
        prenft_filename = save_path+f"{base_filename}_SinaNft_stride32.png"
        Image.fromarray(dilated_mask1).save(prenft_filename)
        print(f"Saved {prenft_filename}")

        
        # Apply transparency to the dilated heatmaps, making black pixels transparent
        rgba_mask1 = apply_transparency2(dilated_mask1)
        
        
        # Ensure `nums` is a 4-channel image (RGBA)
        print(nums.shape,"ajdnjdnd")
        if nums.shape[2] == 3:  # If `nums` has 3 channels, add an alpha channel
            nums_rgba = np.dstack((nums, np.full((nums.shape[0], nums.shape[1]), 255, dtype=np.uint8)))
        else:
            nums_rgba = nums
            
        # Overlay the combined transparent heatmap on top of the nums image
        overlay = np.where(rgba_mask1[:, :, 3] == 255, rgba_mask1, nums_rgba)
        
        # Convert the overlay to a PIL Image and save it
        overlay_image = Image.fromarray(overlay.astype(np.uint8), 'RGBA')
        overlay_image.save(save_path+f"{base_filename}_sinablendedimg_stride32.png")
        
        print(f"Image saved at {save_path}")
        return overlay
    def count_blobs2(mask, size_threshold=20):
        #are there too many labels? check how many to see if this can be optimized
        labels = measure.label(mask, connectivity=2, background=0)
        new_mask = np.zeros(mask.shape, dtype='uint8')
        labeled_mask = np.zeros(mask.shape, dtype='uint16')
        sizes = []
        
        # loop over the unique components
        for label in tqdm(np.unique(labels)):
            # if this is the background label, ignore it
            if label == 0:
                continue
            # otherwise, construct the label mask and count the
            # number of pixels 
            labelMask = np.zeros(mask.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)
            
            # if the number of pixels in the component is sufficiently
            # large, then add it to our mask of "large blobs"
            if numPixels > size_threshold:
                sizes.append(numPixels)
                new_mask = cv2.add(new_mask, labelMask)

                labeled_mask[labels==label] = label

        num_labels = len(np.unique(labels))
        return sizes, new_mask, num_labels,labeled_mask
        

    def overlay_brainseg(wsi_arr, heatmap_arr,wsi_name, downsample_rate = 64, num_dilations = 3):
        
        brainseg = np.load("/cache/Shivam/BrainSec-py/data_ONCE/brainseg/numpy/1-102-Temporal_AT8.czi.npy")
        nums = saveBrainSegImage(brainseg,"data_ONCE/temp/temp.png")
        print(heatmap_arr.shape)
        heatmap = (heatmap_arr[::downsample_rate, ::downsample_rate] > 0.5) * 1.0
        heatmap = cv2.resize(heatmap, (brainseg.shape[1], brainseg.shape[0]), interpolation=cv2.INTER_NEAREST)
        print(heatmap.shape,nums.shape)
        print(heatmap.shape,"heatmap")
        
        # exit()
        sizes, new_mask, num_labels,labeled_mask = count_blobs2(heatmap, size_threshold=1)
        print( "resulta",num_labels,len(sizes))
        preds = len(sizes)
        counts, bg, wm, gm, unknowns = classify_blobs(labeled_mask, seg)
        scoring(preds, bg, wm, gm, unknowns)
        print(counts,preds, bg, wm, gm, unknowns)
        # exit()
        wsi = apply_transparency(nums)
        
    
        
        if num_dilations > 0:
            kernel = np.ones((int(5), int(5)), np.uint8)
            mask = cv2.dilate(heatmap.astype(np.uint8), kernel, iterations=num_dilations)
        plt.imshow(wsi)
        del heatmap,wsi
        gc.collect()
        plt.imshow(mask, cmap = cmap)
        del mask
        gc.collect
        plt.axis('off')
        plt.savefig(wsi_name+'sinaNFT.png')
        

    def scoring(preds, bg, wm, gm, unknowns):
        CSV_FILE ="/cache/Shivam/BrainSec-py/data_ONCE/outputs/NFTscore/WSI_CERAD_AREA.csv"
        # Using existing CSV
        file = pd.read_csv(CSV_FILE)
        filenames = list(file['WSI_ID'])
        img_class = ['NFT']

        

        new_file = file

        new_file['CNN_{}_count'.format(img_class[0])] = preds
        new_file['BG_{}_count'.format(img_class[0])] = bg
        new_file['GM_{}_count'.format(img_class[0])] = gm
        new_file['WM_{}_count'.format(img_class[0])] = wm
        new_file['{}_no-count'.format(img_class[0])] = unknowns
        new_file.to_csv("data_ONCE/"+'sina_nft.csv', index=False)
        
    def overlay_wsi_set(wsis, model_name, num_dilations = 3):
        for wsi_name in wsis:
            #target paths
            wsi_target_path = Path(wsi_path, wsi_name).with_suffix('.zarr')
            wsi_heatmap_target_path = Path(wsi_heatmap_path, model_name, 'stride_1024', wsi_name)

            wsi_arr = zarr.open(wsi_target_path, mode = 'r')
            heatmap_arr = zarr.open(wsi_heatmap_target_path, mode = 'r')
            print("jjjj",heatmap_arr.shape,wsi_arr.shape)
            exit()
            # wsi_arr = wsi_arr[:,:]
            print()
            heatmap_arr = heatmap_arr[:,:]
            
            plt.figure(figsize = (40, 40))
            overlay_brainseg(brainseg, heatmap_arr, wsi_name,downsample_rate = 64, num_dilations = num_dilations)
            del heatmap_arr
            gc.collect()
    

    plt.figure(figsize = (40, 40))
    # wsis = ['1-297-Temporal_AT8','1-466-Temporal_AT8','1-473-Temporal_AT8','1-516-Temporal_AT8','1-573-Temporal_AT8','1-621-Temporal_AT8','1-693-Temporal_AT8','1-717-Temporal_AT8','1-756-Temporal_AT8','1-907-Temporal_AT8','1-102-Temporal_AT8', '1-154-Temporal_AT8', '1-271-Temporal_AT8',]
    wsis =['1-102-Temporal_AT8']
    print("length of all wsis",len(wsis))
    overlay_wsi_set(wsis, model_name, num_dilations = 4)
    # overlay_wsi(z_wsi_arr, z_heatmap_arr,wsi_name)

# compare()