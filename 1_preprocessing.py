#!/usr/bin/env python
# coding: utf-8

# ### 1 Preprocessing - Reinhard Normalization and WSI Tiling
# 
# As a first preprocessing step, all slides were color normalized with respect to a reference image selected by an expert neuropathologist. Color normalization was performed using the method described by [Reinhard et. al](https://ieeexplore.ieee.org/document/946629).
# 
# The resulting color normalized whole slide images were tiled using PyVips to generate 1536 x 1536 images patches.
import sys
sys.path.append("/cache/plaquebox-paper/")

import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyvips as Vips
import pandas as pd
import argparse
import time
import gc
from tqdm import tqdm
import czifile

from utils import vips_utils, normalize

def grabCZI(path, verbose = False):
    img = czifile.imread(path)
    if verbose:
        print(img.shape)
        print(img)
    
    img = np.array(img, dtype = np.uint8)
    
    scenes = img.shape[0]
    time = img.shape[1]
    height = img.shape[2]
    width = img.shape[3]
    channels = img.shape[4]
    
    
    img = img.reshape((height, width, channels))
    if verbose:
        print(img)
        print(img.shape) 
        
    dtype_to_format = {
        'uint8': 'uchar',
        'int8': 'char',
        'uint16': 'ushort',
        'int16': 'short',
        'uint32': 'uint',
        'int32': 'int',
        'float32': 'float',
        'float64': 'double',
        'complex64': 'complex',
        'complex128': 'dpcomplex',
    }
    
    ###codes from numpy2vips
    height, width, bands = img.shape
    img = img.reshape(width * height * bands)
    vips = Vips.Image.new_from_memory(img.data, width, height, bands,
                                      dtype_to_format['uint8'])
    try: 
        del img, height, width, bands
        gc.collect()
    except: 
        pass
    return vips


def tile(WSI_DIR, SAVE_DIR, NORMALIZE):
    # wsi = [pathname.split('/')[-1] for pathname in glob.glob(WSI_DIR+"*.svs")]
    wsi = [pathname.split('/')[-1] for pathname in glob.glob(WSI_DIR+"*.czi")]
    imagenames = sorted(wsi)
    print("##",wsi)
    normalizer = normalize.Reinhard()
    if NORMALIZE:
        ref_imagename = imagenames[0]
        #print(imagenames, ref_imagename)
        ref_image = Vips.Image.new_from_file(WSI_DIR + ref_imagename, level=0)
        normalizer.fit(ref_image)

    # czi_img = czifile.imread("/cache/Shivam/R01 Batch 1/1-102-Temporal_AT8.czi")
    # vips_img = grabCZI("/cache/Shivam/R01 Batch 1/1-102-Temporal_AT8.czi")#WSI_DIR + imagename)
    # print("Loaded Image: " +"/cache/Shivam/R01 Batch 1/1-102-Temporal_AT8.czi")
    
    # print("Pre resize: ", vips_img.height, "x", vips_img.width)
    # vips_img = vips_img.resize(0.5)
    # print("after resize: ", vips_img.height, "x", vips_img.width)
    

    # vips_utils.save_and_tile(vips_img, os.path.splitext(imagename)[0], SAVE_DIR, tile_size = TILE_SIZE)

    
    # del vips_img
    # gc.collect()
    # print("Finish Delete")
    # print("Czi image",czi_img,czi_img.shape,"\n\n ###### \n\n")
    # scene = np.squeeze(czi_img)
    # print("scence",scene,scene.shape)
    # non_zero_count = np.count_nonzero(czi_img)
    # width,height,bands = scene.shape[0],scene.shape[1],scene.shape[2]
    # # scene = scene.reshape(width * height * bands)
    # scene = (scene * 255).astype(np.uint8)
    # vips_image = Vips.Image.new_from_memory(scene.data, width, height, bands, 'uchar')
    # print(vips_image,vips_image.width)
    # tile_size=1536
    # Vips.Image.dzsave(vips_img, SAVE_DIR,
    #                     layout='google',
    #                     suffix='.jpg[Q=90]',
    #                     tile_size=tile_size,
    #                     depth='one',
    #                     properties=True)
    # print("Done Tiling: ", "/cache/Shivam/R01 Batch 1/1-102-Temporal_AT8.czi")
    # exit()

    stats_dict = {}
    print("Starting tiling....")
    for imagename in tqdm(imagenames[:]):
        start = time.time()
        print("dnsjdnjcndjcn",imagename)
        if imagename=='1-154-Temporal_4G8.czi' or imagename=='1-154-Temporal_AT8.czi':# or imagename=='1-102-Temporal_4G8.czi' or imagename=='1-102-Temporal_AT8.czi'  :
            continue
        if '.svs' == imagename[-4:]:
            print(imagename,"sd")
            vips_img = Vips.Image.new_from_file(WSI_DIR + imagename, level=0)
        else:
            print(imagename,"aad")
            vips_img = grabCZI(WSI_DIR + imagename)
        print("____________________________________________")
        print("Loaded Image: " + WSI_DIR + imagename)
        print("Before Width x Height: ", vips_img.width, "x", vips_img.height)
        flag = "ONCE"
        if flag=="SQR":
            print("resizing with sqr")
            vips_img = vips_img.resize((0.11/0.5)**2)
        elif flag=="ONCE":
            print("resizing with once")
            vips_img = vips_img.resize((0.11/0.5))
        else:
            print("no resizing")
        # Vips.Image.new_from_file(WSI_DIR + imagename, level=0)
        print("____________________________________________")
        print("Loaded Image: " + WSI_DIR + imagename)
        print("After Width x Height: ", vips_img.width, "x", vips_img.height)
        if NORMALIZE:
            out = normalizer.transform(vips_img)
            vips_utils.save_and_tile(out,imagename, SAVE_DIR)
            os.rename(os.path.join(SAVE_DIR, out.filename), os.path.join(SAVE_DIR, os.path.basename(vips_img.filename).split('.svs')[0]))
            stats_dict[imagename] = normalizer.image_stats
            try:
                del out
            except:
                pass
        else:
            try:
                out = vips_img
                vips_utils.save_and_tile(vips_img, imagename,SAVE_DIR)
            except:
                print("could not oxmplete tiling")
                continue
            #os.rename(os.path.join(SAVE_DIR, out.filename), os.path.join(SAVE_DIR, os.path.basename(vips_img.filename).split('.svs')[0]))
            #stats_dict[imagename] = normalizer.image_stats
        try:
            del vips_img
            gc.collect()
        except:
            pass
        
        print("processed in ", time.time()-start," seconds")
    
    if NORMALIZE:
        stats = pd.DataFrame(stats_dict)
        stats = stats.transpose()
        stats.columns = 'means', 'stds'
        #print(stats)
        stats.to_csv(SAVE_DIR + "stats.csv")
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi_dir", type=str, default='data/wsi/', help="Directory of Whole Slide Images")
    parser.add_argument("--save_dir", type=str, default='data/norm_tiles/', help="Directory to save the patches for heatmaps")
    parser.add_argument("--normalize", type=bool, default=False)

    args = parser.parse_args()

    print(f"WSI Directory: {args.wsi_dir}")
    print(f"Save Directory: {args.save_dir}")
    print(f"Normalize: {args.normalize}")


    WSI_DIR = args.wsi_dir #'data_1_40/wsi/'
    SAVE_DIR = args.save_dir #'data_1_40/norm_tiles/'
    NORMALIZE = args.normalize


    if not os.path.exists(WSI_DIR):
        print("WSI folder does not exist, script should stop now")
    else:
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)

        print("Found WSI folder... ")
        wsi_files = os.listdir(WSI_DIR)
        filenames = sorted(wsi_files)
        print("All files in wsi_dir: ")
        print(filenames)

    #----------------------------------------------------------

    tile(WSI_DIR, SAVE_DIR, NORMALIZE)
    print("____________________________________________")
    print("WSI tiled for heatmaps")

if __name__ == "__main__":
    main()