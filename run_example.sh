#tiling
python 1_preprocessing_czi.py --path_wsi /home/shivam/braindata_repo/wsis/ --save_dir /home/shivam/braindata_repo/norm_tiles/ 

#inference
python 2_inference_czi.py \
  --img_dir /home/shivam/braindata_repo/norm_tiles/ \
  --save_plaq_dir /home/shivam/braindata_repo/outputs/heatmaps/ \
  --save_img_dir  /home/shivam/braindata_repo/brainseg/images/ \
  --save_np_dir   /home/shivam/braindata_repo/brainseg/numpy/ \
  --plaquebox_root /home/shivam/plaquebox-paper \
  --normalization  /home/shivam/plaquebox-paper/utils/normalization.npy 

#counting

python 3_postprocessing_nobraingsegpostprop.py --data_dir /home/shivam/braindata_repo
