#tiling
python 1_preprocessing_czi.py \
  --input_dir /home/shivam/braindata_repo/wsis/ \
  --output_dir /home/shivam/braindata_repo/norm_tiles/ \
  --um_per_px 0.5 \
  --tile_size 1536

#inference
python 2_inference_czi.py \
  --img_dir /home/shivam/braindata_repo/norm_tiles/ \
  --save_plaq_dir /home/shivam/braindata_repo/outputs/heatmaps/ \
  --save_img_dir  /home/shivam/braindata_repo/brainseg/images/ \
  --save_np_dir   /home/shivam/braindata_repo/brainseg/numpy/ \
  --plaquebox_root /home/shivam/plaquebox-paper \
  --normalization  /home/shivam/plaquebox-paper/utils/normalization.npy 

#counting
