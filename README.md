# BrainSec

Automated Grey and White Matter Segmentation in Digitized A*β*
Human Brain Tissue WSI. This is the implementation details for the paper: 
Z. Lai, L. Cerny Oliveira, R. Guo, W. Xu, Z. Hu, K. Mifflin, C. DeCarlie, S-C. Cheung, C-N. Chuah, and B. N. Dugger, "BrainSec: Automated Brain Tissue Segmentation Pipeline for Scalable Neuropathological Analysis," IEEE Access, 2022.

## BrainSec Authors / Contributors

- Runlin Guo
- Wenda Xu
- Zhengfeng Lai

If you have any questions/suggestions or find any bugs,
please submit a GitHub issue.

## Python Pipeline

- The Python port is based on the original Brainsec pipeline (with post-processing) as implemented in [Plaque_Quantification.ipynb](Plaque_Quantification.ipynb).

## Prerequisite: Apptainer

The apptainer definition file is [python_cuda_1211.def](python_cuda_1211.def).

`apptainer build [path/to/python_cuda_1211.sif] [path/to/python_cuda_1211.def]`

## Execution Example

The following example assumes the data folder is `data/`, and the input WSIs are in `data/wsi/`.

```shell
srun --account=[HPC account] --partition=[HPC partition] --time=[hh:mm:ss] --gres=gpu:[GPU count] --nodes=[node count] --ntasks-per-node=[tasks/node] --cpus-per-task=[cpus/task] --mem=[memory] --pty bash

apptainer run --nv --bind [path/to/BrainSec] [path/to/python_cuda_1211.sif]

python3 ./1_preprocessing.py --wsi_dir 'data/wsi/' --save_dir 'data/norm_tiles/'

python3 ./2_inference.py --img_dir 'data/norm_tiles/' --model_plaq 'models/CNN_model_parameters.pkl' --save_plaq_dir 'data/outputs/heatmaps/' --model_seg 'models/ResNet18_19.pkl' --save_img_dir 'data/brainseg/images/' --save_np_dir 'data/brainseg_test/numpy/'

python3 ./3_postprocessing.py --data_dir 'data/'
```
