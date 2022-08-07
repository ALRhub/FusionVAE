# FusionVAE: A Deep Hierarchical Variational Autoencoder for RGB Image Fusion

This is the official code for the paper "FusionVAE: A Deep Hierarchical Variational Autoencoder for RGB Image Fusion" by Fabian Duffhauss et al. accepted to ECCV 2022. The code allows the users to reproduce and extend the results reported in the study. Please cite the paper when reporting, reproducing or extending the results.

[[Arxiv](https://arxiv.org/abs/2208.01172)]


## Purpose of the project

This software is a research prototype, solely developed for and published as part of the publication "FusionVAE: A Deep Hierarchical Variational Autoencoder for RGB Image Fusion". It will neither be maintained nor monitored in any way.


## Requirements, test, install, use, etc.

The code was successfully tested with Python 3.7, PyTorch 1.8.1 and CUDA 10.2


### Installation

- Clone the NVAE repository to your workspace and install its requirements within a virtual environment
```shell
    cd path/to/workspace
    git clone https://github.com/NVlabs/NVAE.git
    cd NVAE
    pip install -r requirements.txt
```
- Clone this repository to your workspace next to the NVAE folder:
```shell
    cd path/to/workspace
    git clone https://github.com/ALRhub/FusionVAE
```


### Setup Datasets

#### FusionMNIST

The training and validation sets of FusionMNIST are not stored but are created automatically during training and validation 
based on the original MNIST dataset. MNIST will be downloaded automatically when running the first training. 


#### FusionCelebA

FusionCelebA needs to be generated before starting a training. The script `create_fusion_celeba_lmdb.py` automatically 
downloads the CelebA dataset and creates an LMDB folder for either the training split or the validation split:

```shell
    python create_fusion_celeba_lmdb.py  --split train --celeba_dir /path/to/celeba --output_dir /path/to/output/dir
    python create_fusion_celeba_lmdb.py  --split valid --celeba_dir /path/to/celeba --output_dir /path/to/output/dir
```


#### FusionT-LESS

FusionT-LESS can be downloaded [here](datasets). For training, there is a set of background images (128x128) and a set of overlay images (100x100) that can be randomly combined to generate diverse training samples. We also included our evaludation dataset.


### Setup Config File

The `config.py` contains all relevant hyperparameters. For FusionCelebA and FusionT-LESS the dataset paths need to be 
adapted to the path where the datasets are located.


### Changes of the NVAE

Due to the restrictiveness of the license of the NVAE repository, we were not allowed to publish the modified NVAE files. However, you can clone the NVAE repository as mentioned above and apply the following modifications:

1. In `model.py`, the cells that process the input images x as well as the target images y need to be extended to accept
    a stacked tensor. For that you can use the time-distributed layer `TimeDist` from `network_elements.py`.
    
    In `init_pre_process` and `init_encoder_tower`, you can insert
    ```python
    cell = TimeDist(cell, cell_type=cell.cell_type)
    ```
    for `'normal_pre'`, `'down_pre'`, `'normal_enc'`, and `'down_enc'`.
    
    In `init_stem` you can add
    ```python
    stem = TimeDistributed(stem)
    ```
    and in `init_encoder0`
    ```python
    cell = TimeDist(cell)
    ```
   
2. In `datasets.py` you need to exchange the data transform functions with the corresponding data transform functions in `utils.py`. A data processing pipeline for FusionT-LESS can be generated analogously to FusionMNIST and FusionCelebA. For the `train_queue`, the collate function `collate_fn_noisy_imgs` need to be added.
    
3. `module.py` needs to be adapted to perform the encoding and the aggregation mechanism illustrated in Fig. 2 of the paper. The function `aggregate` is used to aggregate features `s` wheras the function `aggregate_dist` is used to aggregate the means and variances of latent distributions. `s_skip_aggregate` is used only for the ablation study to examine the effect of skip connections.


### Training

For reproducing the main results of the paper you can run the following commands using three noisy/occluded input images. Please note that you have to run `train.py` of the NVAE repo with the aforementioned changes. Please read the NVAE documentation for further details.

```shell script
python train.py --dataset mnist --batch_size 800 --epochs 400 --num_channels_enc 8 --num_channels_dec 8 \
    --num_latent_scales 2 --num_groups_per_scale 5 --num_postprocess_cells 1 --num_preprocess_cells 1 \
    --num_cell_per_cond_enc 1 --num_cell_per_cond_dec 1 --num_latent_per_group 10 --num_preprocess_blocks 2 \
    --num_postprocess_blocks 2 --weight_decay_norm 1e-2 --num_nf 0 --ada_groups --num_process_per_node 2 --use_se \
    --res_dist --fast_adamax --learning_rate 1e-2

python train.py --dataset celeba_64 --batch_size 32 --epochs 90 --num_channels_enc 32 --num_channels_dec 32 \
    --num_latent_scales 3 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 \
    --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 20 --num_preprocess_blocks 1 \
    --num_postprocess_blocks 1 --weight_decay_norm 1e-1 --num_nf 0 --ada_groups --num_process_per_node 4 --use_se \
    --res_dist --fast_adamax --learning_rate 1e-2
    
python train.py --dataset tless --batch_size 32 --epochs 500 --num_channels_enc 32 --num_channels_dec 32 \
    --num_latent_scales 3 --num_groups_per_scale 10 --num_postprocess_cells 2 --num_preprocess_cells 2 \
    --num_cell_per_cond_enc 2 --num_cell_per_cond_dec 2 --num_latent_per_group 20 --num_preprocess_blocks 1 \
    --num_postprocess_blocks 1 --weight_decay_norm 1e-1 --num_nf 0 --ada_groups --num_process_per_node 2 --use_se \
    --res_dist --fast_adamax --learning_rate 1e-2
```


## Citation
If you use this work please cite
```
@InProceedings{Duffhauss_2022_ECCV,
    author    = {Duffhauss, Fabian and Vien, Ngo Anh and Ziesche, Hanna and Neumann, Gerhard},
    title     = {FusionVAE: A Deep Hierarchical Variational Autoencoder for RGB Image Fusion},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2022},
}
```



