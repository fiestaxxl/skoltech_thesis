# Conditional Variational Autoencoder for molecular generation



## Overview

Conditional Variational Autoencoder is a versatile and effective moleculr model that can take conditional vector with desired molecular properties and generate new molecules. 

## Installation

- Clone this repository

```shell
git clone https://github.com/fiestaxxl/skoltech_thesis.git
```

- Install the dependencies (Using [Anaconda](https://www.anaconda.com/), tested with CUDA version 11.0)

```shell
cd ./skoltech_thesis/cvae
pip install rdkit pubchempy tensorflow_addons

```

## Checkpoints

| Number of properties | Update Date  | Download Link                                            |
| -------------------- | ------------ |--------------------------------------------------------  |
| 8 properties         | Mar 12, 2024 | https://drive.google.com/drive/folders/1HOzx99rCgsYvi_CC77Ym35HDTJtBFOid?usp=sharing |

```shell
# create paths to checkpoints for evaluation
mkdir checkpoints
# download the above model weights to ./checkpoints
```

## Datasets

- Preprocessed data with computed 8 properties (Molecular Weight, LogP, number of heteroatoms, number
of aromatic rings, number of aliphatic rings, number of primary amine groups, number of aromatic
nitrogen atoms and number of pyridine groups) is stored in smiles_prop.txt

-Raw smiles data is stored at smiles.txt

- You can also directly execute the evaluation/training code to process data from scratch.

## Evaluation

```shell
export data_path='./datasets/pcq-pos'                # path to data
export save_path='./logs/{folder_to_checkpoints}'    # path to checkpoints, e.g., ./logs/L12

export layers=12                                     # set layers=18 for 18-layer model
export hidden_size=768                               # dimension of hidden layers
export ffn_size=768                                  # dimension of feed-forward layers
export num_head=32                                   # number of attention heads
export num_3d_bias_kernel=128                        # number of Gaussian Basis kernels
export batch_size=256                                # batch size for a single gpu
export dataset_name="PCQM4M-LSC-V2-3D"				   
export add_3d="true"
bash evaluate.sh
```

## Training

```shell
# L12. Valid MAE: 0.0785
export data_path='./datasets/pcq-pos'               # path to data
export save_path='./logs/'                          # path to logs

export lr=2e-4                                      # peak learning rate
export warmup_steps=150000                          # warmup steps
export total_steps=1500000                          # total steps
export layers=12                                    # set layers=18 for 18-layer model
export hidden_size=768                              # dimension of hidden layers
export ffn_size=768                                 # dimension of feed-forward layers
export num_head=32                                  # number of attention heads
export batch_size=256                               # batch size for a single gpu
export dropout=0.0
export act_dropout=0.1
export attn_dropout=0.1
export weight_decay=0.0
export droppath_prob=0.1                            # probability of stochastic depth
export noise_scale=0.2                              # noise scale
export mode_prob="0.2,0.2,0.6"                      # mode distribution for {2D+3D, 2D, 3D}
export dataset_name="PCQM4M-LSC-V2-3D"
export add_3d="true"
export num_3d_bias_kernel=128                       # number of Gaussian Basis kernels
bash train.sh
```

Our model is trained on 4 NVIDIA Tesla A100 GPUs (40GB). The time cost for an epoch is around 10 minutes.

## Downstream Task -- (QM9)
Download the checkpoint: L12-old.pt
```shell
export ckpt_path='./L12-old.pt'                # path to checkpoints
bash finetune_qm9.sh
```

## Citation

If you find this work useful, please kindly cite following papers:

```latex
@article{luo2022one,
  title={One Transformer Can Understand Both 2D \& 3D Molecular Data},
  author={Luo, Shengjie and Chen, Tianlang and Xu, Yixian and Zheng, Shuxin and Liu, Tie-Yan and Wang, Liwei and He, Di},
  journal={arXiv preprint arXiv:2210.01765},
  year={2022}
}

@inproceedings{
  ying2021do,
  title={Do Transformers Really Perform Badly for Graph Representation?},
  author={Chengxuan Ying and Tianle Cai and Shengjie Luo and Shuxin Zheng and Guolin Ke and Di He and Yanming Shen and Tie-Yan Liu},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021},
  url={https://openreview.net/forum?id=OeWooOxFwDa}
}

@article{shi2022benchmarking,
  title={Benchmarking Graphormer on Large-Scale Molecular Modeling Datasets},
  author={Yu Shi and Shuxin Zheng and Guolin Ke and Yifei Shen and Jiacheng You and Jiyan He and Shengjie Luo and Chang Liu and Di He and Tie-Yan Liu},
  journal={arXiv preprint arXiv:2203.04810},
  year={2022},
  url={https://arxiv.org/abs/2203.04810}
}
```

## Contact

Shengjie Luo (luosj@stu.pku.edu.cn)

Sincerely appreciate your suggestions on our work!

## License

This project is licensed under the terms of the MIT license. See [LICENSE](https://github.com/lsj2408/Transformer-M/blob/main/LICENSE) for additional details.

