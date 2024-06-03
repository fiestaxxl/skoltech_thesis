# Conditional Variational Autoencoder for molecular generation



## Overview

Conditional Variational Autoencoder is a versatile and effective moleculr model that can take conditional vector with desired molecular properties and generate new molecules. 

## Installation

- Clone this repository

```shell
git clone https://github.com/fiestaxxl/skoltech_thesis.git
```

- Install the dependencies

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

- Raw smiles data is stored at smiles.txt


## Training 

use train.py for training of the model:

```shell
train.py    --prop_file smiles_prop.txt                         #path to smiles file with properties
            --num_epochs 10                                     #number of training epochs
            --save_dir 'path_to_save_checkpoints'               #path to save checkpoints
```


## Generation
```shell
generate.py --smiles 'CCCCC'                  #smiles of parent molecule for latent vector, don't use if interested in random sampling
            --use_parent_prop                 #don't use if want random properties initialization
            --target_props '150 1 12 1 1 0 0' #target properties
            --num_iter 200                    #number of trials
            --checkpoint '../checkpoints/model_8props5.ckpt-5' #path to checkpoint file
```


## Contact

Shengjie Luo (ivan.gurev@skoltech.ru)

Sincerely appreciate your suggestions on our work!

