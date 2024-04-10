import numpy as np
from rdkit.Chem import  Draw
import os
import rdkit.Chem as Chem

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--smiles', help='smiles of molecule', type=str, default=None)
parser.add_argument('--file_path', help='path to file with generated molecules', type=str, default=f'{os.path.join(os.getcwd(),"results")}')
parser.add_argument('--save_path', help='path to dirs with images', type=str, default=f'{os.path.join(os.getcwd(),"images")}')
args = parser.parse_args()

def plot_mols(file_path, save_path):
    f = open(file_path)
    lines = f.read().split('\n')[:-1]
    lines = [l.split() for l in lines]
    smile = [l[0] for l in lines][1:]
    smiles = [Chem.MolFromSmiles(s) for s in smile]

    img=Draw.MolsToGridImage(smiles,molsPerRow=6,subImgSize=(700,700),returnPNG=False)
    img_name = os.path.splitext(os.path.basename(file_path))[0] + '.png'
    filename = os.path.join(save_path, img_name)
    img.save(filename)

def plot_mol(smiles, save_path):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, returnPNG=False)
    img_name = smiles + '.png'
    filename = os.path.join(save_path, img_name)
    img.save(filename)

if __name__=='__main__':

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    plot_mols(args.file_path, args.save_path)
    if args.smiles is not None:
        plot_mol(args.smiles, args.save_path)