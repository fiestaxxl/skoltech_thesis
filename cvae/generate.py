import pickle
import numpy as np
from model import CVAE
from utils import *

import rdkit.Chem as Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcNumAromaticRings
from rdkit.Chem.rdMolDescriptors import CalcNumHeteroatoms
from rdkit.Chem.rdMolDescriptors import CalcNumAliphaticRings
from rdkit.Chem.Fragments import fr_NH2, fr_Ar_N, fr_pyridine

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--smiles', help='smiles of molecule', type=str, default=None)
parser.add_argument('--use_parent_prop', help='use properties of parent molecule', action='store_true')
parser.add_argument('--target_props', help='target properties', type=str, default='150 1 12 1 1 0 0')
parser.add_argument('--num_prop', help='number of properties', type=int, default=8)
parser.add_argument('--num_iter', help='number of rnn layer', type=int, default=200)
parser.add_argument('--batch_size', help='batch_size', type=int, default=1)
parser.add_argument('--latent_size', help='latent_size', type=int, default=200)
parser.add_argument('--unit_size', help='unit_size of rnn cell', type=int, default=512)
parser.add_argument('--n_rnn_layer', help='number of rnn layer', type=int, default=3)
parser.add_argument('--seq_length', help='max_seq_length', type=int, default=120)
parser.add_argument('--prop_file', help='name of property file', type=str, default='smiles_prop.txt')
parser.add_argument('--mean', help='mean of VAE', type=float, default=0.0)
parser.add_argument('--stddev', help='stddev of VAE', type=float, default=1.0)
parser.add_argument('--num_epochs', help='epochs', type=int, default=6)
parser.add_argument('--lr', help='learning rate', type=float, default=0.0001)
parser.add_argument('--device', help='device for train, CPU or GPU', type=str, default='GPU')
parser.add_argument('--vocab_path', help='path to vocab', type=str, default='vocab.pkl')
parser.add_argument('--chars_path', help='path to chars', type=str, default='chars.pkl')
parser.add_argument('--checkpoint', help='path to chekpoint', type=str, default='../drive/MyDrive/cvae_tf/model_7props5.ckpt-5')
args = parser.parse_args()

def generate(args, vocab, char):
    vocab_size = len(char)
    smiles_string=args.smiles

    model = CVAE(vocab_size,
                args
                )
    model.restore(args.checkpoint)

    if ((args.use_parent_prop) and (smiles_string is not None)):
        m = Chem.MolFromSmiles(smiles_string)
        AllChem.Compute2DCoords(m)
        args.target_props = string_mol_properties(m)
    elif (args.use_parent_prop) and (smiles_string is None):
        raise TypeError('Smile string is not provided')

    target_prop = np.array([[float(p) for p in args.target_props.split()] for _ in range(args.batch_size)])
    start_codon = np.array([np.array(list(map(vocab.get, 'X')))for _ in range(args.batch_size)])

    #generate smiles
    smiles = []
    #for _ in range(args.num_iteration):
    for _ in range(args.num_iter):
        if smiles_string is not None:
          vec, l = smiles2vec(smiles_string, vocab, args)
          latent_vector = model.get_latent_vector(vec,target_prop,[l]) + np.random.normal(args.mean/100, args.stddev/5, (args.batch_size, args.latent_size))
        else:
          latent_vector = s = np.random.normal(args.mean, args.stddev, (args.batch_size, args.latent_size))
        generated = model.sample(latent_vector, target_prop, start_codon, args.seq_length)
        smiles.append(convert_to_smiles(generated, char))

    #write smiles and calcualte properties of them
    print ('number of trial : ', len(smiles))
    smiles = list(set([s.split('E')[0] for s in smiles]    ))
    print ('number of generate smiles (after remove duplicated ones) : ', len(smiles))
    ms = [Chem.MolFromSmiles(s) for s in smiles]
    ms = [m for m in ms if m is not None]
    print ('number of valid smiles : ', len(ms))

    if smiles_string is not None:
        filename = args.target_props + smiles_string + '.txt'
    else:
        filename = args.target_props + 'random' + '.txt'

    with open(filename, 'w') as w:
        w.write('smiles\tMW\tLogP\tNumHetAtoms\tNumAromRings\tNumAliphaticRings\tRNH2\tAr_N\tPyr\n')
        for m in ms:
            try:
                w.write('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\n' %(Chem.MolToSmiles(m), ExactMolWt(m), MolLogP(m), CalcNumHeteroatoms(m), CalcNumAromaticRings(m), CalcNumAliphaticRings(m), fr_NH2(m), fr_Ar_N(m), fr_pyridine(m)))
            except:
                continue
    #return ms


if __name__ == '__main__':


    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    with open(args.chars_path, 'rb') as f:
        chars = pickle.load(f)

    generate(args, vocab, chars)
