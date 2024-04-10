from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.Crippen import MolLogP
from rdkit.Chem.rdMolDescriptors import CalcNumAromaticRings
from rdkit.Chem.rdMolDescriptors import CalcNumHeteroatoms
from rdkit.Chem.rdMolDescriptors import CalcNumAliphaticRings
from rdkit.Chem.Fragments import fr_NH2, fr_Ar_N, fr_pyridine

from rdkit import Chem
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--input_filename', help='filename for smiles', type=str, default='smiles.txt')
parser.add_argument('--output_filename', help='name of output file', type=str, default='smiles_prop.txt')
parser.add_argument('--ncpus', help='number of cpus', type=int, default=1)
args = parser.parse_args()

def cal_prop(s):
    m = Chem.MolFromSmiles(s)
    if m is None : return None
    return Chem.MolToSmiles(m), ExactMolWt(m), MolLogP(m), CalcNumHeteroatoms(m), CalcNumAromaticRings(m), CalcNumAliphaticRings(m), fr_NH2(m), fr_Ar_N(m), fr_pyridine(m)

with open(args.input_filename) as f:
    smiles = f.read().split('\n')[:-1]
pool = Pool(8)

r = pool.map_async(cal_prop, smiles)

data = r.get()
pool.close()
pool.join()
w = open(args.output_filename, 'w')
for d in data:
    if d is None:
        continue
    w.write(d[0] + '\t' + str(d[1]) + '\t'+ str(d[2]) + '\t'+ str(d[3]) + '\t'+ str(d[4])+ '\t'+ str(d[5])+ '\t'+ str(d[6]) + '\t'+ str(d[7])+'\t'+ str(d[8])+'\n')
w.close()