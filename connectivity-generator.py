import numpy as np
import input_reader
import gf_method
import tools

f = input_reader.orca_parser("h2o-h2o-h2o2")
int_coord = tools.get_redundant_internals(f.coord, f.atoms) 

fragments=[]
test=bonds
fragments.extend((test[0][0],test[0][1]))
test=np.delete(test, 0, axis=0)
test1=[]
all_fragments=[]
while len(test) != 0:
  if len(test1) != len(test):
    rows_to_delete=[]
    test1=test
    for i in range(len(test)):
      if np.in1d(fragments,test[i]).any() == True:
        fragments = np.unique(np.concatenate((fragments,test[i])))
        rows_to_delete.append(i)
    test = np.delete(test, rows_to_delete, axis=0)
  if len(test1) == len(test) or len(test) == 0:
    all_fragments.append(fragments)
    fragments = []
    if len(test) != 0:
      fragments.extend((test[0][0],test[0][1]))
      test=np.delete(test, 0, axis=0)
      test1=[]

new_bonds=[]
for i in range(len(all_fragments)):
    for j in range(len(all_fragments)):
        if i != j and i < j:
            temporary_storage=[]
            all_intramolecular_bonds = np.stack(np.meshgrid(all_fragments[i], all_fragments[j]), -1).reshape(-1, 2)
            for k in range(len(all_intramolecular_bonds)):
                bond_length = gf_method.length(f.coord[all_intramolecular_bonds[k][0]-1], f.coord[all_intramolecular_bonds[k][1]-1])
                temporary_storage.append((bond_length, all_intramolecular_bonds[k][0], all_intramolecular_bonds[k][1]))
            new_bonds.append(min(temporary_storage)[1:])

print(new_bonds)
