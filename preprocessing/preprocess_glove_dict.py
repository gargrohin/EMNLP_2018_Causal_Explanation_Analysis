import csv
import pickle

glove_300_file=open("../../data/glove_6b/glove.6B.300d.txt")

glove_300_dict=dict()
for line in glove_300_file:
    line=line.split(' ')
    glove_300_dict[line[0]]=[float(dim) for dim in line[1:]]
print("300_dict_made")

pickle.dump(glove_300_dict, open("glove_300.dict","wb"))
