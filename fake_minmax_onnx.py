import onnx
from onnx import helper
import sys,getopt
import numpy as np


path= 'resnet50.onnx'
model = onnx.load(path)

node_num =  len(model.graph.node)
print('node num in model: {}'.format(node_num))
max_dict = {}
min_dict = {}

for i in range(len(model.graph.node)):
    node_name = model.graph.node[i].name
    node_fullname =  node_name + "_0"
    fake_max = np.random.rand()*20 + 20
    fake_min = -20-np.random.rand() * 20
    max_dict[node_fullname] = np.array([fake_max], dtype=np.float)
    min_dict[node_fullname] = np.array([fake_min], dtype=np.float)

np.save('fake_max_dict.npy',max_dict)
np.save('fake_min_dict.npy',min_dict)
