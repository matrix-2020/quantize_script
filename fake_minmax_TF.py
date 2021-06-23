import tensorflow as tf
print(tf.__version__)
from tensorflow.python.platform import gfile
import numpy as np


sess = tf.Session()

max_dict = {}
min_dict = {}

with gfile.FastGFile('mono_depth.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    for i, node in enumerate(graph_def.node):
        #print(node.name)
        if node.op not in ['Const']:
            tensor_fullname =  node.name + "_0"
            #print(tensor_fullname)
            fake_max = np.random.rand()*20 + 20
            fake_min = -20-np.random.rand() * 20
            max_dict[tensor_fullname] = np.array([fake_max], dtype=np.float)
            min_dict[tensor_fullname] = np.array([fake_min], dtype=np.float)

np.save('fake_max_dict.npy',max_dict)
np.save('fake_min_dict.npy',min_dict)
