import tensorflow as tf
print(tf.__version__)
from tensorflow.python.platform import gfile
import numpy as np
import cv2
import glob
from tqdm import tqdm

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#--------------------parameter need to be specified --------------------------------#
model_path = 'config/model_face_landmark.pb'
dataset_path = 'config/calibration_image/'
suffix = '*.jpg'
input = 'input_2'
input_shape = [64,64,1]

output = 'conv_pw_13_relu/Relu6'


#--------------------general parameter--------------------------------#
node_num = 0
node_list = []
tensor_fullname_list = []

max_dict = {}
min_dict = {}

input_max = None
input_min = None


#--------------------list img in dataset dir----------------------#
img_list = glob.glob(dataset_path + suffix)


#-------pre-process for dataset, need modified accordingly-------------#
def preprocess(img):
    img = cv2.resize(img, (input_shape[0], input_shape[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.reshape(1, input_shape[0], input_shape[1], input_shape[2])
    return img


#--------------------load graph----------------------------------#
with gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")


    for op in graph.get_operations():
        if op.type not in ['Const','Identity']:

            node_num += 1
            tensor_fullname = op.name + "_0"
            tensor_fullname_list.append(tensor_fullname)
            #print(op.name, op.type)
            #print(tensor_fullname)
            tensor_node_name = op.name + ':0'
            node_list.append(tensor_node_name)
    print('------------total node num is {}'.format(node_num))

    input_tensor  = graph.get_tensor_by_name(input+':0')

    output_tensor_index = node_list.index(output + ':0')

    max_array_global = np.ones(len(node_list), dtype=np.float) * -1000
    min_array_global = np.ones(len(node_list), dtype=np.float) * 1000

    # --------------------inference iter on each image to get min/max for  each layer----------------------#
    with tf.Session(graph=graph) as sess:
        #----------iter on each image
        #for i in range(len(img_list)):
        for i in tqdm(range(len(img_list))):
            #print('---------process image {}/{}'.format(i,len(img_list)))
            img = cv2.imread(img_list[i])
            img = preprocess(img)
            out = sess.run(node_list, feed_dict={input_tensor: img})


            max_array = np.zeros(len(node_list),dtype=np.float)
            min_array = np.zeros(len(node_list),dtype=np.float)

            #---------------calculate min/max for each layer
            for j,x in enumerate(out):
                max_array[j] = np.max(x)
                min_array[j] = np.min(x)

            #----------------update min/max for global
            max_array_global = np.where(max_array_global > max_array, max_array_global, max_array)
            min_array_global = np.where(min_array_global < min_array, min_array_global, min_array)

            #break



input_max = max_array_global[0]
input_min = min_array_global[0]
print('----input max {}, min {}'.format(input_max,input_min))

output_max = max_array_global[output_tensor_index]
output_min = min_array_global[output_tensor_index]
print('---- output max {}, min {}'.format(output_max,output_min))




# ----------------write min/max to dict and save----------------------#
for i in range(len(node_list)):
    max_dict[tensor_fullname_list[i]] = np.array([max_array_global[i]], dtype=np.float)
    min_dict[tensor_fullname_list[i]] = np.array([min_array_global[i]], dtype=np.float)

#print(max_dict)
#print(min_dict)

np.save('config/real_max_dict.npy',max_dict)
np.save('config/real_min_dict.npy',min_dict)

print('------------------finished--------------------')
