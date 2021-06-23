import numpy as np
import onnx
import onnxruntime
import torch
import glob
from tqdm import tqdm
import cv2
import os

#--------------------------global parameter--------------------
model_path = 'resnet50v2.onnx'
dataset_path = 'calibration_image/'
suffix = '*.jpg'
input = 'data'
output = 'resnetv24_dense0_fwd'
input_shape = [3,224,224]

#--------------------------limitation--------------------
# this script is only for one input and one output, other case, please modify code to fit it 

#--------------------------load model--------------------
onnx_model = onnx.load(model_path)
onnx.checker.check_model(onnx_model)

node_num = len(onnx_model.graph.node)
print('----------total node num: {}'.format(node_num))
print('----------output node num: {}'.format(len(onnx_model.graph.output)))


#--------------------------add node to onnx_model.graph.output--------------------
output_node = []
all_node = []
all_node_save_name = []


for node in onnx_model.graph.output:
    output_node.append(node.name)
    all_node.append(node.name)
    all_node_save_name.append(node.name+'_0')

for i in range(len(onnx_model.graph.node)):
    intermediate_layer_value_info = onnx.helper.ValueInfoProto()
    intermediate_layer_value_info.name = onnx_model.graph.node[i].name
    if intermediate_layer_value_info.name in output_node:
        continue
    onnx_model.graph.output.append(intermediate_layer_value_info)
    all_node.append(intermediate_layer_value_info.name)
    all_node_save_name.append(intermediate_layer_value_info.name+'_0')

print('----output node num after adding all node:{}'.format(len(onnx_model.graph.output))) # not include input node
model_new_name = os.path.splitext(model_path)[0] + '_new.onnx'
onnx.save(onnx_model, model_new_name)



#--------------------------inference--------------------
img_list = glob.glob(dataset_path + suffix)

#-------pre-process for dataset, need modified accordingly-------------#
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[1], input_shape[2]))
    img = img.astype(np.float32)
    #img = img.astype(np.float) / 127.5 - (1.0,1.0,1.0)
    img = np.transpose(img, (2,0,1))
    img = img[np.newaxis, :]
    return img


ort_session = onnxruntime.InferenceSession(model_new_name)
for i in range(len(ort_session.get_inputs())):
    print('-------input node name: {}'.format(ort_session.get_inputs()[i].name))


max_array_global = np.ones(len(all_node)+1, dtype=np.float) * -1000
min_array_global = np.ones(len(all_node)+1, dtype=np.float) * 1000

for i in tqdm(range(len(img_list))):
    img = cv2.imread(img_list[i])
    img = preprocess(img)

    ort_inputs = {ort_session.get_inputs()[0].name: img}
    ort_outs = ort_session.run(None, ort_inputs)


    max_array = np.zeros(len(all_node)+1,dtype=np.float)
    min_array = np.zeros(len(all_node)+1,dtype=np.float)

    #input node
    max_array[0] = np.max(img)
    min_array[0] = np.min(img)

    #---------------calculate min/max for each layer
    for j,x in enumerate(ort_outs):
        max_array[j+1] = np.max(x)  # index start from 1, 0 reserved for input node 
        min_array[j+1] = np.min(x)

    #----------------update min/max for global
    max_array_global = np.where(max_array_global > max_array, max_array_global, max_array)
    min_array_global = np.where(min_array_global < min_array, min_array_global, min_array)


input_max = max_array_global[0]
input_min = min_array_global[0]
print('----input max {}, min {}'.format(input_max,input_min))

output_index = all_node.index(ort_session.get_outputs()[0].name) #all node [], not include input node
output_max = max_array_global[output_index+1]
output_min = min_array_global[output_index+1]
print('----output max {}, min {}'.format(output_max,output_min))



# ----------------write min/max to dict and save----------------------#
max_dict = {}
min_dict = {}
for i in range(len(all_node)):
    max_dict[all_node_save_name[i]] = np.array([max_array_global[i+1]], dtype=np.float)
    min_dict[all_node_save_name[i]] = np.array([min_array_global[i+1]], dtype=np.float)

#add input node
max_dict[ort_session.get_inputs()[0].name+'_0'] = np.array([max_array_global[0]], dtype=np.float)
min_dict[ort_session.get_inputs()[0].name+'_0'] = np.array([min_array_global[0]], dtype=np.float)

#print(max_dict)
#print(min_dict)

np.save('real_max_dict.npy',max_dict)
np.save('real_min_dict.npy',min_dict)

print('------------------finished--------------------')
