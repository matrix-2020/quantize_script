import numpy as np
import tensorflow as tf

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="face_detection_front.tflite")
interpreter.allocate_tensors()

op_list = interpreter.get_tensor_details()
#print('-----node number------',len(op_list))

max_dict = {}
min_dict = {}

for node in op_list:
    tensor_fullname =  node['name'] + "_0"
    #print(tensor_fullname)
    fake_max = np.random.rand()*50 + 50
    fake_min = -50-np.random.rand() * 50
    max_dict[tensor_fullname] = np.array([fake_max], dtype=np.float)
    min_dict[tensor_fullname] = np.array([fake_min], dtype=np.float)

print(max_dict)
np.save('fake_max_dict.npy',max_dict)
np.save('fake_min_dict.npy',min_dict)


print('----------finish-------------')
