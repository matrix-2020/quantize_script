import numpy as np
import cv2
import glob


#input_shape = [224,224,3]
input_shape = [288,800,3]


#--------------------list img in dataset dir----------------------#
img_list = glob.glob('./dataset/*.jpg')
data_len = len(img_list)
print('---------datase len-------',data_len)

#data = np.zeros([data_len,224,224,3],dtype=np.float32)
data = np.zeros([data_len,input_shape[0],input_shape[1],input_shape[2]],dtype=np.float32)
label = []

for i in range(len(img_list)):
    img = cv2.imread(img_list[i])
    img = cv2.resize(img, (input_shape[1], input_shape[0]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = (img - 127.5) / 128
    data[i,:,:,:] = img
    label.append(int('1'))

labels = np.array(label)


np.save('calibration_data.npy',data)
np.save('calibration_label.npy',labels)

print('------------------finished--------------------')
