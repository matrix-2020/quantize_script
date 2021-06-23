import tensorflow as tf
print(tf.__version__)
from tensorflow.python.platform import gfile
import numpy as np
import cv2
import glob
from tqdm import tqdm
import scipy.stats
import copy
import multiprocessing

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#--------------------parameter need to be specified --------------------------------#
model_path = 'config/model_deeplab.pb'
dataset_path = 'config/calibration_image/'
suffix = '*.jpg'
input = 'ImageTensor'
input_shape = [513,513,3]

output = 'ResizeBilinear_3'



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
print('--------------total image num:{} '.format(len(img_list)))

not_list= ['Const','Identity','Assert','ExpandDims','Shape','Cast','Squeeze',
           'StridedSlice','Pad','Pack','ArgMax','Equal','Reshape','GreaterEqual',
           'LogicalAnd','Sub','Maximum']


#-------pre-process for dataset, need modified accordingly-------------#
def preprocess(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (input_shape[0], input_shape[1]), interpolation=cv2.INTER_CUBIC)
    img = img.astype(np.float) / 255.0
    img = img[np.newaxis, :]
    return img

#--------------------satuaration quantize----------------------#
def threshold_distribution(distribution, target_bin=128):
    def _smooth_distribution(p, eps=0.0001):
        is_zeros = (p == 0).astype(np.float32)
        is_nonzeros = (p != 0).astype(np.float32)
        n_zeros = is_zeros.sum()
        n_nonzeros = p.size - n_zeros
        if not n_nonzeros:
            raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
        eps1 = eps * float(n_zeros) / float(n_nonzeros)
        assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
        hist = p.astype(np.float32)
        hist += eps * is_zeros + (-eps1) * is_nonzeros
        assert (hist <= 0).sum() == 0
        return hist

    distribution = distribution[1:]
    length = distribution.size
    threshold_sum = sum(distribution[target_bin:])
    kl_divergence = np.zeros(length - target_bin)

    for threshold in range(target_bin, length):
        sliced_nd_hist = copy.deepcopy(distribution[:threshold])

        # generate reference distribution p
        p = sliced_nd_hist.copy()
        p[threshold - 1] += threshold_sum
        threshold_sum = threshold_sum - distribution[threshold]

        # is_nonzeros[k] indicates whether hist[k] is nonzero
        is_nonzeros = (p != 0).astype(np.int64)
        #
        quantized_bins = np.zeros(target_bin, dtype=np.int64)
        # calculate how many bins should be merged to generate
        # quantized distribution q
        num_merged_bins = sliced_nd_hist.size // target_bin

        # merge hist into num_quantized_bins bins
        for j in range(target_bin):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()
        quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()

        # expand quantized_bins into p.size bins
        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        for j in range(target_bin):
            start = j * num_merged_bins
            if j == target_bin - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)
        # q[p == 0] = 0
        #p = _smooth_distribution(p)
        #q = _smooth_distribution(q)
        p[p == 0] = 0.0001
        q[q == 0] = 0.0001

        # calculate kl_divergence between q and p
        kl_divergence[threshold - target_bin] = scipy.stats.entropy(p, q)

    min_kl_divergence = np.argmin(kl_divergence)
    threshold_value = min_kl_divergence + target_bin

    return threshold_value


#--------------------load graph----------------------------------#
with gfile.FastGFile(model_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="")

    flag = 0
    for op in graph.get_operations():
        if op.type not in not_list:
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
    input_tensor_index = node_list.index(input + ':0')



    # --------------------inference iter on each image to get min/max for  each layer----------------------#
    def infer_quantize(image_list):
        with tf.Session(graph=graph) as sess:
            max_array_local = np.ones(len(node_list), dtype=np.float) * -10000
            for i in range(len(image_list)):
                print('---------process image {}/{}'.format(i,len(image_list)))
                img0 = cv2.imread(image_list[i])
                img = preprocess(img0)
                out = sess.run(node_list[1:], feed_dict={input_tensor: img})

                max_array = np.zeros(len(node_list),dtype=np.float)
                max_array[0] = np.max(img)

                #---------------calculate min/max for each layer
                for j,blob_data in enumerate(out):
                    print('---------satuaration quantize {}/{} image for {}/{} node:{}'.format(i,len(image_list),j+1,len(node_list),node_list[j+1]))
                    if np.all(blob_data == 0):
                        print('---Warning, output are all 0!!!')
                    abs_data = np.fabs(blob_data)
                    max_val = np.max(abs_data)
                    hist, hist_edge = np.histogram(blob_data, bins=2048, range=(0, max_val))
                    threshold = threshold_distribution(hist, target_bin=128)
                    threshold_value = threshold * max_val/2048

                    max_array[j+1] = threshold_value
                   

                #----------------update min/max for local
                max_array_local = np.where(max_array_local > max_array, max_array_local, max_array)
            
            
            return max_array_local

    print('----------------total processor------: {}'.format(multiprocessing.cpu_count()))
    used_processor =  max(multiprocessing.cpu_count() -4, 1)
    #used_processor = 4
    slice_num = len(img_list) // used_processor
    seg_img_list = []
    for i in range(0,len(img_list),slice_num):
      seg_img_list.append(img_list[i:i+slice_num])

    with multiprocessing.Pool(used_processor) as pool:
      results = pool.map(infer_quantize, seg_img_list)

    max_array_global = np.ones(len(node_list), dtype=np.float) * -10000
    for max_local in results:
      max_array_global = np.where(max_array_global >= max_local, max_array_global, max_local)



input_max = max_array_global[input_tensor_index]
print('----input max {}'.format(input_max))

output_max = max_array_global[output_tensor_index]
print('----output max {}'.format(output_max))



# ----------------write min/max to dict and save----------------------#
for i in range(len(node_list)):
    max_dict[tensor_fullname_list[i]] = np.array([max_array_global[i]], dtype=np.float)
    min_dict[tensor_fullname_list[i]] = np.array([0], dtype=np.float)

#print(max_dict)
#print(min_dict)

np.save('config/satuaration_max_dict.npy',max_dict)
np.save('config/satuaration_min_dict.npy',min_dict)

print('------------------finished--------------------')
