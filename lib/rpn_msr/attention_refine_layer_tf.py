# in this file, python style
import numpy as np
from fast_rcnn.config import cfg



def attention_refine_layer(feat, att_map):
    # this function realizes channel-wise Hadamard matrix product operation.
    # input shape(1,H,W,C). attention_map shape(H,W)
    input_shape = feat.shape
    att_shape = att_map.shape
    print('=====input_shape={}\n att_shape={}\n'.format(input_shape, att_shape))

    if not input_shape.size == 4:
        raise RuntimeError('input_shape of feature maps is not 4-dim')
    assert att_shape[1]==input_shape[1]
    attention_map = att_map[att_shape[0],:,:,0]
    # output = np.zeros((input_shape[0], input_shape[1], input_shape[2], input_shape[3]),dtype=np.float32)
    # for j in range(input_shape[0]):
    #     for i in range(input_shape[3]):
    #         channel = feat[j,:,:,i]
    #         channel = np.array(channel)
    #         channel = np.reshape(channel,(input_shape[1],input_shape[2]))
    #         attention_map = np.array(attention_map)
    #         attention_map = np.reshape(attention_map, (att_shape[1],att_shape[2]))
    #         hadmd_product = channel * attention_map
    #         output[j,:,:,i] = hadmd_product
    #         print(i)
    # output = np.array(output)
    # output = output.astype(np.float32,copy=False)
    # print('attention map shape={}'.format(output.shape))
    # return output


    output = np.array([[]])
    r = 0
    for j in range(1):
        for i in range(512):
            channel = feat[j,:,:,i]
            channel = np.array(channel)
            # channel = np.reshape(channel,(input_shape[1],input_shape[2]))
            attention_map = np.array(attention_map)
            # attention_map = np.reshape(attention_map,(att_shape[1],att_shape[2]))
            hadmd_product = channel*attention_map
            if r==0:
                output = hadmd_product
            else:
                output = np.vstack((output,hadmd_product))
            print(i)
    output = np.reshape(output, (1,-1,-1,512))
    output = np.array(output)
    output = output.astype(np.float32, copy=False)
    return output