from collections import defaultdict

import numpy as np
from scipy.misc import imread
from scipy.misc import imresize
from tqdm import tqdm
FIRST_DIMENSION = 300
IMAGE_SIZE_DICT = {'square': (FIRST_DIMENSION, 300),
                   'landscape': (FIRST_DIMENSION, 420),
                   # 'imax': FIRST_DIMENSION, 650,
                   'laptop': (FIRST_DIMENSION, 480),
                   'wxga': (FIRST_DIMENSION, 530)
                   }

"""
            Aspect Ratio	Decimal	
Square          1:1	            1.0	
Landscape       5:4	            1.25	
IMAX            4:3	            1.333	
Laptop          8:5	            1.6	
WXGA           16:9	            1.777	

"""


def get_image_output_ratio(img):
    shape = img.shape
    ratio = float(shape[1]) / shape[0]
    "Orders matter here"
    if ratio <= 1:
        output_shape = 'square'

    # elif ratio <= 1.26:
    #     output_shape = 'landscape'

    elif ratio <= 1.33:
        output_shape = 'landscape'

    elif ratio <= 1.6:
        output_shape = 'laptop'

    else:
        output_shape = 'wxga'

    output_img_size = IMAGE_SIZE_DICT[output_shape]
    return output_img_size, output_shape


def preprocess_image_batch(image_paths, image_true_labels, img_size_dict={}):
    """

    :param image_paths: 
    :param image_true_labels: 
    :param img_size_dict: 
    :return: 
    """
    image_dict = defaultdict(list)
    image_lable_dict = defaultdict(list)

    for im_path, label in tqdm(zip(image_paths, image_true_labels),total=len(image_true_labels)):
        img = imread(im_path, mode='RGB')
        output_img_size, output_shape = get_image_output_ratio(img)
        img_size = (*output_img_size, 3)
        img = imresize(img, img_size)

        img = img.astype('float32')
        # We normalize the colors (in RGB space) with the empirical means on the training set
        img[:, :, 0] -= 123.68
        img[:, :, 1] -= 116.779
        img[:, :, 2] -= 103.939
        img = img.transpose((2, 0, 1))
        image_dict[output_shape].append(img)
        image_lable_dict[output_shape].append(label)
    try:
        img_batch = {key: np.stack(img_list, axis=0) for key, img_list in image_dict.items()}
    except:
        raise ValueError('when img_size and crop_size are None, images'
                         ' in image_paths must have the same shapes.')
    return img_batch, image_lable_dict

def get_image_resize_dimension(network_architecture_used):
    pass
