import pickle
import os
import tensorflow as tf

import utils.DataLoaderUtils as dlu
from utils.AnnotationUtils import write_publaynet_masks

# Static Dataset Config Options
TAG_NAMES = {'title',
             'list',
             'table',
             'figure',
             'text',
             'background'}

TAG_MAPPING = {}

SAVED_TRAIN_PKL_FILE = 'saved_publaynet_train_paths.pkl'
SAVED_VAL_PKL_FILE = 'saved_publaynet_val_paths.pkl'

BUFFER_SIZE = 500
MASK_DIR = ""

def write_masks(dataset_dir, is_val_set=False):
    anno_dir = dataset_dir
    pkl_file = SAVED_VAL_PKL_FILE if is_val_set else SAVED_TRAIN_PKL_FILE
    if os.path.exists(pkl_file):
        used_tags = pickle.load(open(pkl_file, 'rb'))[0]
    else:
        print("Running full mask generation, this may take a long time.")
        used_tags = {}
        anno_path = os.path.join(anno_dir, "val.json") if is_val_set else os.path.join(anno_dir, "train.json")
        used_tags = write_publaynet_masks(anno_path, is_val_set=is_val_set)
        pickle.dump((used_tags, ), open(pkl_file, 'wb'))

    return used_tags, {1: 'text', 2: 'title', 3: 'list', 4: 'table', 5: 'figure', 0: 'background'}
 
def build_publaynet_dataset(dataset_dir, img_size, batch_size, seed, debug=False):
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')

    used_train_tags, class_mapping = write_masks(dataset_dir, is_val_set=False)
    used_val_tags, class_mapping = write_masks(dataset_dir, is_val_set=True)

    train_paths = [os.path.join(train_dir, x) for x in os.listdir(train_dir) if x.endswith('.jpg')]  

    valid_paths = [os.path.join(val_dir, x) for x in os.listdir(val_dir) if x.endswith('.jpg')]

    valid_paths, test_paths = dlu.stratify_train_test_split(used_val_tags, 0.5, seed=seed)

    train_dataset = tf.data.Dataset.from_tensor_slices(train_paths)
    train_dataset = train_dataset.map(lambda x: dlu.parse_image(x, 0, MASK_DIR), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_paths)
    valid_dataset = valid_dataset.map(lambda x: dlu.parse_image(x, 0, MASK_DIR), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
    test_dataset = test_dataset.map(lambda x: dlu.parse_image(x, 0, MASK_DIR), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    train = train_dataset.map(lambda x: dlu.load_image_train(x, img_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train = train.shuffle(buffer_size=BUFFER_SIZE, seed=seed, reshuffle_each_iteration=True)
    train = train.padded_batch(batch_size, drop_remainder=True, padded_shapes=([img_size, img_size, 3], [img_size, img_size, 1], [None, 4]))
    train = train.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    valid = valid_dataset.map(lambda x: dlu.load_image_test(x, img_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid = valid.padded_batch(batch_size, drop_remainder=True, padded_shapes=([img_size, img_size, 3], [img_size, img_size, 1], [None, 4]))
    valid = valid.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test = test_dataset.map(lambda x: dlu.load_image_test(x, img_size), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test = test.padded_batch(batch_size, drop_remainder=True, padded_shapes=([img_size, img_size, 3], [img_size, img_size, 1], [None, 4]))
    test = test.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train, valid, test, class_mapping
 
