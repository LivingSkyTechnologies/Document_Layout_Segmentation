import os
import imageio
import multiprocessing
import sys
import glob
import random
import cv2
import pickle

from models.gated_scnn.gated_shape_cnn.training.utils import flat_label_to_edge_label

import matplotlib.pyplot as plt
import numpy as np

from utils.AnnotationUtils import write_publaynet_masks
from utils.DataLoaderUtils import stratify_train_test_split

#%%
# Hard code this for now, we can't rely on annotation files for this info
TAG_NAMES = {'title',
             'text',
             'table',
             'figure',
             'list',
             'background'}

TAG_MAPPING = {}

SAVED_TRAIN_PKL_FILE = 'saved_publaynet_train_paths.pkl'
SAVED_VAL_PKL_FILE = 'saved_publaynet_val_paths.pkl'

class PubLayNetRaw:
    """
    Process the publaynet dataset under data_dir, process it to
    produce edge segmentations, and provide a self.dataset_paths() method
    for accessing the processed paths inside of the actual tf.Dataset class.

    Should only have to call build_edge_segs, and use the dataset_paths() inside
    of the dataset class
    """
    def __init__(self, data_dir, seed):
        """
        :param data_dir str where your dad data lives:
        """
        self.data_dir = data_dir
        self.seed = seed
        
        self.train_paths, self.valid_paths, self.test_paths, self.num_classes = self.build_and_split(self.seed)
        
    def _write_masks(self, dataset_dir, is_val_set=False, draw_border=True, force=False):
        anno_dir = dataset_dir
        pkl_file = SAVED_VAL_PKL_FILE if is_val_set else SAVED_TRAIN_PKL_FILE
        if os.path.exists(pkl_file) and not force:
            used_tags = pickle.load(open(pkl_file, 'rb'))[0]
        else:
            used_tags = {}
            anno_path = os.path.join(anno_dir, "val.json") if is_val_set else os.path.join(anno_dir, "train.json")
            used_tags = write_publaynet_masks(anno_path, is_val_set=is_val_set, draw_border=draw_border)
            pickle.dump((used_tags, ), open(pkl_file, 'wb'))

        return used_tags, {1: 'text', 2: 'title', 3: 'list', 4: 'table', 5: 'figure', 0: 'background'}
 
    def build_and_split(self, seed, draw_border=True, force=False):
        train_dir = os.path.join(self.data_dir, "train")
        val_dir = os.path.join(self.data_dir, "val")

        used_train_tags, class_mapping = self._write_masks(self.data_dir, is_val_set=False, draw_border=draw_border, force=force)
        used_val_tags, class_mapping = self._write_masks(self.data_dir, is_val_set=True, draw_border=draw_border, force=force)
        
        train_paths = [os.path.join(train_dir, x) for x in os.listdir(train_dir) if x.endswith('.jpg')]

        valid_paths = [os.path.join(val_dir, x) for x in os.listdir(val_dir) if x.endswith('.jpg')]

        valid_paths, test_paths = stratify_train_test_split(used_val_tags, 0.50, seed=self.seed)
        
        return train_paths, valid_paths, test_paths, len(class_mapping)

    #####################################################
    # getting the correct paths relative to datadir
    #####################################################

    def get_img_paths(self, split):
        if split == "train":
            return self.train_paths
        elif split == "test":
            return self.test_paths
        elif split == "valid":
            return self.valid_paths
        else:
            raise ValueError("Requested splits must be train, test, or valid.")

    def _convert_item_path_to_training_paths(self, p):
        img_path = p
        
        label_path = p.replace("jpg", "png")

        edge_dir = p.replace("train", "edges") if "train" in p else p.replace("val", "edges")
        if not os.path.exists(edge_dir):
            os.makedirs(edge_dir)
        edge_label_path = os.path.join(edge_dir, os.path.basename(p).replace("jpg", "png"))
    
        return img_path, label_path, edge_label_path

    def dataset_paths(self, split):
        img_paths = self.get_img_paths(split)
        dataset = [self._convert_item_path_to_training_paths(p) for p in img_paths]
        return dataset

    ####################################################################
    # create edge labels in the label directory of the PubLayNet data
    ####################################################################

    def _create_edge_map_from_path(self, path):
        _, label_path, edge_path = self._convert_item_path_to_training_paths(path)
        #if not os.path.exists(edge_path):
        label = cv2.imread(label_path)[:, :, -1]
        edge_label = flat_label_to_edge_label(label, self.num_classes)
        imageio.imsave(edge_path, edge_label)

    def build_edge_segs(self):
        p = multiprocessing.Pool(4)
        
        # Force re-write with no borders
        self.train_paths, self.valid_paths, self.test_paths, self.num_classes = self.build_and_split(self.seed, draw_border=False, force=True)
        
        image_paths = self.get_img_paths("train")
        image_paths += self.get_img_paths("valid")
        image_paths += self.get_img_paths("test")
        num_ps = len(image_paths)
        print('creating edge maps')
        for i, _ in enumerate(p.imap_unordered(self._create_edge_map_from_path, image_paths), 1):
            sys.stderr.write('\rdone {0:%}'.format(i / num_ps))

        # Re-write original masks
        self.train_paths, self.valid_paths, self.test_paths, self.num_classes = self.build_and_split(self.seed, force=True)

