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

from utils.AnnotationUtils import write_dad_masks
from utils.DataLoaderUtils import stratify_train_test_split

#%%
# Hard code this for now, we can't rely on annotation files for this info
TAG_NAMES = {'highlights',
             'urls_to_supplementary',
             'abbreviation',
             'abstract',
             'additional_file',
             'affiliation',
             'appendice',
             'author_bio',
             'author_contribution',
             'author_name',
             'availability_of_data',
             'caption',
             'conflict_int',
             'contact_info',
             'copyright',
             'core_text',
             'date',
             'doi',
             'figure',
             'funding_info',
             'index',
             'keywords',
             'list',
             'math_formula',
             'note',
             'publisher_note',
             'reference',
             'section_heading',
             'subheading',
             'table',
             'title',
             'nomenclature',
             'code',
             'publisher',
             'journal',
             'corresponding_author',
             'editor',
             'ethics',
             'consent_publication',
             'MSC',
             'article_history',
             'acknowledgment',
             'background'}

TAG_MAPPING = {'abbreviation': 'background',
                          'acknowledgment': 'background',
                          'additional_file': 'background',
                          'affiliation': 'background',
                          'article_history': 'background',
                          'author_contribution': 'background',
                          'availability_of_data': 'background',
                          'code': 'background',
                          'conflict_int': 'background',
                          'consent_publication': 'background',
                          'corresponding_author': 'background',
                          'date': 'background',
                          'ethics': 'background',
                          'index': 'background',
                          'journal': 'background',
                          'nomenclature': 'background',
                          'publisher_note': 'background',
                          'urls_to_supplementary': 'background',
                          'MSC': 'background',
                          'msc': 'background',
                          'highlights': 'background',
                          'subheading': 'section_heading'}

SAVED_PKL_FILE = 'saved_dad_paths.pkl'


class DADRaw:
    """
    Process the DAD dataset under data_dir, process it to
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
        self.img_dir = os.path.join(self.data_dir, 'documents')
        self.seed = seed
        assert os.path.exists(self.img_dir)
        
        self.train_paths, self.valid_paths, self.test_paths, self.num_classes = self.build_and_split()
        
    
    def build_and_split(self, border_size=6, force=False):
        anno_dir = os.path.join(self.data_dir, "annotations")
        mask_dir = os.path.join(self.data_dir, "masks")
        if os.path.exists(SAVED_PKL_FILE) and not force:
            all_used_tags, class_mapping = pickle.load(open(SAVED_PKL_FILE, 'rb'))
        else:
            all_used_tags = {}
            for anno_json in os.listdir(anno_dir):
                anno_path = os.path.join(anno_dir, anno_json)
                _, class_mapping, used_tags, = write_dad_masks(anno_path, 
                                                               mask_dir, 
                                                               tag_names=TAG_NAMES,
                                                               tag_mapping=TAG_MAPPING,
                                                               buffer_size=border_size,
                                                               force=force)
                
                all_used_tags.update(used_tags)
            pickle.dump((all_used_tags, class_mapping), open(SAVED_PKL_FILE, 'wb'))
        
        #%% - get data stats
        usage_numbers = {}
        total_num_tags = 0
        for paper, tags in all_used_tags.items():
            for tag in tags:
                if tag not in usage_numbers:
                    usage_numbers[tag] = 0
                usage_numbers[tag] += 1
                total_num_tags += 1
        
        #%% - attempt to split into stratified samples (this is gross, but works!)
        filtered_used_tags = {}
        for path, used_tags in all_used_tags.items():
            if len(used_tags) != 0:
                filtered_used_tags[path] = used_tags
        
        train_paths, test_paths = stratify_train_test_split(filtered_used_tags, 0.10, seed=self.seed, debug=False)
        
        #%% - further split the test set into test and validation sets
        test_used_tags = {}
        for path, used_tags in filtered_used_tags.items():
            if path in test_paths:
                test_used_tags[path] = used_tags
        
        test_paths, valid_paths = stratify_train_test_split(test_used_tags, 0.50, seed=self.seed, debug=False)
        
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
        
        label_dir = self.img_dir.replace("documents", "masks")
        if not os.path.exists(label_dir):
            raise ValueError("Ensure masks are built prior to running")
        
        label_path = p.replace("documents", "masks").replace("jpg", "png")
 
        edge_dir = self.img_dir.replace("documents", "edges")
        if not os.path.exists(edge_dir):
            os.makedirs(edge_dir)
        edge_label_path = p.replace("documents", "edges").replace("jpg", "png")
        if not os.path.exists(os.path.dirname(edge_label_path)):
            try:
                os.makedirs(os.path.dirname(edge_label_path))
            except FileExistsError as e:
                print("oops")

        return img_path, label_path, edge_label_path

    def dataset_paths(self, split):
        img_paths = self.get_img_paths(split)
        dataset = [self._convert_item_path_to_training_paths(p) for p in img_paths]
        return dataset

    ####################################################################
    # create edge labels in the label directory of the DAD data
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
        self.train_paths, self.valid_paths, self.test_paths, self.num_classes = self.build_and_split(border_size=0, force=True)

        image_paths = self.get_img_paths("train")
        image_paths += self.get_img_paths("valid")
        image_paths += self.get_img_paths("test")
        num_ps = len(image_paths)
        print('creating edge maps')
        for i, _ in enumerate(p.imap_unordered(self._create_edge_map_from_path, image_paths), 1):
            sys.stderr.write('\rdone {0:%}'.format(i / num_ps))

        # Re-write original masks
        self.train_paths, self.valid_paths, self.test_paths, self.num_classes = self.build_and_split(force=True)
