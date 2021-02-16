import pickle
import os
import tensorflow as tf
from glob import glob

import utils.DataLoaderUtils as dlu
from utils.AnnotationUtils import write_dad_masks

# Static Dataset Config Options
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
                  'msc': 'background',
                  'MSC': 'background',
                  'highlights': 'background',
                  'subheading': 'section_heading'}

SAVED_PKL_FILE = 'saved_dad_paths.pkl'

BUFFER_SIZE = 500
MASKS_DIR = "masks"
DOCUMENTS_DIR = "documents"
ANNOTATIONS_DIR = "annotations"

def write_masks(dataset_dir, border_buffer=6):
    anno_dir = os.path.join(dataset_dir, ANNOTATIONS_DIR)
    anno_paths = glob(anno_dir + "/*json")
    if os.path.exists(SAVED_PKL_FILE):
        all_used_tags, class_mapping = pickle.load(open(SAVED_PKL_FILE, 'rb'))
    else:
        print("Running full mask generation, this may take a bit.")
        all_used_tags = {}
        for anno_json in anno_paths:
            _, class_mapping, used_tags = write_dad_masks(anno_json,
                                                          ANNOTATIONS_DIR,
                                                          DOCUMENTS_DIR,
                                                          MASKS_DIR, 
                                                          tag_names=TAG_NAMES,
                                                          tag_mapping=TAG_MAPPING,
                                                          buffer_size=border_buffer,
                                                          force=True)
            all_used_tags.update(used_tags)
        pickle.dump((all_used_tags, class_mapping), open(SAVED_PKL_FILE, 'wb'))

    return all_used_tags, class_mapping
 
def build_dad_dataset(dataset_dir, img_size, batch_size, seed, debug=False):
    all_used_tags, class_mapping = write_masks(dataset_dir)
    
    # Filter out any pages that have no classes (this is helpful when messing around with active classes)
    filtered_used_tags = {}
    for path, used_tags in all_used_tags.items():
        if len(used_tags) != 0:
            filtered_used_tags[path] = used_tags
    
    # Split the paths with stratified sampling, to mainting class distribution
    train_paths, test_paths = dlu.stratify_train_test_split(filtered_used_tags, 0.10, seed=seed, debug=debug)
   
    #%% - further split the test set into test and validation sets
    test_used_tags = {}
    for path, used_tags in filtered_used_tags.items():
        if path in test_paths:
            test_used_tags[path] = used_tags

    test_paths, valid_paths = dlu.stratify_train_test_split(test_used_tags, 0.50, seed=seed, debug=debug)
    
    train_dataset = tf.data.Dataset.from_tensor_slices(train_paths)
    train_dataset = train_dataset.map(lambda x: dlu.parse_image(x, 0, MASKS_DIR), num_parallel_calls=tf.data.experimental.AUTOTUNE)

    valid_dataset = tf.data.Dataset.from_tensor_slices(valid_paths)
    valid_dataset = valid_dataset.map(lambda x: dlu.parse_image(x, 0, MASKS_DIR), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
    test_dataset = test_dataset.map(lambda x: dlu.parse_image(x, 0, MASKS_DIR), num_parallel_calls=tf.data.experimental.AUTOTUNE)

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
 
