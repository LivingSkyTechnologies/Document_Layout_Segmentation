import cv2
import glob
import io
import json
import math
import numpy as np
import os
import tensorflow as tf

from PIL import Image

# TODO: This should be temporary! Fix annotation files to be consistent
TRANSLATE = {"publication": "copyright",
             "copyrights": "copyright",
             "data": "list"}

def write_dad_masks(vott_filepath, mask_out_path, tag_names=None, 
                    tag_mapping=None, buffer_size=0, force=False):
    """
    Takes in an annotation JSON from VOTT, and creates masks files 
    described by the bounding boxes and tags in the JSON. Returns a list of 
    tags found and a mapping of class->tag name, and a set of tags actually used.
    
    tag_names, class_to_tag, file_to_tags_used = write_masks_from_annotation("train/annotations/doc/doc.json", "train/masks")

    Parameters
    ----------
    vott_filepath : string
        Path to a VOTT annotation JSON
    mask_out_path : string
        Folder to place masks into
    tag_names : list
        Override the tag_names list so that the values in the json are ignored
    buffer_size : int
        The size of a buffer "don't care" region between a mask and the background. 
        The buffer is subtracted from the mask, making the mask slightly smaller
    force : bool
        If true, will write files to disk if they already exist

    Returns
    -------
    tag_names, class_dict, file_to_tags_used : list, dict, dict
        A list of tags found, a dict mapping of class->tag name, and a dict mapping of img->tags actually used
    """
    vott_json = open(vott_filepath, 'r')
    dataset = json.load(vott_json)
    vott_json.close()
    
    if tag_names == None:
        tag_names = set([x["name"].lower()for x in sorted(dataset["tags"], key=lambda x: x["name"])])
    
    class_mapping = {}
    i = 1
    for key in sorted(tag_names):
        if key == 'background':
            continue
        if tag_mapping and key in tag_mapping:
            continue
        key = key.lower()
        class_mapping[key] = (i, i ,i)
        i += 1
    
    # Special case the background
    class_mapping['background'] = (0, 0, 0)
    tag_names.add('background')
    
    used_tags = {}
    for ano_id, ano_data in dataset['assets'].items():
        img_index = ano_data['asset']['path'].split('-')[1].split('.')[0].lstrip('0')
        img_name = os.path.basename(vott_filepath).replace("json", "jpg")
        img_dir = os.path.join(os.path.dirname(vott_filepath.replace('annotations', 'documents')),
                               img_name.split('.')[0])
        img_name = img_name.replace('.jpg', '-{}.jpg'.format(img_index))
        img_path = os.path.join(img_dir, img_name)
        
        # Special case. Sometimes img's have a leading zero in vol num, sometimes not
        if not os.path.exists(img_path):
            img_name = img_name.replace("-{}.jpg".format(img_index), "-0{}.jpg".format(img_index))
            img_path = os.path.join(img_dir, img_name)
        
        if not os.path.exists(img_path):
            print("Unable to find image {}".format(img_path))
            continue
        
        box_path = img_path.replace(".jpg", ".txt")
        
        img = cv2.imread(img_path)
        img = np.zeros(img.shape)
        
        area_and_boxes = []
        used_tags[img_path] = set()
        for region in ano_data["regions"]:
            key = region["tags"][-1].lower()
            if key not in class_mapping:
                continue
            
            if key in TRANSLATE:
                key = TRANSLATE[key]
            
            if tag_mapping and key in tag_mapping:
                key = tag_mapping[key]
                
            rgb = class_mapping[key]
            x = int(region["boundingBox"]["left"])
            y = int(region["boundingBox"]["top"])
            w = int(region["boundingBox"]["width"])
            h = int(region["boundingBox"]["height"])
            area = w*h
            area_and_boxes.append((area, x, y, w, h, rgb))
            used_tags[img_path].add(rgb[0])
        
        # sort by area, so smaller boxes are on the top layer, also write out box file
        if os.path.exists(box_path):
            os.remove(box_path)
        
        area_and_boxes.sort(reverse=True)
        box_f = open(box_path, 'w+')
        for area, x, y, w, h, rgb in area_and_boxes:
            if buffer_size > 0:
                cv2.rectangle(img, (x-buffer_size,y-buffer_size), (x+w+buffer_size, y+h+buffer_size), (255, 255, 255), -1)
            cv2.rectangle(img, (x+buffer_size, y+buffer_size), (x+w-buffer_size, y+h-buffer_size), rgb, -1)
            box_f.write("{},{},{},{},{}\n".format(rgb[0], x, y, w, h))
        box_f.close()
                
        if not os.path.exists(mask_out_path):
            os.mkdir(mask_out_path)
        
        mask_dir = os.path.join(mask_out_path, os.path.basename(img_dir))
        if not os.path.exists(mask_dir):
            os.mkdir(mask_dir)
    
        outfile = os.path.join(mask_dir, os.path.basename(img_path).replace('jpg', 'png'))
        
        if not os.path.exists(outfile) or force:
            cv2.imwrite(outfile, img)
    
    actual_class_mapping = {}
    for key in sorted(class_mapping):
        key = key.lower()
        actual_class_mapping[class_mapping[key][0]] = key
    
    
    return tag_names, actual_class_mapping, used_tags

def write_publaynet_masks(json_path, is_val_set=False, draw_border=True):
    with open(json_path, 'r') as fp:
        trainsamples = json.load(fp)

        # organise the dictionary into something more usable
    images = {}
    for image in trainsamples['images']:
        images[image['id']] = {'file_name': image['file_name'],
                               'width': image['width'],
                               'height': image['height'],
                               'annotations': []}
    for ann in trainsamples['annotations']:
        images[ann['image_id']]['annotations'].append(ann)

    num_train_images = len(images.keys())
    used_tags = {}
    for counter, img_id in enumerate(list(images)):
        if counter%1000 == 0:
            print('Creating segmentation masks: image {} of {}'.format(counter, num_train_images))
        
        if is_val_set:
            filename = os.path.join(os.path.join(os.path.dirname(json_path), 'val'), os.path.basename(images[img_id]['file_name']))
        else:
            filename = os.path.join(os.path.join(os.path.dirname(json_path), 'train'), os.path.basename(images[img_id]['file_name']))    

        if os.path.exists(filename):
            seg_filename = filename.replace('jpg', 'png')
        else:
            continue

        used_tags[filename] = set()
        height = int(images[img_id]['height'])
        width = int(images[img_id]['width'])

        seg_mask = np.zeros((width, height))
        corrupted = False
        area_and_boxes = []
        for ann in images[img_id]['annotations']:
            current_bbox = np.asarray(ann['bbox'], dtype = np.int32)
            x1, x2 = current_bbox[0], current_bbox[0] + current_bbox[2]
            y1, y2 = current_bbox[1], current_bbox[1] + current_bbox[3]
            #the object's pixels are updated to its class_id
            seg_mask[x1:x2, y1:y2] = int(ann['category_id'])
            used_tags[filename].add(int(ann['category_id']))
            #the object's border pixels are updated to 255 (unknown) to create contrast and aid with learning
            if draw_border:
                try:
                    seg_mask[x1, y1:y2] = 255
                    seg_mask[x2, y1:y2] = 255
                    seg_mask[x1:x2, y1] = 255
                    seg_mask[x1:x2, y2] = 255
                except Exception as e:
                    print("Invalid box sizes for img {}, skipping border".format(images[img_id]['file_name']))
            area_and_boxes.append((int(ann['category_id']), x1, y1, current_bbox[2], current_bbox[3]))  # we transpose these

        with open(seg_filename.replace("png", "txt"), 'w') as box_f:
            for class_id, x, y, w, h in area_and_boxes:
                box_f.write("{},{},{},{},{}\n".format(class_id, x, y, w, h)) 

        seg_mask = seg_mask.T
        seg_img = Image.fromarray(seg_mask.astype(dtype=np.uint8))
        with tf.io.gfile.GFile(seg_filename, mode='w') as f:
            seg_img.save(f, 'PNG')

    return used_tags


def apply_mask_localization(original_image, iask_image, mask_class):
    """
    Takes in a path to an image, path to corresponding mask image array, and 
    the mask class number to apply from the mask image, and return the new localized 
    image according to the mask. This assumes the mask image is made up of 
    multiple masks.

    localized_image = apply_mask_localization(orig_img, img_mask, class_name_to_class_number["core_text"])

    Parameters
    ----------
    original_image : string
        The image you want to apply a mask to
    mask_image : string
        The mask image to select a mask from.
    mask_class : int
        The class number to select from the mask

    Returns
    -------
    result : 3D numpy array
        Reurns the resulting image under the chosen mask

    """
    if type(original_image) == str:
        original_image = cv2.imread(original_image)
    
    if type(mask_image) == str:
        mask_image = cv2.imread(mask_image)

    mask_lower_bound = np.array([mask_class, 0, 0])
    mask_upper_bound = np.array([mask_class, mask_class, mask_class])
    
    mask = cv2.inRange(mask_image, mask_lower_bound, mask_upper_bound)
    result = cv2.bitwise_and(original_image, original_image, mask=mask)
    
    return result

