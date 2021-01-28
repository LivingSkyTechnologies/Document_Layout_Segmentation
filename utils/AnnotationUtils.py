import cv2
import glob
import io
import json
import math
import numpy as np
import os
import tensorflow as tf

from PIL import Image


def write_dad_masks(json_path, annotation_dir, document_dir, output_dir, tag_names=None, tag_mapping=None, buffer_size=0, force=False):
    anno_json = open(json_path, 'r')
    data = json.load(anno_json)
    anno_json.close()

    img_name = os.path.basename(data['imagePath'])
    img_path = os.path.join(os.path.dirname(json_path).replace(annotation_dir, document_dir), img_name)
    
    img = cv2.imread(img_path)
    try:
      img = np.zeros(img.shape)
    except Exception as e:
      print(img_path)
      print(json_path)
      raise e

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
    annotations = []
    for annotation in data['shapes']:
        used_tags[img_path] = set()
        class_name = annotation['label'].lower()

        if class_name not in class_mapping:
            continue

        if tag_mapping and class_name in tag_mapping:
            class_name = tag_mapping[class_name]

        class_num = class_mapping[class_name][0]
        x = int(annotation['points'][0][0])
        y = int(annotation['points'][0][1])
        w = int(annotation['points'][1][0]) - x
        h = int(annotation['points'][1][1]) - y
        area = w*h
        used_tags[img_path].add(class_num)

        if w < 0:
          x = int(annotation['points'][1][0])
          w = int(annotation['points'][0][0]) - x
          if w < 0:
              print('WTF {}'.format(json_path))
              continue

        if h < 0:
            y = int(annotation['points'][1][1])
            h = int(annotation['points'][0][1]) - y
            if h < 0:
                print('WTF {}'.format(json_path))
                continue

        annotations.append((area, x, y, w, h, class_num))

    annotations.sort(reverse=True)
    box_path = img_path.replace("jpg", "txt")
    box_f = open(box_path, 'w+')
    for area, x, y, w, h, class_num in annotations:
        if buffer_size > 0:
            cv2.rectangle(img, (x-buffer_size,y-buffer_size), (x+w+buffer_size, y+h+buffer_size), (255, 255, 255), -1)
        cv2.rectangle(img, (x+buffer_size, y+buffer_size), (x+w-buffer_size, y+h-buffer_size), (class_num, class_num, class_num), -1)
        box_f.write("{},{},{},{},{}\n".format(class_num, x, y, w, h))
    box_f.close()

    outfile = img_path.replace('jpg', 'png')
    outfile = outfile.replace(document_dir, output_dir)
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    
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

