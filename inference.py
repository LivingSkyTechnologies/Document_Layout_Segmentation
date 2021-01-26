import argparse
import cv2
import json
import matplotlib.pyplot as plt
import os
import pickle
import statistics
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from PIL import Image
from skimage import measure
from train import create_mask

ALLOW_OVERLAP = {"list", "math_formula", "core_text", "caption"}  # Objects that are usually small and close together
REMOVE_THESE = {"appendice", "author_bio", "contact_info", "copyright", "doi", "editor", "list", "funding_info", "keywords", "publisher", "reference"}

def filter_boxes(boxes, class_mapping, tol=0.5):
    keep_box = {x: True for x in range(len(boxes))}
    for (class_num_a, y1_a, x1_a, y2_a, x2_a) in boxes:
        i = -1
        for (class_num_b, y1_b, x1_b, y2_b, x2_b) in boxes:
            i += 1
            if x1_a == x1_b and y1_a == y1_b and x2_a == x2_b and y2_a == y2_b:
                continue  # Skip comparison with itself
            xA = tf.math.maximum(x1_a, x1_b)
            yA = tf.math.maximum(y1_a, y1_b)
            xB = tf.math.minimum(x2_a, x2_b)
            yB = tf.math.minimum(y2_a, y2_b)
        
            # compute the area of intersection rectangle
            area_a = (x2_a-x1_a)*(y2_a-y1_a)
            area_b = (x2_b-x1_b)*(y2_b-y1_b)
            interArea = tf.math.abs(tf.math.maximum(xB - xA, 0) * tf.math.maximum(yB - yA, 0))
            
            # Remove classes that overlap too much
            if interArea > area_b*tol and area_b < area_a:
                if class_mapping[class_num_b] not in ALLOW_OVERLAP or class_num_a == class_num_b:
                    keep_box[i] = False

            # Remove poor performing classes
            if class_mapping[class_num_b] in REMOVE_THESE:
                keep_box[i] = False

    keep_idxs = [x for x in range(len(boxes)) if keep_box[x]]
    new_boxes = np.take(boxes, keep_idxs, axis=0)
    return new_boxes

def scale_boxes(boxes, actual_size, inference_size):
    boxes = np.array(boxes)

    x_scale = actual_size[0] / float(inference_size)
    y_scale = actual_size[1] / float(inference_size)

    boxes[:, 1] = boxes[:, 1] * y_scale
    boxes[:, 2] = boxes[:, 2] * x_scale
    boxes[:, 3] = boxes[:, 3] * y_scale
    boxes[:, 4] = boxes[:, 4] * x_scale

    return boxes

def prepare_single_image(input_img_path, img_size):
    # Load image data
    assert input_img_path.endswith(".jpg"), "Images must be jpg format"
    input_img = tf.io.read_file(input_img_path)
    input_img = tf.image.decode_jpeg(input_img, channels=3)

    # Resize and normalize
    input_img = tf.image.resize(input_img, (img_size, img_size))
    input_img = tf.math.divide(tf.cast(input_img, tf.float32), 255.0) 

    return input_img

def cca(pred_mask, class_mapping, min_area=200, return_boxes=False):
    new_mask = np.zeros(pred_mask.shape)
    lbl = measure.label(pred_mask)

    regions = measure.regionprops(lbl)

    boxes = []
    for region in regions:
        if not region:
            continue
        if region.area <= min_area:
            continue
        last_region = region
        minr, minc, _, maxr, maxc, _ = region.bbox
        
        p1 = (minc, minr)
        p2 = (maxc, maxr)
        
        object_region = pred_mask[minr:maxr, minc:maxc]
        object_region = object_region[object_region != 0]

        # Sometimes a region has equal amounts of two objects, so mode() fails
        # When this happens, we could implement rules about which object is more likely
        # But for now, just take the first one
        try:
            region_label = statistics.mode(object_region.flatten())
        except:
            unique, counts = np.unique(object_region, return_counts=True)
            region_label = unique[np.argmax(counts)]
            #print(unique, counts)
        if return_boxes:
            boxes.append((region_label, minr, minc, maxr, maxc))
        elif region_label != 0:
            new_mask[minr:maxr, minc:maxc] = [region_label]

    if return_boxes:
        return boxes
    
    return new_mask

def write_labelme_json(mask, class_mapping, out_path):
    boxes = cca(mask, class_mapping, return_boxes=True)
    boxes = scale_boxes(boxes, Image.open(out_path.replace("json", "jpg")).size, mask.shape[0])
    boxes = filter_boxes(boxes, class_mapping)
    labelme_template = {"version": "4.2.10",
                        "flags": {},
                        "shapes": [],
                        "imagePath": os.path.basename(out_path.replace("json", "jpg")),
                        "imageData": None,
                        "imageHeight": int(mask.shape[0]), 
                        "imageWidth": int(mask.shape[1])}
    for group_id, miny, minx, maxy, maxx in list(boxes):
        labelme_template['shapes'].append({"label": class_mapping[group_id],
                                           "points": [
                                                [int(minx), int(miny)],
                                                [int(maxx), int(maxy)]
                                           ],
                                           "group_id": int(group_id),
                                           "shape_type": "rectangle",
                                           "flags": {}})

    with open(out_path, 'w') as f:
        json.dump(labelme_template, f, indent=4)

def display_sample(display_list):
    plt.figure(figsize=(18, 18))
    title = ['Input Image', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def post_process_predictions(inputs_masks_and_paths, class_mapping, apply_cca=False, visualize=False, write_json=False):
    for img, mask, name in inputs_masks_and_paths:
        if apply_cca:
            mask = cca(mask, class_mapping)

        if visualize:
            display_sample([img, mask])

        if write_json:
            write_labelme_json(mask, class_mapping, name.replace("jpg", "json"))

        cv2.imwrite(name.replace("jpg", "_mask.png"), mask)

def generic_seg_inference(model, input_imgs, img_paths, class_mapping, is_gscnn=False, apply_cca=False, visualize=False, write_json=False):
    inputs_masks_and_paths = []
    for img, img_path in zip(input_imgs, img_paths):
        y_pred = model(img[tf.newaxis,...], training=False)
        if is_gscnn:
            y_pred = y_pred[...,:-1]  # gscnn has seg and shape head

        # TODO: the numpy call slows things down, can we do everything with tf operations?
        mask = create_mask(y_pred)[0].numpy()
        inputs_masks_and_paths.append((img, mask, img_path))

    post_process_predictions(inputs_masks_and_paths, class_mapping, apply_cca=apply_cca, visualize=visualize, write_json=write_json)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input-image", help="Single image to segment.")
    input_group.add_argument("--input-folder", help="Folder of images to segment.")

    parser.add_argument("--saved-model", help="Directory or h5 file with a saved model.")
    parser.add_argument("--model", help='One of "unet", "fast_fcn", "deeplabv3plus", or "gated_scnn".')
    parser.add_argument("--img-size", type=int, help="Size of images. Should match the size trained on.")
    parser.add_argument("--saved-pkl", help='The saved PKL file from the training step. It is used for the class mapping.')
    parser.add_argument("--apply-cca", default=False, action='store_true', help="Post process with conncected component analysis. Makes segmentations uniform, but might miss objects.")
    parser.add_argument("--visualize", default=False, action='store_true', help="If set, will open a matplotlib plot to see the segmentation visually.")
    parser.add_argument("--write-annotation", default=False, action='store_true', help="If set, will also write a json file with annotations in labelme format.")
    
    args = parser.parse_args()
    
    # Load model
    model = tf.keras.models.load_model(args.saved_model, compile=False, custom_objects={'tf': tf})
    
    print(model.summary())
    raise Exception("NO")

    # Figure out our mode (single or multi)
    is_single = True if args.input_image else False
    
    # Prepare inputs
    if is_single:
        input_img = prepare_single_image(args.input_image, args.img_size)
        input_imgs = [input_img]
        img_paths = [args.input_image]
    else:
        img_paths = []
        input_imgs = []
        for img in os.listdir(args.input_folder):
            img_path = os.path.join(args.input_folder, img)
            if img_path.endswith("jpg"):
                input_image = prepare_single_image(img_path, args.img_size)

                img_paths.append(img_path)
                input_imgs.append(input_image)
    
    # Perform prediction with specified options
    _, class_mapping = pickle.load(open(args.saved_pkl, 'rb'))
    print(class_mapping)

    generic_seg_inference(model, input_imgs, img_paths, class_mapping, is_gscnn="gated" in args.model, 
                                                                       apply_cca=args.apply_cca, 
                                                                       visualize=args.visualize, 
                                                                       write_json=args.write_annotation)
