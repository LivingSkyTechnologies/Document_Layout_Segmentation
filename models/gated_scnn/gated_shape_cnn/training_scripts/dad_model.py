import os
import tensorflow as tf
import numpy as np

#gpus = tf.config.experimental.list_physical_devices('GPU')
#if gpus:
    #try:
        #tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12000)])
        #tf.config.experimental.set_memory_growth(gpus[0], True)
    #except RuntimeError as e:
        #print(e)


import sys
sys.path.append("/home/uosr/segmentation-tensorflow/Gated-SCNN")
sys.path.append("/home/uosr/segmentation-tensorflow")

import gated_shape_cnn.datasets.dad as dad
import gated_shape_cnn.datasets.dad.dataset
from gated_shape_cnn.datasets.dad.dataset import DAD

from gated_shape_cnn.training.train_and_evaluate import train_model

import gated_shape_cnn
from gated_shape_cnn.training.loss import loss as gscnn_loss

from gated_shape_cnn.training import utils
from gated_shape_cnn.model import GSCNN

import pickle
pkl_file = '/home/uosr/segmentation-tensorflow/base_images_224x224_all_reduced.pkl'

if os.path.exists(pkl_file):
    all_used_tags, class_mapping = pickle.load(open(pkl_file, 'rb'))

#os.environ['CUDA_VISIBLE_DEVICES'] = "0"

# build the datasets
batch_size = 4
network_input_h = network_input_w = 512
max_crop_downsample = 0.0
colour_aug_factor = 0.0

# build the dataset loader
data_dir_with_edge_maps = '/home/uosr/dad/'
dad_dataset_loader = DAD(
    batch_size,
    network_input_h,
    network_input_w,
    max_crop_downsample,
    colour_aug_factor,
    debug=False,
    data_dir=data_dir_with_edge_maps)

train = dad_dataset_loader.build_training_dataset()
valid = dad_dataset_loader.build_validation_dataset()
test = dad_dataset_loader.build_test_dataset()

for img, label, edge, box in train.take(1):
    train_img = img
    train_label = label
    train_edge = edge
    train_box = box

for img, label, edge, box in valid.take(1):
    valid_img = img
    valid_label = label
    valid_edge = edge
    valid_box = box

IGNORE_LABEL = 255

class SegmentationAccuracy(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metric_fn = tf.keras.metrics.SparseCategoricalAccuracy()
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.metric_fn.update_state(y_true, y_pred)
    def result(self):
        return self.metric_fn.result()
    def get_config(self):
        base_config = super().get_config()
        return {**base_config}



# optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-6)

def loss(model, input_image, gt_mask, gt_edges, gt_boxes, training, loss_weights):
    out = model(input_image, training=training)
    prediction, shape_head = out[..., :-1], out[..., -1:]

    loss_val = tf.reduce_sum(gscnn_loss(gt_mask, gt_boxes, prediction, shape_head, gt_edges, loss_weights))
    reg_loss = tf.reduce_sum(model.losses)
    return tf.math.add(loss_val, reg_loss)

def grad(model, input_image, gt_mask, gt_edges, gt_boxes, loss_weights):
    with tf.GradientTape() as tape:
        loss_val = loss(model, input_image, gt_mask, gt_edges, gt_boxes, True, loss_weights)
    return loss_val, tape.gradient(loss_val, model.trainable_variables)

num_epochs = 4
best_val_loss = 100000000.0
num_bad_iters = 0

model = GSCNN(n_classes=dad.N_CLASSES)
model_name = "gscnn_model"
loss_weights = (1., 1., 1., 1., 1.)

for epoch in range(num_epochs):
    break
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.Accuracy()
    
    epoch_val_loss_avg = tf.keras.metrics.Mean()
    epoch_val_accuracy = tf.keras.metrics.Accuracy()
    
    step = 0
    for input_image, gt_mask, gt_edges, gt_boxes in train:
        tf.keras.backend.clear_session()
        loss_value, grads = grad(model, input_image, gt_mask, gt_edges, gt_boxes, loss_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        keep_mask = tf.reduce_any(gt_mask == 1., axis=-1)
        flat_label = tf.argmax(gt_mask, axis=-1)
        flat_label = tf.where(keep_mask, flat_label, 0, )
        flat_pred = tf.argmax(model(input_image, training=True)[...,:-1], axis=-1)
        
        flat_label_masked = flat_label[keep_mask]
        flat_pred_masked = flat_pred[keep_mask]

        epoch_loss_avg.update_state(loss_value)
        epoch_accuracy.update_state(flat_label_masked, flat_pred_masked)
        step += 1
        if step % 100 == 0:
            print("Step {}: Loss: {:.3f}, Accuracy: {:.3%}".format(step, epoch_loss_avg.result(), epoch_accuracy.result()))
   
    for input_image, gt_mask, gt_edges, gt_boxes in valid:
        tf.keras.backend.clear_session()
        loss_value = loss(model, input_image, gt_mask, gt_edges, gt_boxes, False, loss_weights)
        
        keep_mask = tf.reduce_any(gt_mask == 1., axis=-1)
        flat_label = tf.argmax(gt_mask, axis=-1)
        flat_label = tf.where(keep_mask, flat_label, 0, )
        flat_pred = tf.argmax(model(input_image, training=True)[...,:-1], axis=-1)
        
        flat_label_masked = flat_label[keep_mask]
        flat_pred_masked = flat_pred[keep_mask]

        epoch_val_loss_avg.update_state(loss_value)
        epoch_val_accuracy.update_state(flat_label_masked, flat_pred_masked)
    
    val_loss = epoch_val_loss_avg.result()
    if val_loss < best_val_loss:
        print("Val Loss decreased from {:.4f} to {:.4f}".format(best_val_loss, val_loss))
        best_val_loss = val_loss
        num_bad_iters = 0
        model.save(model_name)
    else:
        print("Val Loss did not decrease from {:.4f}".format(best_val_loss))
        num_bad_iters += 1
    
    print("Epoch: {:02d} Loss: {:.3f}, Accuracy: {:.3%}, Val Loss: {:.3f}, Val Accuracy: {:.3%}\n".format(epoch, 
                                                                                                        epoch_loss_avg.result(),
                                                                                                        epoch_accuracy.result(),
                                                                                                        epoch_val_loss_avg.result(),
                                                                                                        epoch_val_accuracy.result()))       

#%%
#print("Unfreezing the downsampler...")
#for layer in model.layers:
#    layer.trainable = True

num_epochs = 100
lr_decreased = False
for epoch in range(num_epochs):
    if num_bad_iters >= 5 and lr_decreased:
        print("Val Loss is not improving, exiting...")
        break
    elif num_bad_iters == 5:
        print("Lowering lr, restarting from best model")
        lr_decreased = True
        num_bad_iters = 0
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-7)
        model = tf.keras.models.load_model(model_name)

    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.Accuracy()
    
    epoch_val_loss_avg = tf.keras.metrics.Mean()
    epoch_val_accuracy = tf.keras.metrics.Accuracy()
    
    step = 0
    for input_image, gt_mask, gt_edges, gt_boxes in train:
        tf.keras.backend.clear_session()
        loss_value, grads = grad(model, input_image, gt_mask, gt_edges, gt_boxes, loss_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        
        keep_mask = tf.reduce_any(gt_mask == 1., axis=-1)
        flat_label = tf.argmax(gt_mask, axis=-1)
        flat_label = tf.where(keep_mask, flat_label, 0, )
        flat_pred = tf.argmax(model(input_image, training=True)[...,:-1], axis=-1)
        
        flat_label_masked = flat_label[keep_mask]
        flat_pred_masked = flat_pred[keep_mask]

        epoch_loss_avg.update_state(loss_value)
        epoch_accuracy.update_state(flat_label_masked, flat_pred_masked)

        step += 1
        if step % 100 == 0:
            print("Step {}: Loss: {:.3f}, Accuracy: {:.3%}".format(step, epoch_loss_avg.result(), epoch_accuracy.result()))
    
    for input_image, gt_mask, gt_edges, gt_boxes in valid:
        tf.keras.backend.clear_session()
        loss_value = loss(model, input_image, gt_mask, gt_edges, gt_boxes, False, loss_weights)
        
        keep_mask = tf.reduce_any(gt_mask == 1., axis=-1)
        flat_label = tf.argmax(gt_mask, axis=-1)
        flat_label = tf.where(keep_mask, flat_label, 0, )
        flat_pred = tf.argmax(model(input_image, training=True)[...,:-1], axis=-1)
        
        flat_label_masked = flat_label[keep_mask]
        flat_pred_masked = flat_pred[keep_mask]

        epoch_val_loss_avg.update_state(loss_value)
        epoch_val_accuracy.update_state(flat_label_masked, flat_pred_masked)
    
    val_loss = epoch_val_loss_avg.result()
    if val_loss < best_val_loss:
        print("Val Loss decreased from {:.4f} to {:.4f}".format(best_val_loss, val_loss))
        best_val_loss = val_loss
        num_bad_iters = 0
        model.save(model_name)
    else:
        print("Val Loss did not decrease from {:.4f}".format(best_val_loss))
        num_bad_iters += 1
    
    print("Epoch: {:02d} Loss: {:.3f}, Accuracy: {:.3%}, Val Loss: {:.3f}, Val Accuracy: {:.3%}\n".format(epoch, 
                                                                                                        epoch_loss_avg.result(),
                                                                                                        epoch_accuracy.result(),
                                                                                                        epoch_val_loss_avg.result(),
                                                                                                        epoch_val_accuracy.result()))       


#%%
from skimage import measure
import statistics

def cca(pred_mask):
    new_mask = pred_mask.copy()
    lbl = measure.label(pred_mask)
    
    regions = measure.regionprops(lbl)
    for region in regions:
        if not region:
            continue
        if region.area <= 20:
            continue
        last_region = region
        minr, minc, _, maxr, maxc, _ = region.bbox
        p1 = (minc, minr)
        p2 = (maxc, maxr)
        
        object_region = pred_mask[minr:maxr, minc:maxc]
        object_region = object_region[object_region != 0]
        try:
            region_label = statistics.mode(object_region.flatten())
        except:
            unique, counts = np.unique(object_region, return_counts=True)
            region_label = unique[np.argmax(counts)]
            print(unique, counts)
        if region_label != 0:
            new_mask[minr:maxr, minc:maxc] = [region_label]
    return new_mask

#%%  
from MetricsUtils import evaluate_segmentation
from scipy.ndimage import find_objects


model = tf.keras.models.load_model(model_name)

f1_avg = 0
count = 0
inference_times = []
avg_train_class_iou = np.zeros((len(class_mapping),))
avg_train_class_accuracies = np.zeros((len(class_mapping),))
avg_train_class_count = np.zeros((len(class_mapping),))
for input_images, golden_masks, edges, boxes in test:
    break
    for input_image, golden_mask in zip(input_images, golden_masks):
        one_img_batch = input_image[tf.newaxis, ...]
        import time
        start = time.time()
        y_pred = model(one_img_batch)
        end = time.time()
        inference_times.append(end-start)
        tf.keras.backend.clear_session()
        
        pred_mask = create_mask(y_pred[0]).numpy()
        pred_mask = cca(pred_mask)
        #if count < 10:
        #    # The don't-care regions are distracting
        #    golden_mask_new = tf.where(golden_mask == IGNORE_LABEL, 0.0, golden_mask)
        #    display_sample([input_image, golden_mask_new,
        #                    pred_mask])
        global_accuracy, class_accuracies, prec, rec, f1, iou = evaluate_segmentation(pred_mask.astype(int), golden_mask.numpy().astype(int), len(class_mapping))
        f1_avg += f1
        count += 1
        for i, val in class_accuracies.items():
            avg_train_class_accuracies[i] += val
            avg_train_class_iou[i] += iou[i]
            avg_train_class_count[i] += 1

#f1_avg = f1_avg/float(count)
#print("Global Test F1 Score was {}".format(f1_avg))
#print("Average inference time was {} for {} images".format(sum(inference_times)/len(inference_times), len(test_paths)+len(valid_paths)))


f1_avg = 0
test_count = 0
avg_test_class_iou = np.zeros((len(class_mapping),))
avg_test_class_accuracies = np.zeros((len(class_mapping),))
avg_test_class_count = np.zeros((len(class_mapping),))
for input_images, golden_masks, edges, boxes in test:
    for input_image, golden_mask in zip(input_images, golden_masks):
        one_img_batch = input_image[tf.newaxis, ...]
        y_pred = model(one_img_batch)[...,:-1]
        tf.keras.backend.clear_session()
        
        golden_mask = golden_mask[tf.newaxis,...]
        keep_mask = tf.reduce_any(golden_mask == 1., axis=-1)
        flat_label = tf.argmax(golden_mask, axis=-1)
        flat_label = tf.where(keep_mask, flat_label, tf.cast(gated_shape_cnn.N_COLOURS - 1, tf.int64), )
        flat_pred = tf.argmax(y_pred, axis=-1)
        
        flat_label_masked = flat_label[keep_mask]
        flat_pred_masked = flat_pred[keep_mask]

        #if count < 10:
        #    # The don't-care regions are distracting
        #    golden_mask_new = tf.where(golden_mask == IGNORE_LABEL, 0.0, golden_mask)
        #    display_sample([input_image, golden_mask_new,
        #                    pred_mask])
        global_accuracy, class_accuracies, prec, rec, f1, iou = evaluate_segmentation(flat_pred_masked.numpy().astype(int), flat_label_masked.numpy().astype(int), len(class_mapping))
        f1_avg += f1
        for i, val in class_accuracies.items():
            avg_test_class_accuracies[i] += val
            avg_test_class_iou[i] += iou[i]
            avg_test_class_count[i] += 1
        test_count += 1

f1_avg = f1_avg/float(test_count)
print("Just test F1 Score was {}".format(f1_avg))

f1_avg = 0
valid_count = 0
avg_valid_class_accuracies = np.zeros((len(class_mapping),))
avg_valid_class_iou = np.zeros((len(class_mapping),))
avg_valid_class_count = np.zeros((len(class_mapping),))
for input_images, golden_masks, edges, boxes in valid:
    for input_image, golden_mask in zip(input_images, golden_masks):
        one_img_batch = input_image[tf.newaxis, ...]
        
        y_pred = model(one_img_batch)[...,:-1]
        tf.keras.backend.clear_session()
       
        golden_mask = golden_mask[tf.newaxis, ...]
        keep_mask = tf.reduce_any(golden_mask == 1., axis=-1)
        flat_label = tf.argmax(golden_mask, axis=-1)
        flat_label = tf.where(keep_mask, flat_label, tf.cast(gated_shape_cnn.N_COLOURS - 1, tf.int64), )
        flat_pred = tf.argmax(y_pred, axis=-1)
        
        flat_label_masked = flat_label[keep_mask]
        flat_pred_masked = flat_pred[keep_mask]

        #if count < 10:
        #    # The don't-care regions are distracting
        #    golden_mask_new = tf.where(golden_mask == IGNORE_LABEL, 0.0, golden_mask)
        #    display_sample([input_image, golden_mask_new,
        #                    pred_mask])
        global_accuracy, class_accuracies, prec, rec, f1, iou = evaluate_segmentation(flat_pred_masked.numpy().astype(int), flat_label_masked.numpy().astype(int), len(class_mapping))
        f1_avg += f1

        for i, val in class_accuracies.items():
            avg_valid_class_accuracies[i] += val
            avg_valid_class_iou[i] += iou[i]
            avg_valid_class_count[i] += 1
        valid_count += 1

f1_avg = f1_avg/float(valid_count)
print("Just validation F1 Score was {}".format(f1_avg))

#%%
print("Class Accuracies:")
for i, name in class_mapping.items():
    #if name in all_reduced:
    #    continue
    test_acc_score = avg_test_class_accuracies[i]
    test_class_count = avg_test_class_count[i]
    valid_acc_score = avg_valid_class_accuracies[i]
    valid_class_count = avg_valid_class_count[i]
    #train_acc_score = avg_train_class_accuracies[i]
    #train_class_count = avg_train_class_count[i]
    print("|%s|%f%%|%f%%|" % (name,
                              test_acc_score/float(test_class_count)*100.0, 
                              valid_acc_score/float(valid_class_count)*100.0))

print("\n")
print("Class mIOU:")
for i, name in class_mapping.items():
    test_iou_score = avg_test_class_iou[i]
    test_class_count = avg_test_class_count[i]
    valid_iou_score = avg_valid_class_iou[i]
    valid_class_count = avg_valid_class_count[i]
    #train_iou_score = avg_train_class_iou[i]
    #train_class_count = avg_train_class_count[i]
    print("|%s|%f%%|%f%%|" % (name,
                              test_iou_score/float(test_class_count)*100.0,
                              valid_iou_score/float(valid_class_count)*100.0))


