import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

IGNORE_LABEL = tf.cast(255.0, tf.uint8)


class SegmentationAccuracy(tf.keras.metrics.Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.metric_fn = tf.keras.metrics.SparseCategoricalAccuracy()
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, IGNORE_LABEL))
        y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true[:,:,:,0], IGNORE_LABEL))
        self.metric_fn.update_state(y_true_masked, y_pred_masked)
    def result(self):
        return self.metric_fn.result()
    def get_config(self):
        base_config = super().get_config()
        return {**base_config}
 
@tf.function(experimental_relax_shapes=True)
def build_boxes(labeled):
    """
    Takes in the labeled connected components of the segmentation map, and 
    extracts boxes. Since each object is labeled a unique number, we can extract
    the minimum and maximum coordinates of each label number, to form the box
    as y_min, x_min, y_max, x_max.
    
    The label 0 is ignored by returning a box of [0.0, 0.0, 0.0, 0.0]. This
    "box" gets ignored later on.

    Parameters
    ----------
    labeled : tensor [batch_size, img_size, img_size]
        The output of the connected component analysis, with each unique object
        labeled as a unique number.

    Returns
    -------
    tensor [num_boxes_identified, 4]
        This has a max size of 25. At the start of training, noisy segmentation
        maps lead to thousands of identified "objects", which isn't useful

    """
    img_size = tf.cast(tf.shape(labeled)[-1], tf.float32)

    def batch_unique(x, max_labels=25):
        labels, _ = tf.unique(tf.reshape(x, (-1,)))
        if (tf.greater_equal(tf.shape(labels)[0], max_labels)):
            labels = tf.zeros((max_labels,), dtype=tf.int32)
    
        return tf.pad(labels, [[0, max_labels-tf.shape(labels)[0]]])

    def get_coords(label, single_labeled_img):
        min_y_min_x = tf.math.reduce_min(tf.where(single_labeled_img==label), axis=0)
        max_y_max_x = tf.math.reduce_max(tf.where(single_labeled_img==label), axis=0)
        if tf.equal(tf.gather(min_y_min_x, (0)), 0):
            return tf.cast(tf.reshape(tf.stack([min_y_min_x, min_y_min_x]), (-1,)), tf.float32) # Ignore boxes that start in the top left, it's just padded labels, all zeros means 0 loss later
        else:
            return tf.math.divide(tf.cast(tf.reshape(tf.stack([min_y_min_x, max_y_max_x]), (-1,)), tf.float32), img_size)

    def batch_get_coords(batch_labels, labeled_img):
        return tf.map_fn(lambda x: get_coords(x, labeled_img), batch_labels, parallel_iterations=4, fn_output_signature=tf.float32)
        
    batch_unique_labels = tf.map_fn(batch_unique, labeled, parallel_iterations=4)
    batch_boxes = tf.map_fn(lambda inp: batch_get_coords(inp[0], inp[1]), (batch_unique_labels, labeled), parallel_iterations=4, fn_output_signature=tf.float32)
    return batch_boxes

@tf.function
def compute_iou(boxes1_corners, boxes2_corners):
    """Computes pairwise IOU matrix for given two sets of boxes

    Arguments:
      boxes1: A tensor with shape `(N, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.
        boxes2: A tensor with shape `(M, 4)` representing bounding boxes
        where each box is of the format `[x, y, width, height]`.

    Returns:
      pairwise IOU matrix with shape `(N, M)`, where the value at ith row
        jth column holds the IOU between ith box and jth box from
        boxes1 and boxes2 respectively.
    """
    lu = tf.maximum(boxes1_corners[:, None, :2], boxes2_corners[:, :2])
    rd = tf.minimum(boxes1_corners[:, None, 2:], boxes2_corners[:, 2:])
    intersection = tf.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes1_area = (boxes1_corners[:, 2] - boxes1_corners[:, 0]) * (boxes1_corners[:, 3] - boxes1_corners[:, 1])
    boxes2_area = (boxes2_corners[:, 2] - boxes2_corners[:, 0]) * (boxes2_corners[:, 3] - boxes2_corners[:, 1])
    union_area = tf.maximum(
        boxes1_area[:, None] + boxes2_area - intersection_area, 1e-8
    )
    return tf.clip_by_value(intersection_area / union_area, 0.0, 1.0)

@tf.function
def per_batch_box_loss(pred_boxes, gt_boxes, min_iou=0.05):
    """
    Calculates the giou loss per batch of predicted and ground truth boxes

    Parameters
    ----------
    pred_boxes : tensor [num_predicted_boxes, 4]
        The extracted predicted boxes as y_min, x_min, y_max, x_max
    gt_boxes : tensor [num_gt_boxes, 4]
        The ground truth boxes as y_min, x_min, y_max, x_max
    min_iou : float, optional
        The minimum overlap with ground truth boxes to consider calculating loss.
        The default is 0.4.

    Returns
    -------
    tf.float32
        The giou loss

    """
    iou_matrix = compute_iou(pred_boxes, gt_boxes)
    iou_matrix = tf.where(iou_matrix < min_iou, 0.0, iou_matrix)

    matched_gt_idx = tf.argmax(iou_matrix, axis=1)
    matched_gt_idx = tf.boolean_mask(matched_gt_idx, tf.not_equal(matched_gt_idx, 0))
    matched_gt_boxes = tf.gather(gt_boxes, matched_gt_idx)
    
    matched_pred_idx = tf.reshape(tf.where(matched_gt_idx > 0), (-1,))
    matched_pred_boxes = tf.gather(pred_boxes, matched_pred_idx)

    if tf.equal(tf.shape(matched_pred_boxes)[0], 0):
        return 0.0  # No boxes matched
    else:
        return tf.reduce_mean(tfa.losses.giou_loss(matched_gt_boxes, matched_pred_boxes))

@tf.function
def box_loss(y_true, y_pred):
    """
    Extracts boxes from a segmentation map, calculate giou loss

    Parameters
    ----------
    y_true : tensor [batch_size, num_gt_boxes, 4]
        The ground truth boxes as y_min, x_min, y_max, x_max
    y_pred : tensor [batch_size, img_size, img_size, num_classes]
        The sparse softmax probabilities for the segmentation map

    Returns
    -------
    box_losses : tensor [num_matched_boxes]
        For every box identified and matched, return the giou loss

    """
    y_pred_mask = tf.argmax(y_pred, axis=-1)  # Create segmentation mask from sparse predictions
    y_pred_labeled = tfa.image.connected_components(y_pred_mask)  # Label components
    boxes = build_boxes(y_pred_labeled)  # Returns size [batch_idx, num_labels, 4]

    # Pad y_true with one blank box at the start, helps to ignore "background" boxes
    y_true = tf.pad(y_true, [[0, 0], [1, 0], [0, 0]])
    box_losses = tf.map_fn(lambda inp: per_batch_box_loss(inp[0], inp[1]), (boxes, y_true), parallel_iterations=4, fn_output_signature=tf.float32)  # Calculates for each batch of predicted boxes and true_boxes
    return box_losses

@tf.function
def seg_loss(y_true, y_pred, y_true_boxes, skip_box_loss=False):
    y_true_masked = tf.boolean_mask(y_true, tf.not_equal(y_true, IGNORE_LABEL))
    y_pred_masked = tf.boolean_mask(y_pred, tf.not_equal(y_true[:,:,:,0], IGNORE_LABEL))
    
    y_val, idx, counts = tf.unique_with_counts(y_true_masked)
    total = tf.cast(tf.reduce_sum(counts), tf.float32)
    ratios = tf.divide(1.0, tf.divide(tf.cast(counts, tf.float32), total))
    ratios = tf.divide(ratios, tf.math.reduce_min(ratios))
    sample_weights = tf.gather(ratios, tf.cast(idx, tf.int32), axis=0, batch_dims=-1)
    
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    L_classification = tf.reduce_mean(loss_fn(y_true_masked, y_pred_masked, sample_weight=sample_weights))
    if tf.equal(skip_box_loss, True):
        return L_classification

    L_boxes = tf.reduce_mean(box_loss(y_true_boxes, y_pred))
    return tf.math.add(L_classification, L_boxes)  
 
