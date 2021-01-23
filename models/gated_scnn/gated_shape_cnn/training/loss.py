import tensorflow as tf
import tensorflow_addons as tfa
from models.gated_scnn.gated_shape_cnn.model.layers import gradient_mag


def _generalised_dice(y_true, y_pred, eps=0.0, from_logits=True):
    """
    :param y_true [b, h, w, c]:
    :param y_pred [b, h, w, c]:
    :param eps weight fudge factor for zero counts:
    :return generalised dice loss:

    see https://www.nature.com/articles/s41598-018-26350-3
    """

    # [b, h, w, classes]
    if from_logits:
        y_pred = tf.nn.softmax(y_pred)
    y_true_shape = tf.shape(y_true)

    # [b, h*w, classes]
    y_true = tf.reshape(y_true, [-1, y_true_shape[1] * y_true_shape[2], y_true_shape[3]])
    y_pred = tf.reshape(y_pred, [-1, y_true_shape[1] * y_true_shape[2], y_true_shape[3]])

    # [b, classes]
    counts = tf.reduce_sum(y_true, axis=1)
    weights = 1. / counts**2
    weights = tf.where(tf.math.is_finite(weights), weights, eps)

    multed = tf.reduce_sum(y_true * y_pred, axis=1)
    summed = tf.reduce_sum(y_true + y_pred, axis=1)

    # [b]
    numerators = tf.reduce_sum(weights * multed, axis=-1)
    denom = tf.reduce_sum(weights * summed, axis=-1)
    dices = 1. - 2. * numerators / denom
    dices = tf.where(tf.math.is_finite(dices), dices, tf.zeros_like(dices))
    return tf.reduce_mean(dices)


def _gumbel_softmax(logits, eps=1e-8, tau=1.):
    """

    :param logits:
    :param eps:
    :param tau temprature:
    :return soft approximation to argmax:

    see https://arxiv.org/abs/1611.01144
    """
    g = tf.random.uniform(tf.shape(logits))
    g = -tf.math.log(eps - tf.math.log(g + eps))
    return tf.nn.softmax((logits + g) / tau)


def _segmentation_edge_loss(gt_tensor, logit_tensor, thresh=0.8):
    """

    :param gt_tensor [b, h, w, c] segmentation labels:
    :param pred_tensor [b, h, w, c] segmentation logits:
    :param thresh intensity to be considered edge:
    :return the difference in boundaries between predicted versus actual
            where the boundaries come from the segmentation, rather than
            the shape head:
    """

    # soft approximation to argmax, so we can build an edge
    logit_tensor = _gumbel_softmax(logit_tensor)

    # normalised image gradients to give us edges
    # images will be [b, h, w, n_classes]
    gt_edges = gradient_mag(gt_tensor)
    pred_edges = gradient_mag(logit_tensor)

    # [b*h*w, n]
    gt_edges = tf.reshape(gt_edges, [-1, tf.shape(gt_edges)[-1]])
    pred_edges = tf.reshape(pred_edges, [-1, tf.shape(gt_edges)[-1]])

    # take the difference between these two gradient magnitudes
    # we will first take all the edges from the ground truth image
    # and then all the edges from the predicted
    edge_difference = tf.abs(gt_edges - pred_edges)

    # gt edges and disagreement with pred
    mask_gt = tf.cast((gt_edges > thresh ** 2), tf.float32)
    contrib_0 = tf.boolean_mask(edge_difference, mask_gt)

    contrib_0 = tf.cond(
        tf.greater(tf.size(contrib_0), 0),
        lambda: tf.reduce_mean(contrib_0),
        lambda: 0.)

    # vice versa
    mask_pred = tf.stop_gradient(tf.cast((pred_edges > thresh ** 2), tf.float32))
    contrib_1 = tf.reduce_mean(tf.boolean_mask(edge_difference, mask_pred))
    contrib_1 = tf.cond(
        tf.greater(tf.size(contrib_1), 0),
        lambda: tf.reduce_mean(contrib_1),
        lambda: 0.)
    return tf.reduce_mean(0.5 * contrib_0 + 0.5 * contrib_1)


def _shape_edge_loss(gt_tensor, pred_tensor, pred_shape_tensor, keep_mask, thresh=0.8):
    """
    :param gt_tensor [b, h, w, c]:
    :param pred_tensor [b, h, w, c]:
    :param pred_shape_tensor [b, h, w, 1]:
    :param keep_mask binary mask of pixels to keep (eg in cityscapes we ignore 255):
    :param thresh probability to consider an edge in our prediction:
    :return cross entropy of classifications near on an edge:

     whereever we have predicted an edge, calculated the cross entropy there.
    This penalises the edges more strongly, encouraging them to be correct at the boundary
    """

    # where we have predicted an edge and which are pixels
    # we care about
    mask = pred_shape_tensor > thresh
    mask = tf.stop_gradient(mask[..., 0])
    mask = tf.logical_and(mask, keep_mask)

    # get relavent predicitons and truth
    gt = gt_tensor[mask]
    pred = pred_tensor[mask]

    # cross entropy, we may not have any edges, in which case return 0
    if tf.reduce_sum(tf.cast(mask, tf.float32)) > 0:
        return tf.reduce_mean(tf.losses.categorical_crossentropy(gt, pred, from_logits=True))
    else:
        return 0.


def _weighted_cross_entropy(y_true, y_pred, keep_mask):
    """

    :param y_true [b, h, w, c]:
    :param y_pred [b, h, w, c]:
    :return weighted cross entropy:
    """

    # ignore zertain pixels
    # makes both tensors [n, c]
    y_true = y_true[keep_mask]
    y_pred = y_pred[keep_mask]

    # weights
    #rs = tf.reduce_sum(y_true, axis=0, keepdims=True)
    #N = tf.cast(tf.shape(y_true)[0], tf.float32)
    #weights = (N - rs)/N + 1

    y_vals, idx, counts = tf.unique_with_counts(tf.argmax(y_true, axis=-1))
    total = tf.cast(tf.reduce_sum(counts), tf.float32)
    ratios = tf.divide(1.0, tf.divide(tf.cast(counts, tf.float32), total))
    ratios = tf.divide(ratios, tf.math.reduce_min(ratios))
    weights = tf.gather(ratios, tf.cast(idx, tf.int32), axis=0, batch_dims=-1)

    # everything here is one hot so this essentially picks the class weight
    # per row of y_true
    #weights = tf.reduce_sum(y_true*weights, axis=1)

    # compute your (unweighted) softmax cross entropy loss
    #unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(y_true, y_pred)
    #weighted_losses = unweighted_losses * weights
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    weighted_losses = cce(y_true, y_pred, sample_weight=weights)
    loss = tf.reduce_mean(weighted_losses)
    return loss

import tensorflow_addons as tfa

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

def loss(gt_label, gt_boxes, logits, shape_head, edge_label, loss_weights):
    tf.debugging.assert_shapes([
        (gt_label,     ('b', 'h', 'w', 'c')),
        (logits,       ('b', 'h', 'w', 'c')),
        (shape_head,   ('b', 'h', 'w', 1)),
        (edge_label,   ('b', 'h', 'w', 2)),
        (loss_weights, (5,))],)

    # in cityscapes we ignore some classes, which means that there will
    # be pixels without any class
    keep_mask = tf.reduce_any(gt_label == 1., axis=-1)
    anything_active = tf.reduce_any(keep_mask)

    # standard weighted cross entropy
    # we weight each class by 1 + (1 - batch_prob_of_class)
    # where we get the prob by counting ground truth pixels
    seg_loss = tf.cond(
        anything_active,
        lambda: _weighted_cross_entropy(gt_label, logits, keep_mask) * loss_weights[0],
        lambda: 0.)

    # Generalised dice loss on the edges predicted by the network
    shape_probs = tf.concat([1. - shape_head, shape_head], axis=-1)
    edge_loss = _generalised_dice(edge_label, shape_probs) * loss_weights[1]

    # regularizing loss
    # this ensures that the edges themselves are consistent
    edge_consistency = _segmentation_edge_loss(gt_label, logits) * loss_weights[2]
    # this ensures that the classifcatiomn at the edges is correct
    edge_class_consistency = tf.cond(
        anything_active,
        lambda: _shape_edge_loss(gt_label, logits, shape_head, keep_mask) * loss_weights[3],
        lambda: 0.)

    box_losses = tf.reduce_mean(box_loss(gt_boxes, logits)) * loss_weights[4]
    #box_loss = 0.0 

    return seg_loss, edge_loss, edge_class_consistency, edge_consistency, box_losses

