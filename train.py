import argparse
import numpy as np
import tensorflow as tf

from datasets.DatasetBuilder import get_dataset
from models.gated_scnn.gated_shape_cnn.training.loss import loss as gscnn_loss
from loss import seg_loss, SegmentationAccuracy
from utils.MetricsUtils import evaluate_segmentation
from models.ModelBuilder import build_model

def create_mask(pred_mask):
    """Return a filter mask with the top 1 predicitons
    only.

    Parameters
    ----------
    pred_mask : tf.Tensor
        A [IMG_SIZE, IMG_SIZE, N_CLASS] tensor. For each pixel we have
        N_CLASS values (vector) which represents the probability of the pixel
        being these classes. Example: A pixel with the vector [0.0, 0.0, 1.0]
        has been predicted class 2 with a probability of 100%.

    Returns
    -------
    tf.Tensor
        A [IMG_SIZE, IMG_SIZE, 1] mask with top 1 predictions
        for each pixels.
    """
    # pred_mask -> [IMG_SIZE, SIZE, N_CLASS]
    # 1 prediction for each class but we want the highest score only
    # so we use argmax
    pred_mask = tf.argmax(pred_mask, axis=-1)
    # pred_mask becomes [IMG_SIZE, IMG_SIZE]
    # but matplotlib and others need [IMG_SIZE, IMG_SIZE, 1]
    pred_mask = tf.expand_dims(pred_mask, axis=-1)
    return pred_mask
 

def report_results(model_name, test, valid, class_mapping, is_gscnn=False):
    model = tf.keras.models.load_model(model_name, 
                                   compile=False)

    f1_avg = 0
    test_count = 0
    avg_test_class_iou = np.zeros((len(class_mapping),))
    avg_test_class_accuracies = np.zeros((len(class_mapping),))
    avg_test_class_count = np.zeros((len(class_mapping),))
    for sample in test:
        for input_image, golden_mask in zip(sample[0], sample[1]):
            one_img_batch = input_image[tf.newaxis, ...]
            y_pred = model(one_img_batch)
            tf.keras.backend.clear_session()
            
            if is_gscnn:
                y_pred = y_pred[...,:-1]
                golden_mask = golden_mask[tf.newaxis,...]
                keep_mask = tf.reduce_any(golden_mask == 1., axis=-1)
                flat_label = tf.argmax(golden_mask, axis=-1)
                flat_label = tf.where(keep_mask, flat_label, tf.cast(0, tf.int64), )
                flat_pred = tf.argmax(y_pred, axis=-1)
        
                flat_label_masked = flat_label[keep_mask]
                flat_pred_masked = flat_pred[keep_mask]
                
                pred_mask = flat_pred_masked.numpy()
                golden_mask = flat_label_masked
            else:
                pred_mask = create_mask(y_pred[0]).numpy()
            
            global_accuracy, class_accuracies, prec, rec, f1, iou = evaluate_segmentation(pred_mask.astype(int), golden_mask.numpy().astype(int), len(class_mapping))
            f1_avg += f1
            for i, val in class_accuracies.items():
                avg_test_class_accuracies[i] += val
                avg_test_class_iou[i] += iou[i]
                avg_test_class_count[i] += 1
            test_count += 1
            if test_count % 200 == 0:
                print("Acc. step {} on test set".format(test_count))

    f1_avg = f1_avg/float(test_count)
    print("Just test F1 Score was {}".format(f1_avg))

    f1_avg = 0
    valid_count = 0
    avg_valid_class_accuracies = np.zeros((len(class_mapping),))
    avg_valid_class_iou = np.zeros((len(class_mapping),))
    avg_valid_class_count = np.zeros((len(class_mapping),))
    for sample in valid:
        for input_image, golden_mask in zip(sample[0], sample[1]):
            one_img_batch = input_image[tf.newaxis, ...]
            y_pred = model(one_img_batch)
            tf.keras.backend.clear_session()
            
            if is_gscnn:
                y_pred = y_pred[...,:-1]
                golden_mask = golden_mask[tf.newaxis,...]
                keep_mask = tf.reduce_any(golden_mask == 1., axis=-1)
                flat_label = tf.argmax(golden_mask, axis=-1)
                flat_label = tf.where(keep_mask, flat_label, tf.cast(0, tf.int64), )
                flat_pred = tf.argmax(y_pred, axis=-1)
        
                flat_label_masked = flat_label[keep_mask]
                flat_pred_masked = flat_pred[keep_mask]
                
                pred_mask = flat_pred_masked.numpy()
                golden_mask = flat_label_masked
            else:
                pred_mask = create_mask(y_pred[0]).numpy()

            global_accuracy, class_accuracies, prec, rec, f1, iou = evaluate_segmentation(pred_mask.astype(int), golden_mask.numpy().astype(int), len(class_mapping))
            f1_avg += f1
            for i, val in class_accuracies.items():
                avg_valid_class_accuracies[i] += val
                avg_valid_class_iou[i] += iou[i]
                avg_valid_class_count[i] += 1
            valid_count += 1
            if valid_count % 200 == 0:
                print("Acc. step {} on val set".format(valid_count))

    f1_avg = f1_avg/float(valid_count)
    print("Just validation F1 Score was {}".format(f1_avg))

    print("Class Accuracies:")
    print("|Class Name|Test|Val|")
    print("|----------|----|---|")
    for i, name in class_mapping.items():
        test_acc_score = avg_test_class_accuracies[i]
        test_class_count = avg_test_class_count[i]
        valid_acc_score = avg_valid_class_accuracies[i]
        valid_class_count = avg_valid_class_count[i]
        print("|%s|%f%%|%f%%|" % (name,
                                  test_acc_score/float(test_class_count)*100.0, 
                                  valid_acc_score/float(valid_class_count)*100.0))

    print("\n")
    print("Class mIOU:")
    print("|Class Name|Test|Val|")
    print("|----------|----|---|")
    for i, name in class_mapping.items():
        test_iou_score = avg_test_class_iou[i]
        test_class_count = avg_test_class_count[i]
        valid_iou_score = avg_valid_class_iou[i]
        valid_class_count = avg_valid_class_count[i]
        print("|%s|%f%%|%f%%|" % (name,
                                  test_iou_score/float(test_class_count)*100.0,
                                  valid_iou_score/float(valid_class_count)*100.0))

def train_gscnn(model, train, valid, lr, patience, model_name):
    def calc_loss(model, input_image, gt_mask, gt_edges, gt_boxes, training, loss_weights):
        out = model(input_image, training=training)
        prediction, shape_head = out[..., :-1], out[..., -1:]

        loss_val = tf.reduce_sum(gscnn_loss(gt_mask, gt_boxes, prediction, shape_head, gt_edges, loss_weights))
        reg_loss = tf.reduce_sum(model.losses)
        return tf.math.add(loss_val, reg_loss)

    def grad(model, input_image, gt_mask, gt_edges, gt_boxes, loss_weights):
        with tf.GradientTape() as tape:
            loss_val = calc_loss(model, input_image, gt_mask, gt_edges, gt_boxes, True, loss_weights)
        return loss_val, tape.gradient(loss_val, model.trainable_variables)

    loss_weights = (1., 1., 1., 1., 1.)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    best_val_loss = 100000000.0
    num_bad_iters = 0
    num_epochs = 100
    lr_decreased = False
    for epoch in range(num_epochs):
        if num_bad_iters >= patience and lr_decreased:
            print("Val Loss is not improving, exiting...")
            break
        elif num_bad_iters >= patience:
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
            loss_value = calc_loss(model, input_image, gt_mask, gt_edges, gt_boxes, False, loss_weights)
        
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

def train_generic(model, train, valid, lr, patience, model_name):
    def calc_loss(model, input_image, gt_mask, gt_boxes, training):
        predicted_mask = model(input_image, training=training)
        return seg_loss(gt_mask, predicted_mask, gt_boxes)

    def grad(model, input_image, gt_mask, gt_boxes):
        with tf.GradientTape() as tape:
            loss_value = calc_loss(model, input_image, gt_mask, gt_boxes, True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    best_val_loss = 1000000.0
    num_bad_iters = 0
    num_epochs = 100
    lr_decreased = False
    for epoch in range(num_epochs):
        if num_bad_iters >= patience and lr_decreased:
            print("Val Loss is not improving, exiting...")
            break
        elif num_bad_iters >= patience:
            print("Lowering lr, restarting from best model")
            lr_decreased = True
            num_bad_iters = 0
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
            model = tf.keras.models.load_model(model_name, compile=False)

        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = SegmentationAccuracy() if not "gated" in model_name else tf.keras.metrics.Accuracy()
    
        epoch_val_loss_avg = tf.keras.metrics.Mean()
        epoch_val_accuracy = SegmentationAccuracy() if not "gated" in model_name else tf.keras.metrics.Accuracy()
    
        step = 0
        for input_image, gt_mask, gt_boxes in train:
            loss_value, grads = grad(model, input_image, gt_mask, gt_boxes)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            epoch_loss_avg.update_state(loss_value)
            epoch_accuracy.update_state(gt_mask, model(input_image, training=True))

            step += 1
            if step % 100 == 0:
                print("Step {}: Loss: {:.3f}, Accuracy: {:.3%}".format(step, epoch_loss_avg.result(), epoch_accuracy.result()))
    
        for input_image, gt_mask, gt_boxes in valid:
            loss_value = calc_loss(model, input_image, gt_mask, gt_boxes, training=False)
            epoch_val_loss_avg.update_state(loss_value)
            epoch_val_accuracy.update_state(gt_mask, model(input_image, training=False))

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a segmentation model.')
    
    parser.add_argument('--model', help='One of "unet", "fast_fcn", "gated_scnn", or "deeplabv3plus".')
    parser.add_argument('--ignore-class', type=int, default=255, help='Class number to ignore. Defualt 255.')
    parser.add_argument('--patience', type=int, default=5, help='Set how many epochs to wait for val loss to increase. Default 5.')
    parser.add_argument('--base-lr', type=float, default=1.0e-4, help='Set initial learning rate. After val loss stops increasing for number of epochs specified by --patience, the model reloads to the best point and divides the learning rate by 10 for fine tuning. Default 1.0e-4.')
    parser.add_argument('--box-loss', default=False, action='store_true', help='If set, use box loss regression during loss calculation')

    parser.add_argument('--dataset', help='Either "dad" or "publaynet".')
    parser.add_argument('--dataset-dir', help='Root folder of the dataset.')
    parser.add_argument('--img-size', type=int, default=512, help='Size of input image to train on. Default 512.')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size of datasets. Default 8.')
    parser.add_argument('--seed', type=int, default=45, help='The seed for all random functions. Default 45.')
    
    # Parse the command args
    args = parser.parse_args()
    
    # Build the requested dataset and get the int->label class mapping
    print("Building dataset...\n")
    dataset_builder = get_dataset(args.dataset, args.model)
    train, valid, test, class_mapping = dataset_builder(args.dataset_dir, args.img_size, args.batch_size, args.seed)

    # Build the specified segmentation model
    print("Building model...\n")
    model = build_model(args.model, args.img_size, len(class_mapping))

    # Train the model
    print("Starting train loop...\n")
    model_name = args.model + "_{}_best.h5".format(args.dataset)
    if args.model == "gated_scnn":
        model_name = args.model + "_{}_best".format(args.dataset)
        train_gscnn(model, train, valid, args.base_lr, args.patience, model_name)
    else:
        train_generic(model, train, valid, args.base_lr, args.patience, model_name)
    
    # Report stats from the test set
    print("Gathering accuracy statistics...\n")
    report_results(model_name, test, valid, class_mapping, is_gscnn=args.model == "gated_scnn")

    print("\nCOMPLETE: Model saved to {}\n".format(model_name))

