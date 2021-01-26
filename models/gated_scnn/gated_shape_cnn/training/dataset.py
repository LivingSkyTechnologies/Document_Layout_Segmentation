import tensorflow as tf
import numpy as np

class Dataset:
    """
    All custom datasets should inherit from this class. To do so you need to provide two methods
        self.get_paths(train) -> image_paths, label_paths, edge_paths
        self.flat_to_one_hot(labels, edges) -> converts flat segmentations (h, w) to one_hot (h, w, c)
    """

    def __init__(
            self,
            n_classes,
            batch_size,
            network_input_h,
            network_input_w,
            seed,
            debug,
            val_batch_size=2):
        """
        :param val_batch_size:
        :param batch_size:
        :param network_input_h height of training input:
        :param network_input_w width of training input:
        :param debug setting to true will give you a dataset with 1 element for both train and val:
        """
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.network_input_h = network_input_h
        self.network_input_w =  network_input_w
        self.seed = seed
        self.debug = debug
        self.val_batch_size = val_batch_size

    @staticmethod
    def image_path_process(path):
        raw = tf.io.read_file(path)
        #image = tf.image.decode_image(raw, channels=3)
        image = tf.image.decode_jpeg(raw, channels=3)
        image = tf.math.divide(tf.cast(image, tf.float32), 255.0)
        #image.set_shape([None, None, None])
        return image

    @staticmethod
    def label_path_process(path):
        raw = tf.io.read_file(path)
        label = tf.image.decode_png(raw, channels=1)
        return label

    def resize_and_normalize_boxes(self, boxes, img_height, img_width, target_size):
        boxes = np.array(boxes)
        x_scale = tf.cast(target_size / img_width, tf.float32)
        y_scale = tf.cast(target_size / img_height, tf.float32)
        boxes[:, 0] = (boxes[:, 0] * y_scale) / float(target_size)
        boxes[:, 1] = (boxes[:, 1] * x_scale) / float(target_size)
        boxes[:, 2] = (boxes[:, 2] * y_scale) / float(target_size)
        boxes[:, 3] = (boxes[:, 3] * x_scale) / float(target_size)
        return boxes

    def resize_images(self, image, label, edge_label, boxes):
        """
        :param image tensor:
        :param label tensor:
        :param edge_label tensor:
        :return resized data:

        resize data, for training all inputs are shaped (self.network_input_h, self.network_input_w)
        """
        orig_height = tf.shape(image)[0]
        orig_width = tf.shape(image)[1]
        image = tf.image.resize(image, (self.network_input_h, self.network_input_w))
        label = tf.image.resize(label, (self.network_input_h, self.network_input_w), method='nearest')
        edge_label = tf.image.resize(edge_label, (self.network_input_h, self.network_input_w), method='nearest')
        boxes = tf.py_function(self.resize_and_normalize_boxes, [boxes, orig_height, orig_width, self.network_input_h], tf.float32)
        return image, label, edge_label, boxes

    @staticmethod
    def paths_to_tensors(im_path, label_path, edge_label_path, box_path):
        """
        :param im_path:
        :param label_path:
        :param edge_label_path:
        :return image tensor [h, w, 3] tf.uint8; label [h, w, 1] tf.int32; edge [h, w, 1] tf.int32
        """
        image = Dataset.image_path_process(im_path)
        label = Dataset.label_path_process(label_path)
        edge_label = Dataset.label_path_process(edge_label_path)
        boxes = tf.py_function(Dataset.parse_boxes, [box_path], tf.float32)
        return image, label, edge_label, boxes

    @staticmethod
    def flip_boxes(boxes):
        gt_boxes = np.array(boxes)
        box_width = gt_boxes[:,3] - gt_boxes[:,1]
        gt_boxes[:, 3] = 1.0 - gt_boxes[:, 1]
        gt_boxes[:, 1] = 1.0 - gt_boxes[:, 1] - box_width

        return gt_boxes

    @staticmethod
    def random_flip(image, label, edge, boxes):
        """random left right flips"""
        if tf.equal(tf.cond(tf.greater_equal(tf.random.uniform(()), 0.5), lambda: 1, lambda: 0), 0):
            image = tf.image.flip_left_right(image)
            label = tf.image.flip_left_right(label)
            edge = tf.image.flip_left_right(edge)
            boxes = tf.py_function(Dataset.flip_boxes, [boxes], tf.float32)
        return image, label, edge, boxes

    def get_paths(self, train):
        raise NotImplementedError('you must implement this in sub class')

    def flat_to_one_hot(self, labels, edges):
        labels = tf.one_hot(labels[..., 0], self.n_classes)
        edges = tf.one_hot(edges[..., 0], 2)
        return labels, edges

    @staticmethod
    def validate_flat_to_one_hot(labels, edges):
        # make sure they are of shape [b, h, w, c]
        tf.debugging.assert_rank(labels, 4, 'label')
        tf.debugging.assert_rank(labels, 4, 'edges')

        # make sure have convincing number of channels
        tf.debugging.assert_shapes([
            (edges, ('b', 'h', 'w', 2)),
            (labels, ('b', 'h', 'w', 'c')),
        ])
        label_channels = tf.shape(labels)[-1]
        tf.assert_greater(label_channels, 1)

    def process_training_batch(self, images, labels, edges, boxes):
        """batch convert to one hot and apply colour jitter"""
        labels, edges = self.flat_to_one_hot(labels, edges)
        Dataset.validate_flat_to_one_hot(labels, edges)
        return images, labels, edges, boxes

    def process_validation_batch(self, images, labels, edges, boxes):
        """batch convert to one hot and make the image float32"""
        labels, edges = self.flat_to_one_hot(labels, edges)
        images = tf.cast(images, tf.float32)
        return images, labels, edges, boxes

    @staticmethod
    def parse_boxes(box_path):
        boxes = []
        #for box_path in box_paths:
        with open(box_path.numpy().decode(), 'r') as f:
            for line in f:
                line = line.strip()
                line = line.split(',')
                if len(line) == 5:
                    x, y, w, h = line[1:]
                    x_min = float(x)
                    y_min = float(y)
                    x_max = x_min + float(w)
                    y_max = y_min + float(h)
                    boxes.append(np.array([y_min, x_min, y_max, x_max]))
        return np.array(boxes)

    def get_raw_tensor_dataset(self, train, is_test=False):
        """
        :param train bool which data split to get:
        :return a dataset of tensors [(im, label, edge), ...]:
        """

        # get the paths to the data
        image_paths, label_paths, edge_label_paths = self.get_paths(train=train, is_test=is_test)
        
        box_paths = [x.replace("jpg", "txt") for x in image_paths]
        #boxes = self.parse_boxes(box_paths)

        dataset = tf.data.Dataset.from_tensor_slices((image_paths, label_paths, edge_label_paths, box_paths))
        if train:
            dataset = dataset.shuffle(500, reshuffle_each_iteration=True, seed=self.seed)

        # convert the paths to tensors
        dataset = dataset.map(Dataset.paths_to_tensors, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        return dataset

    def build_training_dataset(self):
        """
        training dataset
            - random left right flips
        """
        # get dataset of tensors (im, label, edge)
        dataset = self.get_raw_tensor_dataset(train=True)

        # training augmentations
        dataset = dataset.map(self.resize_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(Dataset.random_flip, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        # batch processing, convert to one hot, also apply colour jitter here
        # so we can do it on batch rather than per image
        dataset = dataset.padded_batch(self.batch_size, drop_remainder=True, padded_shapes=([None, None, 3], [None, None, None], [None, None, 2], [None, 4]))
        dataset = dataset.map(self.process_training_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        if self.debug:
            dataset = dataset.take(1)
        return dataset

    def build_validation_dataset(self):
        """
        val dataset:
            - full size images
            - no augmentations
            - fixed batch size of VAL_BATCH (=2)
        """
        # get dataset of tensors (im, label, edge)
        dataset = self.get_raw_tensor_dataset(train=False)

        # batch process
        dataset = dataset.map(self.resize_images, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.padded_batch(1, drop_remainder=True, padded_shapes=([None, None, 3], [None, None, None], [None, None, 2], [None, 4]))
        dataset = dataset.map(self.process_validation_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        if self.debug:
            dataset = dataset.take(1)
        return dataset

    def build_test_dataset(self):
        """
        val dataset:
            - full size images
            - no augmentations
            - fixed batch size of VAL_BATCH (=2)
        """
        # get dataset of tensors (im, label, edge)
        dataset = self.get_raw_tensor_dataset(train=False, is_test=True)

        # batch process
        dataset = dataset.map(self.resize_images, num_parallel_calls=tf.data.experimental.AUTOTUNE) 
        dataset = dataset.padded_batch(1, drop_remainder=True, padded_shapes=([None, None, 3], [None, None, None], [None, None, 2], [None, 4]))
        dataset = dataset.map(self.process_validation_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

        if self.debug:
            dataset = dataset.take(1)
        return dataset

