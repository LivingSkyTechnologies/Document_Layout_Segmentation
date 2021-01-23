import os

from models.gated_scnn.gated_shape_cnn.datasets.publaynet.dataset import PubLayNet

class_mapping = {1: 'text', 2: 'title', 3: 'list', 4: 'table', 5: 'figure', 0: 'background'}


def build_gscnn_dataset(dataset_dir, img_size, batch_size, seed):
    publaynet_dataset_loader = PubLayNet(
        batch_size,
        img_size,
        img_size,
        debug=False,
        data_dir=dataset_dir,
        n_classes=len(class_mapping),
        seed=seed)

    train = publaynet_dataset_loader.build_training_dataset()
    valid = publaynet_dataset_loader.build_validation_dataset()
    test = publaynet_dataset_loader.build_test_dataset()

    return train, valid, test, class_mapping

