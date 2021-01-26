import os
import pickle

from models.gated_scnn.gated_shape_cnn.datasets.dad.dataset import DAD

SAVED_PKL_FILE = "saved_dad_paths.pkl"

def build_gscnn_dataset(dataset_dir, img_size, batch_size, seed):
    if os.path.exists(SAVED_PKL_FILE):
        _, class_mapping = pickle.load(open(SAVED_PKL_FILE, "rb"))
    
    dad_dataset_loader = DAD(
        batch_size,
        img_size,
        img_size,
        debug=False,
        data_dir=dataset_dir,
        n_classes=len(class_mapping),
        seed=seed)

    train = dad_dataset_loader.build_training_dataset()
    valid = dad_dataset_loader.build_validation_dataset()
    test = dad_dataset_loader.build_test_dataset()

    return train, valid, test, class_mapping

