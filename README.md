# document-layout-segmentation

This repository provides a framework to train segmentation models to segment document layouts. Currently, the supported training datasets include [DAD](https://github.com/logan-markewich/DAD-Dense-Article-Dataset) and [PubLayNet](https://developer.ibm.com/technologies/artificial-intelligence/data/publaynet/).

[Setup](#Setup)

[Training](#Training)

[Inference](#Inference)

[Credits](#Credits)

## Setup
### Dependencies
This repo has been tested only with tensorflow-gpu==2.3.1 and tensorflow-addons=0.11.2 using python3.6. The full list of pip dependencies is in requirements.txt.

Additionally, there is a minor dependence on tensorflow-examples. This can be installed using:
```
pip install -q git+https://github.com/tensorflow/examples.git
```

### Dataset Preparation
The repo contains empty "dad" and "publaynet" folders.

#### DAD
To setup DAD, run the following starting from the repo root dir:

```
mkdir dad/documents
mkdir dad/annotations
cd ..
git clone https://github.com/logan-markewich/DAD-Dense-Article-Dataset
cp DAD-dense-article-dataset/Articles/*/JSON_Rev1/* ./document-layout-segmentation/dad/annotations/
cp -r DAD-dense-article-dataset/Articles/*/IMAGE/* ./document-layout-segmentation/dad/docouments/
```
Then, your dad folder should have an annotations folder full of json files, while the documents folder is full of folders for each journal article.

#### PubLayNet
To setup PubLayNet, run the following from the repo root dir:
```
wget https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/train-0.tar.gz
tar -xzf train-0.tar.gz
rm -f train-0.tar.gz
wget https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/val.tar.gz
tar -xzf val.tar.gz
rm -f val.tar.gz
wget https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/labels.tar.gz
tar -xzf labels.tar.gz
rm -f labels.tar.gz
```
This code downloads the train-0 publaynet tarball (you could also add more from [PubLayNet's download page](https://dax-cdn.cdn.appdomain.cloud/dax-publaynet/1.0.0/PubLayNet.html?_ga=2.26184379.726029236.1609705747-1098120955.1605642577&cm_mc_uid=21438371251816056425771&cm_mc_sid_50200000=85056661609705747259), the val tarball, and the labels tarball. It then extracts everything into the publaynet folder into the proper structure.

## Training
### Help
```
python ./train.py -h
usage: train.py [-h] [--model MODEL] [--ignore-class IGNORE_CLASS]
                [--patience PATIENCE] [--base-lr BASE_LR] [--box-loss]
                [--dataset DATASET] [--dataset-dir DATASET_DIR]
                [--img-size IMG_SIZE] [--batch-size BATCH_SIZE] [--seed SEED]

Train a segmentation model.

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         One of "unet", "fast_fcn", "gated_scnn", or
                        "deeplabv3plus".
  --ignore-class IGNORE_CLASS
                        Class number to ignore. Defualt 255.
  --patience PATIENCE   Set how many epochs to wait for val loss to increase.
                        Default 5.
  --base-lr BASE_LR     Set initial learning rate. After val loss stops
                        increasing for number of epochs specified by
                        --patience, the model reloads to the best point and
                        divides the learning rate by 10 for fine tuning.
                        Default 1.0e-4.
  --box-loss            If set, use box loss regression during loss
                        calculation
  --dataset DATASET     Either "dad" or "publaynet".
  --dataset-dir DATASET_DIR
                        Root folder of the dataset.
  --img-size IMG_SIZE   Size of input image to train on. Default 512.
  --batch-size BATCH_SIZE
                        Batch size of datasets. Default 8.
  --seed SEED           The seed for all random functions. Default 45.
```
### Example
```
python ./train.py --model unet \
--ignore-class 255 \
--patience 5 \
--base-lr 0.0001 \
--box-loss \
--dataset dad \
--dataset-dir ./dad 
--img-size 512 \
--batch-size 16 \
--seed 42
```

## Inference

The inference script is able to predict segmentation masks for either a single image or a folder of images. The outputs are saved next to the inputs.

The labelme annotations and connected component analysis (CCA) options use the same algorithm. CCA is performed on the image and bounding boxes are extracted for each region. The label of each region is based on the median class in the region (this sometimes results in a tie, in which we just pick the class with the higher number. This could be expanded to expect certain rules). Objects less than 200 pixels of area are filtered out. 

Additionally, when writing the labelme json, we assume only lists and math formulas can fully overlap, all other objects are excluded if they overlap another by over 50% of the smaller objects area.

### Help
```
python ./inference.py -h
usage: inference.py [-h]
                    (--input-image INPUT_IMAGE | --input-folder INPUT_FOLDER)
                    [--saved-model SAVED_MODEL] [--model MODEL]
                    [--img-size IMG_SIZE] [--saved-pkl SAVED_PKL]
                    [--apply-cca] [--visualize] [--write-annotation]

optional arguments:
  -h, --help            show this help message and exit
  --input-image INPUT_IMAGE
                        Single image to segment.
  --input-folder INPUT_FOLDER
                        Folder of images to segment.
  --saved-model SAVED_MODEL
                        Directory or h5 file with a saved model.
  --model MODEL         One of "unet", "fast_fcn", "deeplabv3plus", or
                        "gated_scnn".
  --img-size IMG_SIZE   Size of images. Should match the size trained on.
  --saved-pkl SAVED_PKL
                        The saved PKL file from the training step. It is used
                        for the class mapping.
  --apply-cca           Post process with conncected component analysis. Makes
                        segmentations uniform, but might miss objects.
  --visualize           If set, will open a matplotlib plot to see the
                        segmentation visually.
  --write-annotation    If set, will also write a json file with annotations
                        in labelme format.
```

Example
```
python ./inference.py --input-image ./publaynet/train/PMC3388004_00003.jpg \
--saved-model ./final_models/deeplabv3_plus_box_loss.h5 \
--model deeplabv3plus \
--img-size 512 \
--saved-pkl ./saved_dad_paths.pkl \
--apply-cca \
--visualize \
--write-annotation
```
The above example performs inference on a publaynet image. The png mask is saved next to the input image in ./publaynet/train/PMC3388004_00003_mask.png.

Since --write-annotation is present, a labelme json file is also saved at ./publaynet/train/PMC3388004_00003.json containing the extracted bounding boxes for each object.

## Credits
MetricsUtils: based on https://github.com/GeorgeSeif/Semantic-Segmentation-Suite/blob/master/utils/utils.py

DataLoaderUtils: based on https://github.com/dhassault/tf-semantic-example/blob/master/01_semantic_segmentation_basic.ipynb

DeepLabV3Plus Model Definition: based on https://github.com/srihari-humbarwadi/DeepLabV3_Plus-Tensorflow2.0

Gated-SCNN Training and Dataset Loading: based on https://github.com/ben-davidson-6/Gated-SCNN

FastFCN: translated from pytorch to tensorflow from https://github.com/wuhuikai/FastFCN

PubLayNet Mask Generation: based on https://github.com/Lambert-Shirzad/PubLayNet_tfrecords

