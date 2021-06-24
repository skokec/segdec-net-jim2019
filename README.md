# Surface Defect Detection with Segmentation-Decision Network on KolektorSDD

Official TensorFlow implementation for [Segmentation-based deep-learning approach for surface-defect detection](https://prints.vicos.si/publications/370) that uses segmentation and decision networks for the detection of surface defects. This work was done in collaboration with [Kolektor Group d.o.o.](http://www.kolektorvision.com/en/).

[![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa] 

Code and the dataset are licensed under [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa]. For comerical use please contact danijel.skocaj@fri.uni-lj.si.

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg

# Citation:

Please cite JIM 2019 journal paper:

```
@article{Tabernik2019JIM,
  author = {Tabernik, Domen and {\v{S}}ela, Samo and Skvar{\v{c}}, Jure and Sko{\v{c}}aj, Danijel},
  journal = {Journal of Intelligent Manufacturing},
  title = {{Segmentation-Based Deep-Learning Approach for Surface-Defect Detection}},
  year = {2019},
  month = {May},
  day = {15},
  issn={1572-8145},
  doi={10.1007/s10845-019-01476-x}
}

```

# Dependencies:

* python2.7
* TensorFlow r1.1 or newer (tested up to r1.8)
* python libs: numpy, scipy, six, PIL, sklearn, pylab, matplotlib


# Dataset

The full dataset Kolektor Surface Defect Dataset (KolektorSDD) is available [here](https://www.vicos.si/Downloads/KolektorSDD).

We split the dataset into three folds to perform 3-fold cross validation. The splits are available at [http://box.vicos.si/skokec/gostop/KolektorSDD-training-splits.zip](http://box.vicos.si/skokec/gostop/KolektorSDD-training-splits.zip).

Fully prepared TensorFlow dataset split into 3 folds is available at [http://box.vicos.si/skokec/gostop/KolektorSDD-dilate=5-tensorflow.zip](http://box.vicos.si/skokec/gostop/KolektorSDD-dilate=5-tensorflow.zip).


# Usage of training/evaluation code

The following files are used to train/evaluate the model:
* `segdec_train.py`: MAIN ENTRY for training and evaluation
* `segdec_model.py`: model file for the network
* `segdec_data.py`: dataset class for training the model


Using the TensorFlow ready [KolektorSDD](https://www.vicos.si/Downloads/KolektorSDD) (with dilate=5 for mask) dataset you can train and evaluate with the following:



```bash

# 1. Download and extract `KolektorSDD-dilate=5-tensorflow.zip`
mkdir db
cd db
wget http://box.vicos.si/skokec/gostop/KolektorSDD-dilate=5-tensorflow.zip
unzip -x KolektorSDD-dilate=5-tensorflow.zip
cd ..


# Empty folder where models/results will be stored
export OUTPUT_FOLDER=`pwd`/output

# folder where `KolektorSDD-dilate=5-tensorflow.zip` is extracted (must contain `KolektorSDD-dilate=5` subfolder).
export DATASET_FOLDER=`pwd`/db

mkdir $OUTPUT_FOLDER

# 2. Train only segmentation network first:

python -u segdec_train.py --fold=0,1,2 --gpu=0 --max_steps=6600 --train_subset=train \
        --seg_net_type=ENTROPY \
        --size_height=1408 \
        --size_width=512 \
        --with_seg_net=True \
        --with_decision_net=False \
        --storage_dir=$OUTPUT_FOLDER \
        --dataset_dir=$DATASET_FOLDER \
        --datasets=KolektorSDD-dilate=5 \
        --name_prefix=full-size_cross-entropy

# 3. Train and evaluate decision network based on existing segmentation network:

# The `--pretrained_main_folder` must point to the folder where 'fold_XY' subfolders with the trained segmentation models are.
# NOTE: Getting several `Not found: Key tower_0//decision` warrnings when loading the model is OK since the pre-trained model does not have decision net layers yet.

python -u segdec_train.py --fold=0,1,2 --gpu=0 --max_steps=6600 --train_subset=train \
            --seg_net_type=ENTROPY \
            --size_height=1408 \
            --size_width=512 \
            --with_seg_net=False \
            --with_decision_net=True \
            --storage_dir=$OUTPUT_FOLDER \
            --dataset_dir=$DATASET_FOLDER \
            --datasets=KolektorSDD-dilate=5 \
            --name_prefix=decision-net_full-size_cross-entropy \
            --pretrained_main_folder=$OUTPUT_FOLDER/segdec_train/KolektorSDD-dilate=5/full-size_cross-entropy


# 4. Print evaluation metrics combined from all folds

python -u segdec_print_eval.py $OUTPUT_FOLDER/segdec_eval/KolektorSDD-dilate=5/decision-net_full-size_cross-entropy

```

Note: The model is sensitive to random data shuffles during the training and will lead to different performance with different runs.
