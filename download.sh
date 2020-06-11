#!bin/bash

DATASET_URL='http://slazebni.cs.illinois.edu/research/uiuc_texture_dataset.zip'
DATASET_ZIP=`basename $DATASET_URL`
SAVE_PATH='data/uiuc_texture_dataset'

wget $DATASET_URL
mkdir -p $SAVE_PATH
unzip -d $SAVE_PATH $DATASET_ZIP
rm $DATASET_ZIP

