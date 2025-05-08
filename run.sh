#!/bin/zsh

conda activate bertnado

# make a dataset
bertnado-data \
    --dataset_name=imdb \
    --dataset_dir=./data/imdb \
    --train_size=0.8 \
    --val_size=0.1 \
    --test_size=0.1