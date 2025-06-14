#!/bin/bash
model_file="./data/trained_models/GMA_models/GMA_celltype_balanced_basicRun.pth"
h5ad_folder="./data/pytest_data/train_val_test_mode/test/"
poetry run scAgeClock --model_file ${model_file} --testing_h5ad_files_dir ${h5ad_folder} --output_file './tmp/test_predicted.xlsx'
