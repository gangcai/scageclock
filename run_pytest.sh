#!/bin/bash
mkdir -p tmp/
poetry run pytest  --k_fold_data_dir="data/pytest_data/k_fold_mode/" --test_data_dir="data/pytest_data/train_val_test_mode/" --meta_file="data/pytest_data/pytest_dataset_metadata.parquet" --out_root_dir="./tmp/"
