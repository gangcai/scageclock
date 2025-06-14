#!/bin/bash
mkdir -p tmp/
#testing for a single test file
#poetry run pytest tests/test_evaluation.py --k_fold_data_dir="data/pytest_data/k_fold_mode/" --test_data_dir="data/pytest_data/train_val_test_mode/" --meta_file="data/pytest_data/pytest_dataset_metadata.parquet" --out_root_dir="./tmp/" --model_file="./data/trained_models/GMA_models/GMA_celltype_balanced_basicRun.pth"

#fully testing
poetry run pytest  --k_fold_data_dir="data/pytest_data/k_fold_mode/" --test_data_dir="data/pytest_data/train_val_test_mode/" --meta_file="data/pytest_data/pytest_dataset_metadata.parquet" --out_root_dir="./tmp/" --model_file="./data/trained_models/GMA_models/GMA_celltype_balanced_basicRun.pth"
