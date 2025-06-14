# scAgeClock
scAgeClock: a single-cell human aging clock model based on gated multi-head attention neural networks and single-cell transcriptome
## installation
pip install scageclock
## example
### example data and model
- example data can be found at data/pytest_data
- example GMA model file can be found at data/trained_models/GMA_models

### making age prediction
```python
from scageclock.evaluation import prediction
model_file="./data/trained_models/GMA_models/GMA_celltype_balanced_basicRun.pth"
h5ad_folder="./data/pytest_data/train_val_test_mode/test/"
results_df = prediction(model_file=model_file,
		    h5ad_dir=h5ad_folder)
```
### making age prediction (command-line version)
```bash
#!/bin/bash
model_file="./data/trained_models/GMA_models/GMA_celltype_balanced_basicRun.pth"
h5ad_folder="./data/pytest_data/train_val_test_mode/test/"
scAgeClock --model_file ${model_file} --testing_h5ad_files_dir ${h5ad_folder} --output_file './tmp/test_predicted.xlsx'
```

### model training with testing
```python
from scageclock.scAgeClock import training_pipeline
model_name = "GMA" # Gated Multihead Attention Neural Network, default model of scAgeClock
ad_dir_root = "data/pytest_data/train_val_test_mode/"
meta_file = "data/pytest_data/pytest_dataset_metadata.parquet"
dataset_folder_dict = {"training": "train", "validation": "val", "testing": "test"}
predict_dataset = "testing"
loader_method = "scageclock"
out_root_dir = "./tmp/"
results = training_pipeline(model_name=model_name,
			    ad_dir_root=ad_dir_root,
			    meta_file_path=meta_file,
			    dataset_folder_dict=dataset_folder_dict,
			    predict_dataset=predict_dataset,
			    validation_during_training=True,
			    loader_method=loader_method,
			    out_root_dir=out_root_dir)
```

### model training with cross-validation
```python
from scageclock.scAgeClock import training_pipeline
model_name = "GMA" # Gated Multihead Attention Neural Network, default model of scAgeClock
k_fold_data_dir="data/pytest_data/k_fold_mode/" # h5ad files are located at train_val/Fold1; train_val/Fold2; train_val/Fold3
meta_file = "data/pytest_data/pytest_dataset_metadata.parquet"
dataset_folder_dict = {"training_validation": "train_val"}
predict_dataset = "validation" ## prediction based on the validation dataset
loader_method = "scageclock"
out_root_dir = "./tmp/"

results = training_pipeline(model_name=model_name,
			ad_dir_root=k_fold_data_dir,
			meta_file_path=meta_file,
			dataset_folder_dict=dataset_folder_dict,
			predict_dataset=predict_dataset,
			K_fold_mode=True,
			K_fold_train=("Fold1", "Fold2"),
			K_fold_val="Fold3",
			validation_during_training=False,
			loader_method=loader_method,
			out_root_dir=out_root_dir)
```

### model training with cross-validation (catboost)
```python
from scageclock.scAgeClock import training_pipeline
model_name = "catboost" # Gated Multihead Attention Neural Network, default model of scAgeClock
ad_dir_root = "data/pytest_data/train_val_test_mode/"
meta_file = "data/pytest_data/pytest_dataset_metadata.parquet"
dataset_folder_dict = {"training": "train", "validation": "val", "testing": "test"}
predict_dataset = "testing"
loader_method = "scageclock"
out_root_dir = "./tmp/"
results = training_pipeline(model_name=model_name,
			    ad_dir_root=ad_dir_root,
			    meta_file_path=meta_file,
			    dataset_folder_dict=dataset_folder_dict,
			    predict_dataset=predict_dataset,
			    validation_during_training=True,
			    loader_method=loader_method,
			    train_dataset_fully_loaded=True, ##make sure the memory is enough
			    out_root_dir=out_root_dir)

```
## about
authors: Gangcai Xie (Medical School of Nantong University); [ORCID](https://orcid.org/0000-0002-8286-2987)
