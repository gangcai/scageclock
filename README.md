# scAgeClock
scAgeClock: a single-cell human aging clock model based on gated multi-head attention neural networks and single-cell transcriptome
## installation
pip install scageclock
## about data
### data basic
- feature file: data/metadata/h5ad_var.tsv
- categorical index: data/metadata/categorical_features_index (assay, sex, tissue_general, and cell_type)
- h5ad example file: data/pytest_data/k_fold_mode/train_val/Fold1/Pytest_Fold1_200K_chunk27.h5ad (500 cells sampled)
- shape of anndata from h5ad file: N x 19183, where N is the number of cells

### anndata example
```bash
AnnData object with n_obs × n_vars = 500 × 19183
    obs: 'soma_joinid', 'age'
    var: 'feature_id', 'feature_name'
```

## example
### example data and model
- example data can be found at "data/pytest_data" of this repository
- example GMA model file can be found at "data/trained_models/GMA_models" of this repository

### current supported model types
- $${\color{red}GMA\space(Gated \space Multi-head \space Attention \space Neural \space Networks, default \space and \space recommended)}$$
- MLP (Multilayer Perceptron)
- linear (Elastic Net based Linear regression model)
- xgboost 
- catboost

### making age prediction
```python
from scageclock.evaluation import prediction
model_file="./data/trained_models/GMA_models/GMA_celltype_balanced_basicRun.pth"
h5ad_folder="./data/pytest_data/train_val_test_mode/test/"
h5ad_feature_file="./data/metadata/h5ad_var.tsv"
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

### get model feature importance (GMA model)###
```python
from scageclock.scAgeClock import load_GMA_model, get_feature_importance
model_file = "./data/trained_models/GMA_models/GMA_celltype_balanced_basicRun.pth"
gma_model = load_GMA_model(model_file)
feature_file = "data/metadata/h5ad_var.tsv"
feature_importance = get_feature_importance(gma_model,feature_file=feature_file)
#sort by feature importance score
feature_importance = feature_importance.sort_values(by="feature_importance",ascending=False)
```

### model training with validation and testing
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
- Author: Gangcai Xie (Medical School of Nantong University); 
- [ORCID](https://orcid.org/0000-0002-8286-2987)
