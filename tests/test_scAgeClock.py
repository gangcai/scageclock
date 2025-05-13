import pytest
from ..scageclock.scAgeClock import training_pipeline
from scipy.stats import pearsonr

# ["linear", "xgboost", "catboost", "MLP", "GMA"]
@pytest.mark.parametrize(
    "model_name, validation_during_training,predict_dataset,loader_method",
    [
        ("GMA",True,"validation","scageclock"),
        ("linear",True,"validation","scageclock"),
        ("xgboost",True,"validation","scageclock"),
        ("catboost",True,"validation","scageclock"),
        ("MLP",True,"validation","scageclock"),
    ]
)

def test_training_pipeline(model_name,
                           validation_during_training,
                           predict_dataset,
                           loader_method,
                           request):
    ad_dir_root = request.config.getoption("--test_data_dir")
    meta_file = request.config.getoption("--meta_file")
    dataset_folder_dict = {"training": "train", "validation": "val", "testing": "test"}
    if model_name in ["xgboost","catboost"]:
        results = training_pipeline(model_name=model_name,
                                    ad_dir_root=ad_dir_root,
                                    meta_file_path=meta_file,
                                    dataset_folder_dict=dataset_folder_dict,
                                    predict_dataset=predict_dataset,
                                    validation_during_training=validation_during_training,
                                    loader_method=loader_method,
                                    train_dataset_fully_loaded=True, )
    else:
        results = training_pipeline(model_name=model_name,
                                    ad_dir_root=ad_dir_root,
                                    meta_file_path=meta_file,
                                    dataset_folder_dict=dataset_folder_dict,
                                    predict_dataset=predict_dataset,
                                    validation_during_training=validation_during_training,
                                    loader_method=loader_method)
    assert results == True
