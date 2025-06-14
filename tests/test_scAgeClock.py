import pytest
from ..scageclock.scAgeClock import training_pipeline
from scipy.stats import pearsonr

# ["linear", "xgboost", "catboost", "MLP", "GMA"]
@pytest.mark.parametrize(
    "model_name, validation_during_training,predict_dataset,loader_method,K_fold_mode",
    [
        ("GMA", False, "training", "scageclock", True),
        ("GMA", True, "training", "scageclock", True),
        ("linear",True,"validation","scageclock",True),
        ("MLP", True, "validation", "scageclock", True),
        ("linear", False, "validation", "scageclock", True),
        ("MLP", False, "validation", "scageclock", True),
        ("GMA",True,"validation","scageclock",False),
        ("linear",True,"validation","scageclock",False),
        ("xgboost",True,"validation","scageclock",False),
        ("catboost",True,"validation","scageclock",False),
        ("MLP",True,"validation","scageclock",False),

    ]
)

def test_training_pipeline(model_name,
                           validation_during_training,
                           predict_dataset,
                           loader_method,
                           K_fold_mode,
                           request):
    ad_dir_root = request.config.getoption("--test_data_dir")
    k_fold_data_dir = request.config.getoption("--k_fold_data_dir")
    out_root_dir = request.config.getoption("--out_root_dir")


    meta_file = request.config.getoption("--meta_file")
    if model_name in ["xgboost","catboost"]:
        dataset_folder_dict = {"training": "train", "validation": "val", "testing": "test"}
        results = training_pipeline(model_name=model_name,
                                    ad_dir_root=ad_dir_root,
                                    meta_file_path=meta_file,
                                    dataset_folder_dict=dataset_folder_dict,
                                    predict_dataset=predict_dataset,
                                    validation_during_training=validation_during_training,
                                    loader_method=loader_method,
                                    train_dataset_fully_loaded=True,
                                    out_root_dir=out_root_dir)
    else:
        if K_fold_mode:
            dataset_folder_dict = {"training_validation": "train_val"}
            results = training_pipeline(model_name=model_name,
                                        ad_dir_root=k_fold_data_dir,
                                        meta_file_path=meta_file,
                                        dataset_folder_dict=dataset_folder_dict,
                                        predict_dataset=predict_dataset,
                                        K_fold_mode=K_fold_mode,
                                        K_fold_train=("Fold1", "Fold2"),
                                        K_fold_val=("Fold3"),
                                        validation_during_training=validation_during_training,
                                        loader_method=loader_method,
                                        out_root_dir=out_root_dir)
        else:
            dataset_folder_dict = {"training": "train", "validation": "val", "testing": "test"}
            results = training_pipeline(model_name=model_name,
                                        ad_dir_root=ad_dir_root,
                                        meta_file_path=meta_file,
                                        dataset_folder_dict=dataset_folder_dict,
                                        predict_dataset=predict_dataset,
                                        K_fold_mode=K_fold_mode,
                                        validation_during_training=validation_during_training,
                                        loader_method=loader_method,
                                        out_root_dir=out_root_dir)
    assert results == True
