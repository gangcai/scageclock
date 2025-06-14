import pytest
import torch
from ..scageclock.model.GatedMultiheadAttention import GatedMultiheadAttentionAgeClock
from scipy.stats import pearsonr

@pytest.mark.parametrize(
    "validation_during_training,predict_dataset,loader_method,K_fold_mode",
    [
        (True,"validation","scageclock",False),
        (False,"validation","scageclock",False),
        (False,"testing","scageclock",False),
        (True,"validation","scageclock_balanced",False),
        (False,"validation","scageclock_balanced",False),
        (True, "validation", "scageclock", True),
        (False, "validation", "scageclock", True),
    ]
)

def test_GMAAgeClock(validation_during_training,
                         predict_dataset,
                         loader_method,
                         request,
                         K_fold_mode):
    ad_dir_root = request.config.getoption("--test_data_dir")
    k_fold_data_dir = request.config.getoption("--k_fold_data_dir")
    meta_file = request.config.getoption("--meta_file")

    if loader_method == "scageclock_balanced":
        h5ad_files_meta_file = meta_file
        h5ad_files_folder_path = f"{ad_dir_root}/train"
        h5ad_files_index_file = f"{ad_dir_root}/index_train.parquet"

        balanced_dataloader_parameters = {"h5ad_files_folder_path": h5ad_files_folder_path,
                                          "h5ad_files_index_file": h5ad_files_index_file,
                                          "h5ad_files_meta_file": h5ad_files_meta_file,
                                          "meta_balanced_column": "cell_type",
                                          "batch_iter_max": 10000, }
    else:
        balanced_dataloader_parameters = None

    if torch.backends.mps.is_available():
        print("Mac mps is found, and device is set to be mps")
        device = 'mps'
    elif torch.cuda.is_available():
        print("Cuda is found, and device is set to be cuda")
        device = "cuda"
    else:
        print("Both of cuda and mps are not available, and the cpu is used instead")
        device = "cpu"

    if K_fold_mode:
        dataset_folder_dict = {"training_validation": "train_val"}
        age_clock = GatedMultiheadAttentionAgeClock(anndata_dir_root=k_fold_data_dir,
                                                    dataset_folder_dict=dataset_folder_dict,
                                                    predict_dataset=predict_dataset,
                                                    validation_during_training=validation_during_training,
                                                    loader_method=loader_method,
                                                    K_fold_mode=K_fold_mode,
                                                    K_fold_train=("Fold1", "Fold2"),
                                                    K_fold_val="Fold3",
                                                    device=device
                                                    )
    else:
        dataset_folder_dict = {"training": "train", "validation": "val", "testing": "test"}
        age_clock = GatedMultiheadAttentionAgeClock(anndata_dir_root=ad_dir_root,
                                                    dataset_folder_dict=dataset_folder_dict,
                                                    predict_dataset=predict_dataset,
                                                    validation_during_training=validation_during_training,
                                                    balanced_dataloader_parameters=balanced_dataloader_parameters,
                                                    loader_method=loader_method,
                                                    batch_size_train=64,
                                                    batch_size_val=64,
                                                    batch_size_test=64,
                                                    device=device)
    age_clock.train()
    predict_list = age_clock.predict()
    corr, p_value = pearsonr(predict_list[0], predict_list[1])
    feature_importances = age_clock.get_feature_importance()
    test_metrics_df, y_test_pred, y_test, soma_ids_all = age_clock.cell_level_test()
    assert len(predict_list) == 3
    assert len(feature_importances) > 0
    assert len(y_test_pred) > 0
