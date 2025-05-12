import pytest
from ..scageclock.model.XGBoost import XGBoostAgeClock
from scipy.stats import pearsonr

@pytest.mark.parametrize(
    "validation_during_training, train_dataset_fully_loaded,predict_dataset_fully_loaded,validation_dataset_fully_loaded,predict_dataset",
    [
        (True,True,True,True,"validation"),
        (False,True,True,True,"validation"),
        (True,False,False,False,"validation"),
        (False,True,True,True,"validation"),
        (False,True,True,True,"testing")
    ]
)

def test_XGBoostAgeClock(validation_during_training,
                         train_dataset_fully_loaded,
                         predict_dataset_fully_loaded,
                         validation_dataset_fully_loaded,
                         predict_dataset,
                         request):
    ad_dir_root = request.config.getoption("--test_data_dir")
    dataset_folder_dict = {"training": "train", "validation": "val", "testing": "test"}
    age_clock = XGBoostAgeClock(anndata_dir_root=ad_dir_root,
                                dataset_folder_dict=dataset_folder_dict,
                                predict_dataset=predict_dataset,
                                validation_during_training=validation_during_training,
                                train_dataset_fully_loaded=train_dataset_fully_loaded,
                                predict_dataset_fully_loaded=predict_dataset_fully_loaded,
                                validation_dataset_fully_loaded=validation_dataset_fully_loaded)
    age_clock.train()
    predict_list = age_clock.predict()
    corr, p_value = pearsonr(predict_list[0], predict_list[1])
    feature_importances = age_clock.get_feature_importance()
    test_metrics_df, y_test_pred, y_test, soma_ids_all = age_clock.cell_level_test()
    assert len(predict_list) == 3
    assert len(feature_importances) > 0
    assert len(y_test_pred) > 0
