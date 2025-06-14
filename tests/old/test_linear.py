import pytest
from ..scageclock.model.Linear import TorchElasticNetAgeClock
from scipy.stats import pearsonr

@pytest.mark.parametrize(
    "validation_during_training,predict_dataset,loader_method",
    [
        (True,"validation","scageclock"),
        (False,"validation","scageclock"),
        (False,"testing","scageclock"),
        (True,"validation","scageclock_balanced"),
    ]
)

def test_TorchElasticNetAgeClock(validation_during_training,
                         predict_dataset,
                         loader_method,
                         request):
    ad_dir_root = request.config.getoption("--test_data_dir")
    dataset_folder_dict = {"training": "train", "validation": "val", "testing": "test"}
    age_clock = TorchElasticNetAgeClock(anndata_dir_root=ad_dir_root,
                            dataset_folder_dict=dataset_folder_dict,
                            predict_dataset=predict_dataset,
                            validation_during_training=validation_during_training,
                            loader_method=loader_method)
    age_clock.train()
    predict_list = age_clock.predict()
    corr, p_value = pearsonr(predict_list[0], predict_list[1])
    test_metrics_df, y_test_pred, y_test, soma_ids_all = age_clock.cell_level_test()
    assert len(predict_list) == 3
    assert len(y_test_pred) > 0
