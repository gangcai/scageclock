import pytest
from ..scageclock.evaluation import prediction
import os

@pytest.mark.parametrize(
    "model_file_type",
    [("pth"),
    ]
)

def test_prediction(model_file_type,
                    request):
    ad_dir_root = request.config.getoption("--test_data_dir")
    model_file = request.config.getoption("--model_file")

    #model_file = "./data/trained_models/GMA_models/GMA_celltype_balanced_basicRun.pth"
    h5ad_dir = os.path.join(ad_dir_root,"test")
    results_df = prediction(model_file=model_file,
                            h5ad_dir=h5ad_dir,
                            model_file_type=model_file_type)
    assert results_df["cell_age_predicted"].shape[0] > 0
