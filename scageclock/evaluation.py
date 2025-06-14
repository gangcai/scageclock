from .scAgeClock import load_GMA_model
import glob
import anndata
import scanpy as sc
import torch
import numpy as np
import pandas as pd
from .utility import get_validation_metrics
import os
import re
from .utility import donor_level_test


def prediction(model_file: str,
               model_file_type: str = "pth",
               h5ad_dir: str | None = None,
               adata: anndata.AnnData | None = None,
               ad_file: str | None = None,
               age_col: str = "age",  # age column name in adata.obs
               cell_id_col: str = "soma_joinid",
               output_file: str | None = None,
               ):
    model = load_GMA_model(model_file=model_file, model_file_type=model_file_type)

    if h5ad_dir is not None:
        h5ad_files = glob.glob(f"{h5ad_dir}/*.h5ad")
        ad_list = []
        for h5ad_file in h5ad_files:
            ad_each = sc.read_h5ad(h5ad_file)
            ad_list.append(ad_each)
        adata = anndata.concat(ad_list)
    elif ad_file is not None:
        adata = sc.read_h5ad(ad_file)
    else:
        if adata is None:
            raise ValueError("Inputs error")

    X_inputs = adata.X.toarray()
    X_inputs_tensor = torch.from_numpy(X_inputs)
    X_inputs_tensor = X_inputs_tensor.to(torch.float32)
    y_predicted = model(X_inputs_tensor)
    y_predicted = y_predicted.flatten().detach()
    y_true = list(adata.obs[age_col])
    y_predicted = list(np.array(y_predicted))
    age_diff = np.array(y_predicted) - np.array(y_true)

    cell_df = pd.DataFrame({"cell_id": list(adata.obs[cell_id_col]),
                            "cell_age_true": y_true,
                            "cell_age_predicted": y_predicted,
                            "cell_age_diff": age_diff})

    if output_file is not None:
        cell_df.to_excel(output_file)

    return cell_df

def calculate_group_metrics(df,
                      group_id="cell_type",
                      cell_true_age_col: str = "cell_age_true",
                      cell_predicted_age_col: str = "cell_age_predicted"):
    metrics = {}
    for cell_type, group in df.groupby(group_id):
        if group.shape[0] == 0:
            continue
        correlation = group[cell_true_age_col].corr(group[cell_predicted_age_col])
        mae = np.mean(np.abs(group[cell_true_age_col] - group[cell_predicted_age_col]))
        metrics[cell_type] = {'Correlation': correlation, 'MAE': mae}

    metrics_df = pd.DataFrame.from_dict(metrics, orient='index').reset_index()
    metrics_df.columns = [group_id, 'Correlation', 'MAE']
    return metrics_df

def calculate_metrics(df,
                      cell_true_age_col: str = "cell_age_true",
                      cell_predicted_age_col: str = "cell_age_predicted"):

    metrics_dict = get_validation_metrics(df[cell_true_age_col], df[cell_predicted_age_col])
    return metrics_dict

def group_eval(cell_df,
               meta_data_file,
               group_col: str = "cell_type",
               sort_by: str = "MAE",
               ascending: bool = True,
               cell_df_id: str = "cell_id",
               meta_data_id: str = "soma_joinid"):
    meta_df = pd.read_parquet(meta_data_file)
    cell_df_new = pd.merge(cell_df, meta_df,
                           left_on=cell_df_id, right_on=meta_data_id, how="left")

    eval_metrics_df = calculate_group_metrics(cell_df_new, group_id=group_col)

    eval_metrics_df = eval_metrics_df.sort_values(by=sort_by, ascending=ascending)

    return eval_metrics_df


def multi_models_evaluation(model_path: str,
                            eval_h5ad_folder_path: str,
                            eval_meta_file_path: str,
                            cell_id_column: str = "soma_joinid",
                            donor_id_column: str = "donor_id_general",
                            model_file_type: str = "pth",):
    if model_file_type == "pth":
        pth_files = glob.glob(os.path.join(model_path, "*.pth"))
        runtype2metrics = {}
        donor2metrics = {}
        for pth in pth_files:
            filename = pth.split("/")[-1]
            prefix = re.sub(".pth", "", filename)
            cell_df = prediction(model_file=pth,
                                 h5ad_dir=eval_h5ad_folder_path)
            runtype2metrics[prefix] = calculate_metrics(cell_df)

            donor_true_age, donor_pre_age, donor_level_test_metrics_dict = donor_level_test(meta_file_path=eval_meta_file_path,
                                                                                            cell_id_column=cell_id_column,
                                                                                            donor_id_column=donor_id_column,
                                                                                            test_soma_joinids=list(cell_df["cell_id"]),
                                                                                            y_test_true=list(cell_df["cell_age_true"]),
                                                                                            y_test_predict=list(cell_df["cell_age_predicted"]))
            donor2metrics[prefix] = donor_level_test_metrics_dict
        cell_metrics_df = pd.DataFrame.from_dict(runtype2metrics, orient='index').reset_index()
        cell_metrics_df = cell_metrics_df.sort_values(by="MAE", ascending=True)
        donor_metrics_df = pd.DataFrame.from_dict(donor2metrics, orient='index').reset_index()
        donor_metrics_df = donor_metrics_df.sort_values(by="MAE", ascending=True)
        return cell_metrics_df, donor_metrics_df
    else:
        raise ValueError("Currently only model_file_type pth is supported!")
