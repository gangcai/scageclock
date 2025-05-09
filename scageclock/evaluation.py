from .scAgeClock import load_GMA_model
import glob
import anndata
import scanpy as sc
import torch
import numpy as np
import pandas as pd

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
    age_diff = np.array(y_true) - np.array(y_predicted)

    cell_df = pd.DataFrame({"cell_id": list(adata.obs[cell_id_col]),
                            "cell_age_true": y_true,
                            "cell_age_predicted": y_predicted,
                            "cell_age_diff": age_diff})

    if output_file is not None:
        cell_df.to_excel(output_file)

    return cell_df

def calculate_metrics(df,
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

    eval_metrics_df = calculate_metrics(cell_df_new, group_id=group_col)

    eval_metrics_df = eval_metrics_df.sort_values(by=sort_by, ascending=ascending)

    return eval_metrics_df