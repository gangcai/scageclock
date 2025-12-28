import random

from torch.xpu import device

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
from tqdm import tqdm


def prediction(model_file: str,
               model_file_type: str = "pth",
               h5ad_dir: str | None = None,
               adata: anndata.AnnData | None = None,
               ad_file: str | None = None,
               age_col: str = "age",  # age column name in adata.obs
               cell_id_col: str = "soma_joinid",
               output_file: str | None = None,
               cat_cardinalities: list[int] | None = None,
               num_numeric_features: int = 19234,
               projection_dim=512,
               prediction_hidden_sizes: list[int] | None = None,
               num_heads: int = 8,
               ):
    # default cardinalities for each categorical feature column
    if prediction_hidden_sizes is None:
        prediction_hidden_sizes = [256, 128]

    if cat_cardinalities is None:
        # ['assay', 'cell_type', 'tissue_general', 'sex'],
        # the cardinalities for each categorical feature column, the first len(cat_car_list) columns
        cat_cardinalities = [21, 664, 52, 3]

    model = load_GMA_model(model_file=model_file,
                           model_file_type=model_file_type,
                           cat_cardinalities=cat_cardinalities,
                           num_numeric_features=num_numeric_features,
                           prediction_hidden_sizes=prediction_hidden_sizes,
                           projection_dim=projection_dim,
                           num_heads=num_heads,)
    model.eval() # don't miss this, otherwise the prediction will be different


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

    with torch.no_grad():
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

def prediction_with_Monte_Carlo_dropout(model_file: str,
               model_file_type: str = "pth",
               h5ad_dir: str | None = None,
               adata: anndata.AnnData | None = None,
               ad_file: str | None = None,
               age_col: str = "age",
               cell_id_col: str = "soma_joinid",
               output_file: str | None = None,
               cat_cardinalities: list[int] | None = None,
               num_numeric_features: int = 19234,
               projection_dim=512,
               prediction_hidden_sizes: list[int] | None = None,
               l1_lambda: float = 0.1,
               l2_lambda: float = 0.5,
               num_heads: int = 8,
               dropout_prob: float = 0.2,
               n_mc: int = 50,
            ):
    # default cardinalities for each categorical feature column
    if prediction_hidden_sizes is None:
        prediction_hidden_sizes = [256, 128]

    if cat_cardinalities is None:
        # ['assay', 'cell_type', 'tissue_general', 'sex'],
        # the cardinalities for each categorical feature column, the first len(cat_car_list) columns
        cat_cardinalities = [21, 664, 52, 3]

    model = load_GMA_model(model_file=model_file,
                           model_file_type=model_file_type,
                           cat_cardinalities=cat_cardinalities,
                           num_numeric_features=num_numeric_features,
                           prediction_hidden_sizes=prediction_hidden_sizes,
                           projection_dim=projection_dim,
                           l1_lambda=l1_lambda,
                           l2_lambda=l2_lambda,
                           num_heads=num_heads,
                           dropout_prob=dropout_prob,)

    # Enable dropout at inference
    model.train()

    # Load AnnData
    if h5ad_dir is not None:
        h5ad_files = glob.glob(f"{h5ad_dir}/*.h5ad")
        ad_list = [sc.read_h5ad(f) for f in h5ad_files]
        adata = anndata.concat(ad_list)
    elif ad_file is not None:
        adata = sc.read_h5ad(ad_file)
    elif adata is None:
        raise ValueError("Inputs error")

    # Prepare inputs
    X_inputs = adata.X.toarray()
    X_inputs_tensor = torch.from_numpy(X_inputs).to(torch.float32)

    y_true = np.array(adata.obs[age_col])
    cell_ids = list(adata.obs[cell_id_col])

    dfs = []

    with torch.no_grad():
        for mc_index in tqdm(range(n_mc), desc="Monte Carlo Dropout"):
            y_predicted = model(X_inputs_tensor).flatten().cpu().numpy()
            age_diff = np.array(y_predicted) - np.array(y_true)

            cell_df = pd.DataFrame({
                "cell_id": cell_ids,
                "cell_age_true": y_true,
                "cell_age_predicted": y_predicted,
                "cell_age_diff": age_diff,
                "Monte_Carlo_index": mc_index
            })

            dfs.append(cell_df)

    # Concatenate all MC runs
    cell_df = pd.concat(dfs, ignore_index=True)

    if output_file is not None:
        cell_df.to_excel(output_file, index=False)

    return cell_df


def prediction_with_Monte_Carlo_dropout_chunks(model_file: str,
               model_file_type: str = "pth",
               h5ad_dir: str | None = None,
               age_col: str = "age",
               cell_id_col: str = "soma_joinid",
               output_dir: str = "GMA_Monte_Carlo_Prediction_Out",
               output_prefix: str = "scageclock_monte_carlo",
               cat_cardinalities: list[int] | None = None,
               num_numeric_features: int = 19234,
               projection_dim=512,
               prediction_hidden_sizes: list[int] | None = None,
               l1_lambda: float = 0.1,
               l2_lambda: float = 0.5,
               num_heads: int = 8,
               dropout_prob: float = 0.2,
               n_mc: int = 50,
               base_seed: int = 42,
               mc_index_start: int | None = None,
            ):
    # default cardinalities for each categorical feature column
    if prediction_hidden_sizes is None:
        prediction_hidden_sizes = [256, 128]

    if cat_cardinalities is None:
        # ['assay', 'cell_type', 'tissue_general', 'sex'],
        # the cardinalities for each categorical feature column, the first len(cat_car_list) columns
        cat_cardinalities = [21, 664, 52, 3]

    model = load_GMA_model(model_file=model_file,
                           model_file_type=model_file_type,
                           cat_cardinalities=cat_cardinalities,
                           num_numeric_features=num_numeric_features,
                           prediction_hidden_sizes=prediction_hidden_sizes,
                           projection_dim=projection_dim,
                           l1_lambda=l1_lambda,
                           l2_lambda=l2_lambda,
                           num_heads=num_heads,
                           dropout_prob=dropout_prob,)

    # Enable dropout at inference
    model.train()

    # Load AnnData
    if h5ad_dir is not None:
        h5ad_files = glob.glob(f"{h5ad_dir}/*.h5ad")
        # ad_list = [sc.read_h5ad(f) for f in h5ad_files]
        #adata = anndata.concat(ad_list)
    else:
        raise ValueError("Inputs error")

    device = next(model.parameters()).device

    if not os.path.exists(output_dir):
        print(f"Creating output directory {output_dir}")
        os.makedirs(output_dir)
    else:
        print(f"Output directory {output_dir} already exists")

    with torch.no_grad():
        for mc_index in tqdm(range(n_mc), desc="Monte Carlo Dropout"):

            if mc_index_start is not None:
                if mc_index < mc_index_start:
                    print(f"Skipping mc_index: {mc_index}")
                    continue

            ## set seed for each Monte Carlo interation, to ensure the same dropout masked neural networks for each chunk file
            seed = base_seed + mc_index
            torch.manual_seed(seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            np.random.seed(seed)
            random.seed(seed)
            dfs = []
            for h5ad_file in tqdm(h5ad_files, desc="H5AD Files"):
                adata = sc.read_h5ad(h5ad_file)
                # Prepare inputs
                X_inputs = adata.X.toarray()
                X_inputs_tensor = torch.from_numpy(X_inputs).to(torch.float32)

                y_true = np.array(adata.obs[age_col])
                cell_ids = list(adata.obs[cell_id_col])
                y_predicted = model(X_inputs_tensor).flatten().cpu().numpy()
                age_diff = np.array(y_predicted) - np.array(y_true)

                cell_df = pd.DataFrame({
                    "cell_id": cell_ids,
                    "cell_age_true": y_true,
                    "cell_age_predicted": y_predicted,
                    "cell_age_diff": age_diff,
                    "Monte_Carlo_index": mc_index
                })

                dfs.append(cell_df)

                # memory clean up
                del adata, X_inputs_tensor, X_inputs
                torch.cuda.empty_cache() if device.type == 'cuda' else None

            ## concatenate for each MC iteration
            cell_df = pd.concat(dfs, ignore_index=True)
            output_file = os.path.join(output_dir, f"{output_prefix}_Iter{mc_index}.csv")
            cell_df.to_csv(output_file, index=False)

    return True


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
