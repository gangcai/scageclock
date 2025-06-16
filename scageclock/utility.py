from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import scipy
import os
import scanpy as sc
import pandas as pd
from typing import List, Literal
import glob
import anndata

from .formatting import format_anndata_single_category as format_anndata

from scipy.stats import pearsonr
from scipy.sparse import issparse
#from ...scripts.disease_study.convert_disease_inputs import adata_formated


## get the validation metrics based on the true donor age and the predicted donor age
## TODO: add other method, such as squared mean
def donor_level_test(meta_file_path: str,
                     test_soma_joinids: List,
                     y_test_true: List,
                     y_test_predict: List,
                     cell_id_column: str = "soma_joinid",
                     donor_id_column: str = "donor_id_general",
                     method: str = "mean"):

    if not method in ["mean","median"]:
        print("Error: method can only be one of ['mean', 'median']")
        return False

    meta_df = pd.read_parquet(meta_file_path) # should contain at least two clumns: "soma_joinid" and "donor_id_general"
    test_donor_ids = []
    for test_soma_id in test_soma_joinids:
        test_soma_id = int(test_soma_id)
        donor_id_general = meta_df[meta_df[cell_id_column] == int(test_soma_id)][donor_id_column].values[0]
        test_donor_ids.append(donor_id_general)

    test_donor_df = pd.DataFrame({"donor": test_donor_ids,
                                  "test_true": y_test_true,
                                  "test_pre": y_test_predict})

    ## use donor cell-level mean age as the donor age
    donor_true_age = None
    donor_pre_age = None
    if method == "mean":
        donor_true_age = test_donor_df.groupby('donor')['test_true'].mean()
        donor_pre_age = test_donor_df.groupby('donor')['test_pre'].mean()
    elif method == "median":
        donor_true_age = test_donor_df.groupby('donor')['test_true'].median()
        donor_pre_age = test_donor_df.groupby('donor')['test_pre'].median()
    test_metrics_dict = get_validation_metrics(y_true=donor_true_age,
                                             y_pred=donor_pre_age,
                                             print_metrics=False)
    return donor_true_age, donor_pre_age, test_metrics_dict

def get_validation_metrics(y_true,
                           y_pred, 
                           print_metrics=False):
    """
    Get the validation metrics for aging prediction
    :param y_true: List of true age values
    :param y_pred: List of model predicted age values
    :param print_metrics: whether to print out the validation metrics
    :return: dictionary with the metrics values
    """
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true=y_true, y_pred=y_pred)
    correlation, p_value = scipy.stats.pearsonr(y_pred, y_true)

    rmse = float(rmse)
    correlation = float(correlation)
    p_value = float(p_value)

    if print_metrics:
        print(f"Mean Squared Error (MSE): {mse}")
        print(f"Root Mean Squared Error (RMSE): {rmse}")
        print(f"Mean Absolute Error (MAE): {mae}")
        print(f"R^2 Score: {r2}")
        print("Pearson correlation coefficient:", correlation)
        print("P-value:", p_value)

    results = {"MSE":mse,
              "RMSE":rmse,
              "MAE":mae,
              "R2_Score":r2,
              "Pearson's r": correlation,
              "Pearson P value": p_value}
    return results

## too much memory needed for large .h5ad datasets
def concatenate_h5ad_files(h5ad_path: str):

    # Initialize an empty AnnData object
    adata = None
    is_first = True
    # Iterate through the directory and load each .h5ad file
    for filename in os.listdir(h5ad_path):
        if filename.endswith(".h5ad"):
            filepath = os.path.join(h5ad_path, filename)
            adata_temp = sc.read_h5ad(filepath)
            if is_first:
                adata = adata_temp
                is_first = False
            else:
                adata = adata.concatenate(adata_temp)
    return adata





def check_model_device(model):
    # Get the device of the first parameter
    device = next(model.parameters()).device

    # Check if the model is on CUDA
    if device.type == 'cuda':
        return "cuda"
    elif device.type == 'cpu':
        return "cpu"
    else:
        print(f"Model is on an unknown device: {device}")
        return False


def check_tensor_device(inputs):
    if inputs.is_cuda:
        return "cuda"
    elif inputs.is_cpu:
        return "cpu"
    else:
        return False


### TODO: delete extract function for h5ad files, moved to h5ad_sampler
## extract for a given cell type anndata from a list of h5ad files
## Those .h5ad files should contain the similar structure of .obs and .var
def extract_cell_type_data_from_h5ad_files(meta_df,
                                           h5ad_file_path,
                                           cell_type: str = "retinal rod cell",
                                           cell_type_colname: str = "cell_type",
                                           cell_id: str = "soma_joinid",
                                           backed: Literal["r", "r+"] | bool | None = None,
                                           sample_num: int | None = None):
    #meta_df = pd.read_parquet(parquet_meta_file)
    meta_df_s = meta_df[meta_df[cell_type_colname] == cell_type]
    if sample_num:
        meta_df_s = meta_df_s.sample(sample_num)

    if meta_df_s.empty:
        raise ValueError(f"{cell_type} not found in the meta data")
    cell_ids = list(meta_df_s[cell_id])
    ad_files = glob.glob(os.path.join(h5ad_file_path, "*.h5ad"))
    ad_list = []
    for ad_file in ad_files:
        adata = sc.read_h5ad(ad_file,backed=backed)

        adata_s = adata[adata.obs[cell_id].isin(cell_ids)]

        if adata_s.shape[0] > 0:
            ad_list.append(adata_s)
    ad_join = anndata.concat(ad_list)
    ad_join.var_names_make_unique()
    return ad_join


def extract_cell_types_from_h5ad_files(meta_df,
                                           h5ad_file_path_root,
                                           sub_folders = ["train","val","test"],
                                           cell_type_list = ["retinal rod cell"],
                                           cell_type_colname: str = "cell_type",
                                           cell_id: str = "soma_joinid",
                                           backed: Literal["r", "r+"] | bool | None = None,
                                           sample_num: int | None = None):
    #meta_df = pd.read_parquet(parquet_meta_file)
    meta_df_s = meta_df[meta_df[cell_type_colname].isin(cell_type_list)]
    if sample_num:
        meta_df_s = meta_df_s.sample(sample_num)

    if meta_df_s.empty:
        raise ValueError(f"{cell_type_list} not found in the meta data")
    cell_ids = list(meta_df_s[cell_id])
    ad_files = []
    for sub_folder in sub_folders:
        ad_files += glob.glob(os.path.join(h5ad_file_path_root, sub_folder,"*.h5ad"))
    ad_list = []
    for ad_file in ad_files:
        adata = sc.read_h5ad(ad_file,backed=backed)

        adata_s = adata[adata.obs[cell_id].isin(cell_ids)]

        if adata_s.shape[0] > 0:
            ad_list.append(adata_s)
    ad_join = anndata.concat(ad_list)
    ad_join.var_names_make_unique()
    return ad_join

def extract_tissue_from_h5ad_files(meta_df,
                                   h5ad_file_path_root,
                                   sub_folders = ["train"],
                                   tissue_list = ["breast"],
                                   tissue_colname: str = "tissue_general",
                                   cell_id: str = "soma_joinid",
                                   backed: Literal["r", "r+"] | bool | None = None,
                                   sample_num: int | None = None):
    #meta_df = pd.read_parquet(parquet_meta_file)
    meta_df_s = meta_df[meta_df[tissue_colname].isin(tissue_list)]
    if sample_num:
        if meta_df_s.shape[0] > sample_num:
            meta_df_s = meta_df_s.sample(sample_num)

    if meta_df_s.empty:
        raise ValueError(f"{tissue_list} not found in the meta data")
    cell_ids = list(meta_df_s[cell_id])
    ad_files = []
    for sub_folder in sub_folders:
        ad_files += glob.glob(os.path.join(h5ad_file_path_root, sub_folder,"*.h5ad"))
    ad_list = []
    for ad_file in ad_files:
        adata = sc.read_h5ad(ad_file,backed=backed)

        adata_s = adata[adata.obs[cell_id].isin(cell_ids)]

        if adata_s.shape[0] > 0:
            ad_list.append(adata_s)
    ad_join = anndata.concat(ad_list)
    ad_join.var_names_make_unique()
    return ad_join

def sample_cells(meta_df,
                 h5ad_file_path_root,
                 sub_folders=["train", "val", "test"],
                 cell_id: str = "soma_joinid",
                 backed: Literal["r", "r+"] | bool | None = None,
                 sample_num: int =10000):
    if meta_df.shape[0] > sample_num:
        meta_df_s = meta_df.sample(sample_num)
    else:
        print(f"sample_num {sample_num} is larger than the meta_df {meta_df.shape[0]}, the whole dataset is used instead")
        meta_df_s = meta_df

    cell_ids = list(meta_df_s[cell_id])
    ad_files = []
    for sub_folder in sub_folders:
        ad_files += glob.glob(os.path.join(h5ad_file_path_root, sub_folder,"*.h5ad"))
    ad_list = []
    for ad_file in ad_files:
        adata = sc.read_h5ad(ad_file,backed=backed)

        adata_s = adata[adata.obs[cell_id].isin(cell_ids)]

        if adata_s.shape[0] > 0:
            ad_list.append(adata_s)
    ad_join = anndata.concat(ad_list)
    ad_join.var_names_make_unique()
    return ad_join

def extract_individual_cell_type_from_h5ad_files(meta_df,
                                           h5ad_file_path_root,
                                           sub_folders = ["test"],
                                           cell_type_list = ["macrophage"],
                                           cell_type_colname: str = "cell_type",
                                           donor_id: str = "TSP1",
                                           donor_id_column: str = "donor_id_general",
                                           cell_id: str = "soma_joinid",
                                           backed: Literal["r", "r+"] | bool | None = None,
                                           sample_num: int | None = None):
    meta_df_s = meta_df[meta_df[cell_type_colname].isin(cell_type_list)]
    meta_df_s = meta_df_s[meta_df[donor_id_column] == donor_id]

    if sample_num:
        meta_df_s = meta_df_s.sample(sample_num)

    if meta_df_s.empty:
        raise ValueError(f"{cell_type_list} for {donor_id} not found in the meta data")
    cell_ids = list(meta_df_s[cell_id])
    ad_files = []
    for sub_folder in sub_folders:
        ad_files += glob.glob(os.path.join(h5ad_file_path_root, sub_folder,"*.h5ad"))
    ad_list = []
    for ad_file in ad_files:
        adata = sc.read_h5ad(ad_file,backed=backed)

        adata_s = adata[adata.obs[cell_id].isin(cell_ids)]

        if adata_s.shape[0] > 0:
            ad_list.append(adata_s)
    ad_join = anndata.concat(ad_list)
    ad_join.var_names_make_unique()
    return ad_join


## format the given single cell expression matrix into the scageclock model inputs
def anndata_formatting(adata,
                                model_genes,  ## genes used in the scageclock Model
                                cat_folder,
                                gene_column="feature_name",
                                normalize: bool = True,
                                assay: str = "10x 3' v3",
                                sex: str = "male",
                                tissue_general: str = "brain",
                                cell_type="neuron"
                                ):
    adata_f = format_anndata(adata_raw=adata,
                             model_genes=model_genes,
                             cat_folder=cat_folder,
                             gene_column=gene_column,
                             normalize=normalize,
                             assay=assay,
                             sex=sex,
                             tissue_general=tissue_general,
                             cell_type=cell_type)

    return adata_f

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# calculate cell-level Aging Deviation Index (ADI)
def get_ADI(age_true,
            age_predicted,
            norm_size: int = 25):
    age_true = np.array(age_true)
    age_predicted = np.array(age_predicted)
    age_diff = age_predicted - age_true
    ADI = sigmoid(age_diff/norm_size)
    return ADI




def calculate_ADI_correlation(ad, score_column='ADI'):
    """
    Calculate Pearson correlations and p-values between gene expression and ADI score in AnnData.

    Parameters:
    -----------
    ad : anndata.AnnData
        AnnData object containing single-cell RNA sequencing data.
    score_column : str, optional (default: 'score')
        Name of the column in `ad.obs` containing the score variable.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with columns 'gene', 'pearson_correlation', and 'p_value',
        sorted by absolute correlation.
    """
    # Extract the score variable from ad.obs
    score = ad.obs[score_column].values

    # Initialize lists to store results
    correlations = []
    p_values = []

    # Handle sparse matrix (convert to dense if needed)
    if issparse(ad.X):
        X_dense = ad.X.toarray()
    else:
        X_dense = ad.X

    # Iterate over each gene
    for gene_idx in range(X_dense.shape[1]):
        gene_expression = X_dense[:, gene_idx]

        corr, p_val = pearsonr(list(gene_expression), list(score))
        correlations.append(corr)
        p_values.append(p_val)

    # Create DataFrame with results
    result_df = pd.DataFrame({
        'gene': list(ad.var_names),
        'pearson_correlation': correlations,
        'p_value': p_values
    })

    result_df = result_df[~result_df["pearson_correlation"].isna()]

    # Sort by absolute correlation (descending)
    result_df = result_df.iloc[np.argsort(result_df['pearson_correlation'])[::-1]]

    return result_df