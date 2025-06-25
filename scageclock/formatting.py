import pandas as pd
import scanpy as sc
import anndata as ad
import gc
from scipy.sparse import csr_matrix, hstack, issparse
import numpy as np


def update_anndata_with_missing_genes(adata, gene_list):
    # Step 1: Identify genes in gene_list that are not in adata.var_names
    existing_genes = set(adata.var_names)
    gene_list_not_exist = [gene for gene in gene_list if gene not in existing_genes]

    # If no missing genes, return early
    if not gene_list_not_exist:
        print("No missing genes to add.")
        return adata

    # Step 2: Create an empty sparse array for the missing genes
    n_missing_genes = len(gene_list_not_exist)
    n_cells = adata.n_obs

    # Create an empty sparse matrix (CSR format is efficient for row-wise concatenation)
    empty_sparse_matrix = csr_matrix((n_cells, n_missing_genes), dtype=adata.X.dtype)

    # Step 3: Create metadata for the new genes
    new_var_data = pd.DataFrame(
        index=gene_list_not_exist,
        data={
            'gene': gene_list_not_exist,
        }
    )

    # Step 4: Concatenate the empty sparse matrix to the original adata.X
    print(empty_sparse_matrix.shape)
    print(adata.X.shape)
    if issparse(adata.X):
        # Note: We need to transpose the empty matrix and use vstack
        new_X = hstack([adata.X, empty_sparse_matrix])
    else:
        # For dense matrices (not recommended for large datasets)
        empty_dense_matrix = np.zeros((n_missing_genes, n_cells), dtype=adata.X.dtype)
        new_X = np.hstack([adata.X, empty_dense_matrix])

    # Step 5: Update adata.var by concatenating with new_var_data
    new_var = pd.concat([adata.var, new_var_data])

    # Step 6: Ensure var_names matches the new order
    # adata.var_names = pd.Index(list(adata.var_names) + gene_list_not_exist)

    adata_new = ad.AnnData(new_X)
    adata_new.var = new_var
    adata_new.obs = adata.obs
    return adata_new

# cat_folder:
# /home/gangcai/database/public_db/CZCELLxGENE/whole_datasets/CZCELLxGENE_Human_All/normal/categorical_numeric_index
## TODO: need further checking
def format_anndata_single_category(adata_raw,
                                   model_genes,  ## genes used in the scageclock Model
                                   cat_folder,
                                   gene_column="feature_name",
                                   normalize: bool = True,
                                   assay: str = "10x 3' v3",
                                   sex: str = "male",
                                   tissue_general: str = "brain",
                                   cell_type="neuron",
                                   normalization_target_sum: int | None = None,
                                   normalization_exclude_highly_expressed: bool = False,
                                   normalization_max_fraction: float = 0.05
                                   ):
    cat_cols = ["assay", "cell_type", "tissue_general", "sex"]
    cat_dict = {}
    for cat in cat_cols:
        df = pd.read_csv(
            f"{cat_folder}/{cat}_numeric_index.tsv",
            sep="\t")
        cat_dict[cat] = {}
        for idx, row in df.iterrows():
            n_idx = int(row["numeric_index"])
            cat_val = row["categorical_value"]
            cat_dict[cat][cat_val] = n_idx

    n_cell = adata_raw.shape[0]


    assay_idx = cat_dict["assay"][assay]
    sex_idx = cat_dict["sex"][sex]
    tissue_idx = cat_dict["tissue_general"][tissue_general]
    cell_idx = cat_dict["cell_type"][cell_type]

    cat_df = pd.DataFrame({"assay": [assay_idx] * n_cell,
                           "cell_type": [cell_idx] * n_cell,
                           "tissue_general": [tissue_idx] * n_cell,
                           "sex": [sex_idx] * n_cell})

    # make adata_raw contain all model genes
    adata = update_anndata_with_missing_genes(adata_raw, model_genes)

    #adata_raw.var_names = adata_raw.var[gene_column]

    #adata = adata_raw

    obs_df = adata.obs

    if normalize:
        ## normalize the gene expression data based on all filtered genes
        # Normalizing to median total counts (default)
        sc.pp.normalize_total(adata,
                              target_sum=normalization_target_sum,
                              exclude_highly_expressed=normalization_exclude_highly_expressed,
                              max_fraction=normalization_max_fraction
                              )
        # Logarithmize the data
        sc.pp.log1p(adata)
        print(f"shape during normalization: {adata.shape}")

    ## filter by protein coding genes
    adata = adata[:, model_genes]

    print(f"shape after protein coding selection: {adata.shape}")

    # merge categorical features and gene expression features
    X_merged = hstack([csr_matrix(cat_df), adata.X])

    del adata  # free the memory
    adata_m = ad.AnnData(csr_matrix(X_merged))
    adata_m.obs = obs_df
    adata_m.obs.index = adata_m.obs.index.astype(str)  ## convert .obs.index to str type
    del X_merged
    gc.collect()
    print(f"shape after adding categorical data: {adata_m.shape}")
    return adata_m

def format_anndata_multiple(adata_raw, # gene_name should be in adata_raw.var_names
                            model_genes,  ## genes used in the scageclock Model
                            normalize: bool = True,
                            cat_cols: None | list[str] = None,
                            normalization_target_sum: int | None = None,
                            normalization_exclude_highly_expressed: bool = False,
                            normalization_max_fraction: float = 0.05
                            ):

    if cat_cols is None:
        cat_cols = ["assay_index", "cell_type_index", "tissue_index", "sex_index"]

    cat_df = adata_raw.obs[cat_cols]
    # make adata_raw contain all model genes
    adata = update_anndata_with_missing_genes(adata_raw, model_genes)

    #adata_raw.var_names = adata_raw.var[gene_column]

    #adata = adata_raw

    obs_df = adata.obs

    if normalize:
        ## normalize the gene expression data based on all filtered genes
        # Normalizing to median total counts
        sc.pp.normalize_total(adata,
                              target_sum=normalization_target_sum,
                              exclude_highly_expressed=normalization_exclude_highly_expressed,
                              max_fraction=normalization_max_fraction
                              )
        # Logarithmize the data
        sc.pp.log1p(adata)
        print(f"shape during normalization: {adata.shape}")

    ## filter by protein coding genes
    adata = adata[:, model_genes]

    print(f"shape after protein coding selection: {adata.shape}")

    # merge categorical features and gene expression features
    X_merged = hstack([csr_matrix(cat_df), adata.X])

    del adata  # free the memory
    adata_m = ad.AnnData(csr_matrix(X_merged))
    adata_m.obs = obs_df
    adata_m.obs.index = adata_m.obs.index.astype(str)  ## convert .obs.index to str type
    del X_merged
    gc.collect()
    print(f"shape after adding categorical data: {adata_m.shape}")
    return adata_m

## match name with the model categorical index
def match_categorical_features(name_match_dict, model_cat_dict, cat="assay"):
    original_name_lst = []
    model_name_lst = []
    model_index_lst = []
    for original_name in name_match_dict.keys():
        matched_name = name_match_dict[original_name]
        #print([original_name,matched_name]) # for debug
        df = model_cat_dict[cat]
        matched_index = list(df[df["categorical_value"]==matched_name]['numeric_index'])[0]
        original_name_lst.append(original_name)
        model_name_lst.append(matched_name)
        model_index_lst.append(matched_index)
    match_df = pd.DataFrame({"original_cat_name":original_name_lst,
                              "model_cat_name":model_name_lst,
                              "model_cat_index":model_index_lst})
    return match_df