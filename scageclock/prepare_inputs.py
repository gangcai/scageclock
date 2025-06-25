import scanpy as sc
import os
import time
import pandas as pd
from scipy.sparse import csr_matrix, hstack, issparse
import numpy as np
import anndata
import glob

class InputsPrepare:
    def __init__(self,
                 h5ad_files_path,
                 meta_file,
                 gene_file,
                 outdir,
                 chunk_size: int = 50000,
                 split_tag: str = "train_val",
                 fold_id="Fold1", ##the fold id for k-fold cross-validation, not-used if split_tag == 'test'
                 under_sub_folders: bool = True,
                 backed_load: str | None = 'r',
                 gene_file_gene_column: str = "gene",
                 meta_file_split_first_column: str = "split_tag",
                 meta_file_split_second_column: str | None = "fold_id", ## only for train_val datasets
                 cat_cols =("assay", "cell_type", "tissue_general", "sex"),
                 cat_folder = "../meta_data/categorical_numeric_index/",
                 test_mode: bool = False,
                 test_mode_num: int = 5,
                 normalization_target_sum: int | None = None,
                 normalization_exclude_highly_expressed: bool = False,
                 normalization_max_fraction: float = 0.05):

        self.h5ad_files_path = h5ad_files_path
        self.meta_file = meta_file
        self.gene_file = gene_file
        self.outdir = outdir
        self.chunk_size = chunk_size
        self.split_tag = split_tag
        self.fold_id = fold_id
        self.under_sub_folders = under_sub_folders
        self.backed_load = backed_load
        self.gene_file_gene_column = gene_file_gene_column
        self.meta_file_split_first_column = meta_file_split_first_column
        self.meta_file_split_second_column = meta_file_split_second_column

        self.cat_cols = list(cat_cols)
        self.cat_folder = cat_folder
        self.test_mode = test_mode

        #for normalization setting
        self.normalization_target_sum = normalization_target_sum
        self.normalization_exclude_highly_expressed = normalization_exclude_highly_expressed
        self.normalization_max_fraction = normalization_max_fraction

        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)


        ## get all h5ad files
        if under_sub_folders:
            self.h5ad_files = glob.glob(os.path.join(self.h5ad_files_path, "*/*.h5ad"))
        else:
            self.h5ad_files = glob.glob(os.path.join(self.h5ad_files_path, "*.h5ad"))

        ## load the meta data
        self.meta_data = pd.read_parquet(self.meta_file)
        ## load all anndata objects into a list
        self.ad_loaded_lst = self.load_h5ad()
        if test_mode:
            ad_num = len(self.ad_loaded_lst)
            if ad_num > test_mode_num:
                self.ad_loaded_lst = self.ad_loaded_lst[:test_mode_num]

        ## load the selected genes
        self.gene_id_selected = self.get_genes()

        ## load the categorical data numeric file
        self.cat_dict = self.load_cat_numeric_dict()




    def get_genes(self):
        genes_df = pd.read_csv(self.gene_file,sep="\t") # with at least one column with gene id
        gene_id_selected = list(genes_df[self.gene_file_gene_column])
        return gene_id_selected


    def load_h5ad(self):

        ad_lst = []
        for h5ad_file in self.h5ad_files:
            ad = sc.read_h5ad(h5ad_file, backed=self.backed_load)
            ad_lst.append(ad)
        return ad_lst


    def split_h5ad(self, input_adata, chunk_size):
        num_cells = input_adata.shape[0]

        num_chunks = (num_cells + chunk_size - 1) // chunk_size

        adata_chunks = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, num_cells)
            adata_subset = input_adata[start_idx:end_idx, :]

            adata_chunks.append(adata_subset)
        print("check adata_chunks")
        print(adata_chunks[0])
        return adata_chunks

    ## add the categorical data to the anndata expression sparse matrix
    def format_and_write(self,
                         adata,
                         prefix,):

        obs_df = pd.merge(adata.obs, self.meta_data[["soma_joinid", "age"]], on="soma_joinid", how="left")

        adata.obs = obs_df[["soma_joinid", "assay", "sex", "tissue_general", "cell_type", "age"]]

        ## add categorical feature to the feature id and feature names
        feature_ids = self.cat_cols + list(adata.var['feature_id'])
        feature_names = self.cat_cols + list(adata.var['feature_name'])

        ### add categorical data (numeric) to the matrix
        cat_numeric_lst = []
        for cat in self.cat_cols:
            cat_values = adata.obs[cat]
            cat_values_numeric = [self.cat_dict[cat][x] for x in cat_values]
            cat_numeric_lst.append(cat_values_numeric)
        cat_numeric_np = np.array(cat_numeric_lst)
        X_merged = hstack([csr_matrix(cat_numeric_np.T), adata.X])
        del adata
        adata_m = anndata.AnnData(csr_matrix(X_merged))
        adata_m.obs = obs_df[["soma_joinid", "age"]]
        adata_m.obs_names = [str(i) for i in range(adata_m.n_obs)]  ## re-name the obs index

        var_df_new = pd.DataFrame({"feature_id": feature_ids,
                                   "feature_name": feature_names})
        adata_m.var = var_df_new
        adata_m.var_names = [str(i) for i in range(adata_m.n_vars)]  ## re-name the var index
        h5ad_file = os.path.join(self.outdir, f"{prefix}.h5ad")
        adata_m.write_h5ad(h5ad_file)
        return True


    def ad_normalize(self, adata):
        if not self.backed_load is None:
            adata = adata.to_memory()
        sc.pp.normalize_total(adata,
                              target_sum=self.normalization_target_sum,
                              exclude_highly_expressed=self.normalization_exclude_highly_expressed,
                              max_fraction=self.normalization_max_fraction) # TODO: optimize the normalization
        # Logarithmize the data
        sc.pp.log1p(adata)
        return adata


    def extract_h5ad(self):
        start_time = time.time()
        ad_lst = self.ad_loaded_lst
        gene_lst = self.gene_id_selected
        i = 0
        pre_ad = None
        chunk_id_num = 0
        # ad_var = ad_lst[0].var  ## all chunks should share the same .var information, after concat .var will be lost
        if not self.split_tag in ["train_val", "test"]:
            print(f"{self.split_tag} not supported, supported values: train_val, test")

        print(f"chunk size: {self.chunk_size}, split_tag: {self.split_tag}")

        if self.split_tag == "train_val":
            prefix1 = f"{self.split_tag}_{self.fold_id}"
        else:
            prefix1 = self.split_tag

        for ad in ad_lst:
            ad_var_df = ad.var
            i += 1
            print(f"start to process adata list: {i}")
            soma_ids = list(ad.obs["soma_joinid"])
            meta_data_s = self.meta_data[self.meta_data["soma_joinid"].isin(soma_ids)]
            if meta_data_s.shape[0] == 0:
                print(f"skipped for {i}, no filtered data")
                continue
            meta_data_s = meta_data_s[meta_data_s[self.meta_file_split_first_column] == self.split_tag]
            if self.split_tag == "train_val":
                meta_data_s = meta_data_s[meta_data_s[self.meta_file_split_second_column] == self.fold_id]

            soma_ids_f = meta_data_s["soma_joinid"]
            adata_f = ad[ad.obs["soma_joinid"].isin(list(soma_ids_f))]
            if adata_f.shape[0] == 0:
                print(f"skipped for {i}, no {self.split_tag} data")
                continue

            if pre_ad:  # not None
                adata_f = sc.concat([adata_f, pre_ad])
                adata_f.var = ad_var_df

            if adata_f.shape[0] > self.chunk_size:
                h5ad_lst = self.split_h5ad(adata_f, chunk_size=self.chunk_size)
                imcomplete_num = 0
                for h5ad_chunk in h5ad_lst:
                    if h5ad_chunk.shape[0] == self.chunk_size:
                        chunk_id_num += 1
                        print(f"write chunk{chunk_id_num} to file with complete chunk size:{h5ad_chunk.shape}")
                        h5ad_chunk = self.ad_normalize(h5ad_chunk)  ## normalize based on all the genes
                        h5ad_chunk_s = h5ad_chunk[:, h5ad_chunk.var["feature_id"].isin(gene_lst)]  ## filter genes
                        self.format_and_write(h5ad_chunk_s,
                                              prefix=f"{prefix1}_Chunk{chunk_id_num}",)
                    else:
                        pre_ad = h5ad_chunk
                        imcomplete_num += 1
                if imcomplete_num > 1:
                    print("error for the number of imcomplete chunks")
                    return False
                continue

            if adata_f.shape[0] < self.chunk_size:
                pre_ad = adata_f
                continue

            if adata_f.shape[0] == self.chunk_size:
                chunk_id_num += 1
                print(f"write chunk{chunk_id_num} to file with complete chunk size:{adata_f.shape}")
                adata_f = self.ad_normalize(adata_f)
                adata_f_s = adata_f[:, adata_f.var["feature_id"].isin(gene_lst)]
                self.format_and_write(adata_f_s,
                                      prefix=f"{prefix1}_Chunk{chunk_id_num}",)
                pre_ad = None
        # output the last chunk
        if not pre_ad:
            print("No extra last chunk anndata")
            return 0
        if pre_ad.shape[0] > self.chunk_size:
            print("Error for the last chunk")
            return -1
        chunk_id_num += 1
        print(f"output for the remained part after chunking with shape: {pre_ad.shape}")
        print(f"write chunk{chunk_id_num} to file")
        pre_ad = self.ad_normalize(pre_ad)
        print(pre_ad)
        pre_ad_s = pre_ad[:, pre_ad.var["feature_id"].isin(gene_lst)]
        self.format_and_write(pre_ad_s,
                              prefix=f"{prefix1}_Chunk{chunk_id_num}",)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"elapsed time: {elapsed_time:.2f} seconds")
        return True
    def load_cat_numeric_dict(self):
        cat_dict = {}
        for cat in self.cat_cols:
            df = pd.read_csv(
                f"{self.cat_folder}/{cat}_numeric_index.tsv",
                sep="\t")
            cat_dict[cat] = {}
            for idx, row in df.iterrows():
                n_idx = int(row["numeric_index"])
                cat_val = row["categorical_value"]
                cat_dict[cat][cat_val] = n_idx
        return cat_dict