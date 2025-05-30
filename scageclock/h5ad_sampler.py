import glob
import os
import pandas as pd
from typing import Literal
import scanpy as sc
import anndata

class H5ADSampler:
    def __init__(self,
                 h5ad_file_path: str,
                 h5ad_meta_file: str,
                 meta_file_format: str = "parquet",
                 split_column: str | None = "split_tag",
                 split_category_choose: str = "train", # train, test, val ; ignore if split_column is None
                 meta_cell_id_colname: str = "soma_joinid",
                 anndata_cell_id_colname: str = "soma_joinid", # in anndata.obs
                 backed: Literal["r", "r+"] | bool | None = 'r',
                 meta_category_colname: str = "tissue_general", # or cell_type
                 category_list: list | None = None,
                 category_balanced: bool = True,
                 sample_num: int | None = 10000,
                 sample_replace: bool = False,
                 concat_anndata: bool = True, # set to be False if too many is sampled
                 prefix: str = "scAgeClock_Sampled",
                 outdir: str = "./"
                 ):

        if meta_file_format == "parquet":
            meta_df = pd.read_parquet(h5ad_meta_file)

            if split_column is None:
                meta_df_s = meta_df
            else:
                if not split_column in meta_df.columns:
                    raise ValueError(f"{split_column} not found in the meta file columns")
                else:
                    meta_df_s = meta_df[meta_df[split_column] == split_category_choose]
                    if meta_df_s.empty:
                        raise  ValueError(f"{split_category_choose} not found in meta file column {split_column}")
        else:
            raise ValueError(f"currently only parquet format is supported")

        self.meta_df = meta_df_s.reset_index(drop=True)

        if self.meta_df.empty:
            raise ValueError(f"Empty meta dataframe found")

        self.ad_files = glob.glob(os.path.join(h5ad_file_path, "*.h5ad"))

        self.meta_cell_id_colname = meta_cell_id_colname
        self.anndata_cell_id_colname = anndata_cell_id_colname
        self.backed = backed
        self.meta_category_colname = meta_category_colname
        self.category_list = category_list
        self.category_balanced = category_balanced
        self.sample_num = sample_num
        self.sample_replace = sample_replace
        self.concat_anndata = concat_anndata
        self.prefix = prefix
        self.outdir = outdir
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        else:
            print(f"{self.outdir} already exist.")

        self.total_cell_num = self.meta_df.shape[0]

        if not self.sample_num is None:
            if self.total_cell_num < self.sample_num:
                raise ValueError(f"sampling number ({self.sample_num}) is larger than total cell number ({self.total_cell_num})")


    def random_sample(self):
        if self.sample_num is None:
            raise ValueError(f"sample_num is None, please set a number for it, for example 1000")

        meta_df_s = self.meta_df.sample(self.sample_num, replace=self.sample_replace)
        ad_join = self._extract_by_metadata(meta_df_s)
        return ad_join

    def sample_by_category(self):
        if self.category_list is None:
            print("No category selection")
            meta_df_s = self.meta_df
        else:
            print("Start category selection")
            meta_df_s = self.meta_df[self.meta_df[self.meta_category_colname].isin(self.category_list)]
        if meta_df_s.empty:
            raise ValueError(f"Empty meta dataframe found for {self.category_list} in {self.meta_category_colname} column")

        if not self.sample_num is None:
            if self.category_balanced:
                ## category level balanced sampling of the cells
                cat_num = len(self.meta_df[self.meta_category_colname].unique())
                print(f"Total Number of unique category: {cat_num}")
                sample_size_per_cat = self.sample_num // cat_num
                print(f"Average For each category: {sample_size_per_cat}")
                meta_df_sampled_1 = meta_df_s.groupby(self.meta_category_colname, group_keys=False).apply(
                    lambda x: x.sample(n=min(sample_size_per_cat, len(x)), replace=self.sample_replace)
                )
                remained_sample_size = self.sample_num - len(meta_df_sampled_1)
                if remained_sample_size > 0:
                    ## add further sampling to make the total sampling size as required
                    remained_df = meta_df_s[~meta_df_s.index.isin(meta_df_sampled_1.index)]
                    meta_df_sampled_2 = remained_df.sample(n=remained_sample_size,replace=self.sample_replace)
                    # combine and randomly shuffle
                    meta_df_s2 = pd.concat([meta_df_sampled_1,meta_df_sampled_2]).sample(frac=1).reset_index(drop=True)
                else:
                    meta_df_s2 = meta_df_sampled_1.sample(frac=1).reset_index(drop=True)
            else:
                meta_df_s2 = meta_df_s.sample(self.sample_num, replace=self.sample_replace).reset_index(drop=True)
        else:
            meta_df_s2 = meta_df_s.reset_index(drop=True)

        ad_join = self._extract_by_metadata(meta_df_s2)
        return ad_join

    # TODO: handle with large sampling
    def _extract_by_metadata(self, meta_df_s):
        cell_ids = list(meta_df_s[self.meta_cell_id_colname])
        if self.concat_anndata:
            ad_list = []
            for ad_file in self.ad_files:
                adata = sc.read_h5ad(ad_file, backed=self.backed)
                adata_s = adata[adata.obs[self.anndata_cell_id_colname].isin(cell_ids)]
                if adata_s.shape[0] > 0:
                    ad_list.append(adata_s)
            ad_join = anndata.concat(ad_list)
            ad_join.var_names_make_unique()
            ad_join.write_h5ad(os.path.join(self.outdir, f"{self.prefix}.h5ad"))
            return ad_join
        else:
            chunk_id = 0
            for ad_file in self.ad_files:
                adata = sc.read_h5ad(ad_file, backed=self.backed)
                adata_s = adata[adata.obs[self.anndata_cell_id_colname].isin(cell_ids)]
                if adata_s.shape[0] > 0:
                    chunk_id += 1
                    adata_s.write_h5ad(os.path.join(self.outdir, f"{self.prefix}_chunk{chunk_id}.h5ad"))
            return None







