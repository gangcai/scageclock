import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from typing import List
import scanpy as sc
import anndata
import torch
import pandas as pd
import glob
import os
import time

class H5ADDataLoader:

    def __init__(self,
                 file_paths: List[str],
                 age_column: str = "age",
                 cell_id: str = "soma_joinid",  ## for tracing the data
                 batch_size: int = 1000,
                 shuffle: bool = True,
                 num_workers: int = 1): ## TODO: multiple workers doesn't improve the speed
        """
        Create a DataLoader based on a list of .h5ad files

        :param file_paths: path to the .h5ad files
        :param age_column: age column name in the adata.obs
        :param cell_id: cell id column name in the adata.obs # default using CELLxGENE soma_joinid
        :param batch_size: batch size of the DataLoader
        :param shuffle: whether to shuffle the data for batching loading
        :param num_workers: number of parallel jobs for Data Loading
        """
        self.file_paths = file_paths
        self.age_column = age_column
        self.cell_id = cell_id
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_workers = num_workers
        # Store the total number of samples across all .h5ad files
        self.total_samples = sum(sc.read_h5ad(file, backed="r").shape[0] for file in file_paths)
        # Optionally, store the cumulative sizes of each file to efficiently index
        self.cumulative_sizes = self._compute_cumulative_sizes()

        self.batch_indices = self.get_batch_indices()

        self.batch_iter_start = 0
        self.batch_iter_end = len(self.batch_indices)



    def __iter__(self):
        self.batch_iter_start = 0
        if self.shuffle:
            self.batch_indices = self.get_batch_indices()
        return self

    def __next__(self):
        if self.batch_iter_start < self.batch_iter_end:
            exp_arr, age_soma_arr = self.get_batch(batch_index=self.batch_iter_start)
            self.batch_iter_start += 1
            return torch.tensor(exp_arr, dtype=torch.float32), torch.tensor(age_soma_arr, dtype=torch.int32)
        else:
            raise StopIteration

    def get_batch(self,
                  batch_index: int = 0):
        if self.num_workers <= 1:
            return self._get_batch_single_worker(batch_index=batch_index)
        else:
            return self._get_batch_multiple_workers(batch_index=batch_index,
                                                    num_workers=self.num_workers)

    def _get_batch_single_worker(self,
                                 batch_index: int = 0):
        if batch_index >= len(self.batch_indices):
            print(f"batch index out of range")
            return False

        batch_index_list = self.batch_indices[batch_index]
        files2index = self._get_file_indices(batch_index_list)

        exp_arr = None
        age_soma_arr = None
        i = 0
        for file_path in files2index.keys():
            i += 1
            index_selected = files2index[file_path]
            sample_X, age_soma = self._process_h5ad_file(file_path=file_path,
                                                         index_selected=index_selected)
            if i == 1:
                exp_arr = sample_X
                age_soma_arr = age_soma
            else:
                exp_arr = np.vstack((exp_arr, sample_X))
                age_soma_arr = np.vstack((age_soma_arr, age_soma))

        return exp_arr, age_soma_arr

    ## TODO: speed is not improved as expected, needs to improve
    def _get_batch_multiple_workers(self, batch_index: int = 0, num_workers: int = 4):
        if batch_index >= len(self.batch_indices):
            print(f"batch index out of range")
            return False

        batch_index_list = self.batch_indices[batch_index]
        files2index = self._get_file_indices(batch_index_list)

        exp_arr_list = []
        age_soma_arr_list = []

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self._process_h5ad_file, file_path, index_selected): (file_path, index_selected) for
                       file_path, index_selected in files2index.items()}

            for future in as_completed(futures):
                sample_X, age_soma = future.result()
                exp_arr_list.append(sample_X)
                age_soma_arr_list.append(age_soma)

        if exp_arr_list:
            exp_arr = np.vstack(exp_arr_list)
            age_soma_arr = np.vstack(age_soma_arr_list)
            return exp_arr, age_soma_arr
        else:
            return None, None


    def get_batch_indices(self):
        indices_list = list(range(self.total_samples))
        if self.shuffle:
            random.shuffle(indices_list)

        batches = []

        # Loop through the list in steps of batch_size
        for i in range(0, len(indices_list), self.batch_size):
            # Slice the list from the current index to the current index plus batch_size
            batch = indices_list[i:i + self.batch_size]
            # Append the sliced batch to the list of batches
            batches.append(batch)

        return batches

    ## given a list of index, return the dictionary: filename : list of local-indices
    def _get_file_indices(self,
                           index_list : List[int]):
        file2index = {}
        for idx in index_list:
            file_idx = self._find_file_index(idx)
            file_path = self.file_paths[file_idx]

            # get local index for that file
            row_idx = idx - (self.cumulative_sizes[file_idx - 1] if file_idx > 0 else 0)
            if file_path in file2index:
                file2index[file_path].append(row_idx)
            else:
                file2index[file_path] = [row_idx]

        return file2index

    def _find_file_index(self, idx):
        # Binary search to find the file index
        left, right = 0, len(self.file_paths) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.cumulative_sizes[mid] == idx:
                return mid + 1
            elif self.cumulative_sizes[mid] < idx:
                left = mid + 1
            else:
                right = mid - 1
        return left

    def _compute_cumulative_sizes(self):
        sizes = [sc.read_h5ad(file, backed="r").shape[0] for file in self.file_paths]
        cumulative_sizes = []
        cumulative_sum = 0
        for size in sizes:
            cumulative_sum += size
            cumulative_sizes.append(cumulative_sum)
        return cumulative_sizes

    def _process_h5ad_file(self, file_path, index_selected):
        try:
            ad = sc.read_h5ad(file_path, backed="r")
            ad_select = ad[index_selected]
            sample_X = ad_select.X.toarray()
            age_soma = ad_select.obs[[self.age_column, self.cell_id]].values
            age_soma = np.array(age_soma, dtype=np.int32)
        except Exception as e:
            print(f"Warning: backed mode failed with error. Try normal mode h5ad loading without backed-end")
            ad = sc.read_h5ad(file_path)
            ad_select = ad[index_selected]
            sample_X = ad_select.X.toarray()
            age_soma = ad_select.obs[[self.age_column, self.cell_id]].values
            age_soma = np.array(age_soma, dtype=np.int32)
        return sample_X, age_soma

    def __len__(self):
        return self.total_samples

class BalancedH5ADDataLoader:

    def __init__(self,
                 h5ad_files_folder_path: str,
                 h5ad_files_index_file: str,
                 h5ad_files_meta_file: str,
                 index_file_format: str = "parquet",
                 meta_file_format: str = "parquet",
                 age_column: str = "age",
                 h5ad_cell_id_column: str = "soma_joinid", # cell id column name in anndata.obs
                 index_cell_id_column: str = "cell_id",  # cell id column name in h5ad_files_index_file
                 meta_cell_id_column: str = "soma_joinid", # cell id column name in h5ad_files_meta_file
                 meta_balanced_column: str = "tissue_general", # column used for balanced data retrieving (in h5ad_files_meta_file)
                 batch_size: int = 1000,
                 batch_iter_max: int = 10000,
                 is_meta_index_file_name_full_path: bool = False, ## whether the file_name is recorded as a full path file name
                 ):

        self.h5ad_files_folder_path = h5ad_files_folder_path
        self.h5ad_files_index_file = h5ad_files_index_file
        self.h5ad_files_meta_file = h5ad_files_meta_file
        self.index_file_format = index_file_format
        self.meta_file_format = meta_file_format
        self.age_column = age_column
        self.h5ad_cell_id_column = h5ad_cell_id_column
        self.index_cell_id_column = index_cell_id_column
        self.meta_cell_id_column = meta_cell_id_column
        self.meta_balanced_column = meta_balanced_column


        self.batch_size = batch_size
        self.batch_iter_max = batch_iter_max
        self.is_meta_index_file_name_full_path = is_meta_index_file_name_full_path


        if index_file_format == "parquet":
            index_df = pd.read_parquet(self.h5ad_files_index_file)
        else:
            raise ValueError("Only parquet format is supported for index file")

        if meta_file_format == "parquet":
            meta_df = pd.read_parquet(self.h5ad_files_meta_file)
        else:
            raise ValueError("Only parquet format is support for meta file")

        ## process the index and meta file
        self.meta_index_df = pd.merge(index_df,
                                 meta_df[[self.meta_cell_id_column,self.meta_balanced_column]],
                                 left_on=self.index_cell_id_column,
                                 right_on=self.meta_cell_id_column)

        # Store the total number of samples across all .h5ad files
        self.total_samples = len(self.meta_index_df)
        self.batch_iter_start = 0

        self.cats = list(self.meta_index_df[self.meta_balanced_column].unique())

        self.cats_num = len(self.cats)
        self.mini_batch_size = self.batch_size // self.cats_num  ## batch size for each selected feature category

    def __iter__(self):
        self.batch_iter_start = 0
        return self

    def __next__(self):
        if self.batch_iter_start < self.batch_iter_max:
            exp_arr, age_soma_arr = self.sample_batch()
            self.batch_iter_start += 1
            return torch.tensor(exp_arr, dtype=torch.float32), torch.tensor(age_soma_arr, dtype=torch.int32)
        else:
            raise StopIteration

    def sample_batch(self):
        batch_index_list = self.balanced_indices_sampling()
        files2index = self._get_file_indices(batch_index_list)

        exp_arr = None
        age_soma_arr = None
        i = 0
        for file_path in files2index.keys():
            i += 1
            index_selected = files2index[file_path]
            sample_X, age_soma = self._process_h5ad_file(file_path=file_path,
                                                         index_selected=index_selected)
            if i == 1:
                exp_arr = sample_X
                age_soma_arr = age_soma
            else:
                exp_arr = np.vstack((exp_arr, sample_X))
                age_soma_arr = np.vstack((age_soma_arr, age_soma))

        return exp_arr, age_soma_arr


    def balanced_indices_sampling(self):
        sampled_idx = self._index_sampling(batch_size=self.mini_batch_size)

        sampled_len = len(sampled_idx)

        if sampled_len == self.batch_size:
            return sampled_idx
        elif sampled_len > self.batch_size:
            raise ValueError(f"sampled length ({sampled_len}) is larger than the batch size ({self.batch_size})")
        else:
            remained_sample_size = self.batch_size - sampled_len
            feature_idx_df_remained = self.meta_index_df[~self.meta_index_df["global_index"].isin(sampled_idx)]
            remained_sampled = feature_idx_df_remained.sample(n=remained_sample_size,replace=False)
            remained_idx = list(remained_sampled["global_index"])
            sampled_idx = sampled_idx + remained_idx
            return sampled_idx

    def _index_sampling(self,  batch_size):
        sampled_idx = []
        for cat in self.cats:
            feature_idx_df_s = self.meta_index_df[self.meta_index_df[self.meta_balanced_column] == cat]
            count = feature_idx_df_s.shape[0]
            if count < batch_size:
                sample_df = feature_idx_df_s
            else:
                sample_df = feature_idx_df_s.sample(batch_size)
            idx = list(sample_df["global_index"])
            sampled_idx = sampled_idx + idx
        return sampled_idx

    ## given a list of index, return the dictionary: filename : list of local-indices
    def _get_file_indices(self,
                          index_list: List[int]):
        file2index = {}
        meta_index_df_s = self.meta_index_df[self.meta_index_df["global_index"].isin(index_list)]
        filenames = list(meta_index_df_s["file_name"].unique())

        for filename in filenames:
            meta_index_df_s2 = meta_index_df_s[meta_index_df_s["file_name"] == filename]

            meta_index_df_s3 = meta_index_df_s2[meta_index_df_s2["global_index"].isin(index_list)]

            local_index = meta_index_df_s3["local_index"]

            if self.is_meta_index_file_name_full_path:
                filename_full_path = filename
            else:
                filename_full_path = os.path.join(self.h5ad_files_folder_path, filename)
            file2index[filename_full_path] = local_index
        return file2index

    def _process_h5ad_file(self, file_path, index_selected):
        ad = sc.read_h5ad(file_path, backed="r")
        ad_select = ad[index_selected]
        sample_X = ad_select.X.toarray()
        age_soma = ad_select.obs[[self.age_column, self.h5ad_cell_id_column]].values
        age_soma = np.array(age_soma, dtype=np.int32)
        return sample_X, age_soma

    def __len__(self):
        return self.total_samples


# TODO: index the .h5ad files, cell_id, file_name, local_index, global_index
class BalancedH5ADDataLoader_old:

    def __init__(self,
                 file_paths: List[str],
                 age_column: str = "age",
                 cell_id: str = "soma_joinid",  ## for tracing the data
                 balanced_feature_col: int = 3,## the first four columns: assay, cell_type, tissue_general, sex. Here default for tissue level balanced sampling
                 balanced_feature_col_max: int = 4,
                 batch_size: int = 1000,
                 batch_iter_max: int = 10000,
                 feature_idx_df: str | None = None,
                 ):
        """
        Create a DataLoader based on a list of .h5ad files, and balanced sampling of the cells

        :param file_paths: path to the .h5ad files
        :param age_column: age column name in the adata.obs
        :param cell_id: cell id column name in the adata.obs # default using CELLxGENE soma_joinid
        :param balanced_feature_col: the column index (start from 1) used for balanced sampling # the first four columns: assay, cell_type, tissue_general, sex. default 3 (tissue_general)
        :param balanced_feature_col_max: maximal number of categorical data that can be used for balancing
        :param batch_size: batch size of the DataLoader
        :param batch_iter_max: maximal iteration allowed
        :param feature_idx_df: provided file_paths matched feature_idx_df (save time for creating of this dataframe), if None, create it.
        """
        self.file_paths = file_paths
        self.age_column = age_column
        self.cell_id = cell_id
        self.balanced_feature_col = balanced_feature_col
        self.balanced_feature_col_max = balanced_feature_col_max
        self.batch_size = batch_size
        self.batch_iter_max = batch_iter_max

        if (self.balanced_feature_col > self.balanced_feature_col_max) or (self.balanced_feature_col < 1):
            raise ValueError(f"{self.balanced_feature_col} out of range, [1, {self.balanced_feature_col_max}]")

        # Store the total number of samples across all .h5ad files
        self.total_samples = sum(sc.read_h5ad(file, backed="r").shape[0] for file in file_paths)
        # Optionally, store the cumulative sizes of each file to efficiently index
        self.cumulative_sizes = self._compute_cumulative_sizes()
        self.batch_iter_start = 0
        if feature_idx_df is None:
            print("creating global index and category dataframe")
            self.feature_idx_df, self.cats = self.get_feature_idx_df()
            print("global index and category dataframe is created")
        else:
            self.feature_idx_df = feature_idx_df
            self.cats = list(feature_idx_df["category"].unique())

        self.cats_num = len(self.cats)
        self.mini_batch_size = self.batch_size // self.cats_num  ## batch size for each selected feature category

    def __iter__(self):
        self.batch_iter_start = 0
        return self

    def __next__(self):
        if self.batch_iter_start < self.batch_iter_max:
            exp_arr, age_soma_arr = self.sample_batch()
            self.batch_iter_start += 1
            return torch.tensor(exp_arr, dtype=torch.float32), torch.tensor(age_soma_arr, dtype=torch.int32)
        else:
            raise StopIteration

    def sample_batch(self):
        batch_index_list = self.balanced_indices_sampling()
        files2index = self._get_file_indices(batch_index_list)

        exp_arr = None
        age_soma_arr = None
        i = 0
        for file_path in files2index.keys():
            i += 1
            index_selected = files2index[file_path]
            sample_X, age_soma = self._process_h5ad_file(file_path=file_path,
                                                         index_selected=index_selected)
            if i == 1:
                exp_arr = sample_X
                age_soma_arr = age_soma
            else:
                exp_arr = np.vstack((exp_arr, sample_X))
                age_soma_arr = np.vstack((age_soma_arr, age_soma))

        return exp_arr, age_soma_arr


    def get_feature_idx_df(self):
        """
        Creates a DataFrame mapping feature indices to their values from multiple h5ad files.

        Returns:
            pd.DataFrame: DataFrame with columns 'index' and 'category' and unique list of category
        """

        start_time = time.time()

        data = []
        cumulative_index = 0

        total_files_num = len(self.file_paths)
        print(f"total number of files: {total_files_num}")
        counter = 0
        for h5ad_file in self.file_paths:
            try:
                counter += 1
                ad = sc.read_h5ad(h5ad_file, backed='r')

                col_idx = self.balanced_feature_col - 1
                if col_idx >= ad.n_vars:
                    raise ValueError(f"Feature column {self.balanced_feature_col} out of range in {h5ad_file}")

                # Read the entire column (sparse or dense)
                col_data = ad[:, col_idx].X

                features = np.array(col_data.toarray()).flatten()


                indices = np.arange(cumulative_index, cumulative_index + len(features))
                data.append(pd.DataFrame({"index": indices, "category": features}))

                # Update category stats on the fly
                cumulative_index += len(features)

                del ad

                if counter % 30 == 0:
                    print(f"[INFO] Processed file: {counter} and the cumulative index at : {cumulative_index + 1}")
                    print(f"[INFO] Progress: {counter / total_files_num}")
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"[INFO] Time elapsed in {elapsed_time:.2f} seconds.")

            except Exception as e:
                print(f"Error processing {h5ad_file}: {str(e)}")
                continue

        # Combine DataFrames only once
        feature_idx_df = pd.concat(data, ignore_index=True)

        cats = list(feature_idx_df["category"].unique())

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"[INFO] get_feature_idx_df executed in {elapsed_time:.2f} seconds.")
        return feature_idx_df, cats

    def get_feature_idx_df_old(self):
        index_lst = []
        feature_lst = []
        indx = -1
        for h5ad_file in self.file_paths:
            ad = sc.read_h5ad(h5ad_file, backed='r')
            features = ad[:, self.balanced_feature_col - 1].X.toarray().flatten()
            for f in features:
                indx += 1
                index_lst.append(indx)
                feature_lst.append(f)

        feature_idx_df = pd.DataFrame({"index": index_lst,
                                       "category": feature_lst})

        cat_stats = self.feature_idx_df["category"].value_counts().reset_index()
        cats = list(cat_stats["category"])


        return feature_idx_df, cats


    def balanced_indices_sampling(self):
        sampled_idx = self._index_sampling(cats=self.cats, feature_idx_df=self.feature_idx_df, batch_size=self.mini_batch_size)

        sampled_len = len(sampled_idx)

        if sampled_len == self.batch_size:
            return sampled_idx
        elif sampled_len > self.batch_size:
            raise ValueError(f"sampled length ({sampled_len}) is larger than the batch size ({self.batch_size})")
        else:
            remained_sample_size = self.batch_size - sampled_len
            feature_idx_df_remained = self.feature_idx_df[~self.feature_idx_df["index"].isin(sampled_idx)]
            remained_sampled = feature_idx_df_remained.sample(n=remained_sample_size,replace=False)
            remained_idx = list(remained_sampled["index"])
            sampled_idx = sampled_idx + remained_idx
            return sampled_idx

    def _index_sampling(self, cats, feature_idx_df, batch_size):
        sampled_idx = []
        for cat in cats:
            feature_idx_df_s = feature_idx_df[feature_idx_df["category"] == cat]
            count = feature_idx_df_s.shape[0]
            if count < batch_size:
                sample_df = feature_idx_df_s
            else:
                sample_df = feature_idx_df_s.sample(batch_size)
            idx = list(sample_df["index"])
            sampled_idx = sampled_idx + idx
        return sampled_idx

    ## given a list of index, return the dictionary: filename : list of local-indices
    def _get_file_indices(self,
                          index_list: List[int]):
        file2index = {}
        for idx in index_list:
            file_idx = self._find_file_index(idx)
            file_path = self.file_paths[file_idx]

            # get local index for that file
            row_idx = idx - (self.cumulative_sizes[file_idx - 1] if file_idx > 0 else 0)
            if file_path in file2index:
                file2index[file_path].append(row_idx)
            else:
                file2index[file_path] = [row_idx]

        return file2index

    def _find_file_index(self, idx):
        # Binary search to find the file index
        left, right = 0, len(self.file_paths) - 1
        while left <= right:
            mid = (left + right) // 2
            if self.cumulative_sizes[mid] == idx:
                return mid + 1
            elif self.cumulative_sizes[mid] < idx:
                left = mid + 1
            else:
                right = mid - 1
        return left

    def _compute_cumulative_sizes(self):
        sizes = [sc.read_h5ad(file, backed="r").shape[0] for file in self.file_paths]
        cumulative_sizes = []
        cumulative_sum = 0
        for size in sizes:
            cumulative_sum += size
            cumulative_sizes.append(cumulative_sum)
        return cumulative_sizes

    def _process_h5ad_file(self, file_path, index_selected):
        ad = sc.read_h5ad(file_path, backed="r")
        ad_select = ad[index_selected]
        sample_X = ad_select.X.toarray()
        age_soma = ad_select.obs[[self.age_column, self.cell_id]].values
        age_soma = np.array(age_soma, dtype=np.int32)
        return sample_X, age_soma

    def __len__(self):
        return self.total_samples


## given a folder path with .h5ad files, load them all into memory
def fully_loaded(h5ad_file_path: str,
                 age_column: str = "age",
                 cell_id: str = "soma_joinid",
                 return_anndata: bool = False,
                 ):
    ad_files = glob.glob(os.path.join(h5ad_file_path, "*.h5ad"))
    ad_list = [sc.read_h5ad(f) for f in ad_files]

    ad_concat = anndata.concat(ad_list, label="chunk", keys=[os.path.basename(f) for f in ad_files])

    if return_anndata:
        return ad_concat
    else:
        X = ad_concat.X.toarray()
        age_soma = ad_concat.obs[[age_column, cell_id]].values
        age_soma = np.array(age_soma, dtype=np.int32)
        return X, age_soma

def fully_loaded_KFolds(h5ad_file_path: str,
                       age_column: str = "age",
                       cell_id: str = "soma_joinid",
                       return_anndata: bool = False,
                       K_fold_train: tuple[str] = ("Fold1", "Fold2", "Fold3", "Fold4"),
                       ):
    ad_files = []
    for fold in K_fold_train:
        ad_files += glob.glob(os.path.join(h5ad_file_path, fold, "*.h5ad"))

    ad_list = [sc.read_h5ad(f) for f in ad_files]

    ad_concat = anndata.concat(ad_list, label="chunk", keys=[os.path.basename(f) for f in ad_files])

    if return_anndata:
        return ad_concat
    else:
        X = ad_concat.X.toarray()
        age_soma = ad_concat.obs[[age_column, cell_id]].values
        age_soma = np.array(age_soma, dtype=np.int32)
        return X, age_soma


def get_cell_ids(h5ad_file_path: str,
                 cell_id: str = "soma_joinid",
                 ):
    ad_files = glob.glob(os.path.join(h5ad_file_path, "*.h5ad"))

    series_list = []
    for ad_file in ad_files:
        ad = sc.read_h5ad(ad_file, backed="r")
        cell_ids = ad.obs[cell_id]
        series_list.append(cell_ids)
    result = pd.concat(series_list)
    return list(result)

def get_cell_id_index(h5ad_file_path: str,
                      cell_id: str = "soma_joinid",
                      sub_folders: tuple[str] | None = None,
                      is_file_name_full_path: bool = False,):
    if sub_folders is None:
        ad_files = glob.glob(os.path.join(h5ad_file_path, "*.h5ad"))
    else:
        is_file_name_full_path = True
        ad_files = []
        for sub_f in sub_folders:
            this_ad_files = glob.glob(os.path.join(h5ad_file_path,sub_f, "*.h5ad"))
            ad_files += this_ad_files

    global_index = 0
    data = []
    for ad_file in ad_files:
        ad = sc.read_h5ad(ad_file, backed="r")
        cell_ids = list(ad.obs[cell_id])

        num_cell = len(cell_ids)
        indices = np.arange(global_index, global_index + num_cell)
        global_index += num_cell

        local_indices = np.arange(0,num_cell)

        if is_file_name_full_path:
            file_name = ad_file
        else:
            file_name = ad_file.split("/")[-1]

        data.append(pd.DataFrame({"cell_id": cell_ids,
                                  "global_index": indices,
                                  "local_index": local_indices,
                                  "file_name": [file_name] * num_cell
                                  }))

    result = pd.concat(data)
    return result




