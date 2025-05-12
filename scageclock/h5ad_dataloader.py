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
        ad = sc.read_h5ad(file_path, backed="r")
        ad_select = ad[index_selected]
        sample_X = ad_select.X.toarray()
        age_soma = ad_select.obs[[self.age_column, self.cell_id]].values
        age_soma = np.array(age_soma, dtype=np.int32)
        return sample_X, age_soma

    def __len__(self):
        return self.total_samples


class BalancedH5ADDataLoader:

    def __init__(self,
                 file_paths: List[str],
                 age_column: str = "age",
                 cell_id: str = "soma_joinid",  ## for tracing the data
                 balanced_feature_col: int = 3,## the first four columns: assay, cell_type, tissue_general, sex. Here default for tissue level balanced sampling
                 balanced_feature_col_max: int = 4,
                 batch_size: int = 1000,
                 batch_iter_max: int = 10000):
        """
        Create a DataLoader based on a list of .h5ad files, and balanced sampling of the cells

        :param file_paths: path to the .h5ad files
        :param age_column: age column name in the adata.obs
        :param cell_id: cell id column name in the adata.obs # default using CELLxGENE soma_joinid
        :param balanced_feature_col: the column index (start from 1) used for balanced sampling # the first four columns: assay, cell_type, tissue_general, sex. default 3 (tissue_general)
        :param balanced_feature_col_max: maximal number of categorical data that can be used for balancing
        :param batch_size: batch size of the DataLoader
        :param batch_iter_max: maximal iteration allowed
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

        cat_stats = feature_idx_df["category"].value_counts().reset_index()
        cats = list(cat_stats["category"])
        cat_num = len(cats)
        mini_batch_size = self.batch_size // cat_num  ## batch size for each selected feature category
        sampled_idx = self._index_sampling(cats=cats, feature_idx_df=feature_idx_df, batch_size=mini_batch_size)
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
                 cell_id: str = "soma_joinid",  ## for tracing the data
                 ):
    ad_files = glob.glob(os.path.join(h5ad_file_path, "*.h5ad"))
    ad_list = [sc.read_h5ad(f) for f in ad_files]

    ad_concat = anndata.concat(ad_list, label="chunk", keys=[os.path.basename(f) for f in ad_files])

    X = ad_concat.X.toarray()
    age_soma = ad_concat.obs[[age_column, cell_id]].values
    age_soma = np.array(age_soma, dtype=np.int32)
    return X, age_soma



