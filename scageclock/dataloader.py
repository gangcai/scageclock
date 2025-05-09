import os
import torch
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
from typing import List
import pandas as pd
from .h5ad_dataloader import H5ADDataLoader, BalancedH5ADDataLoader
import numpy as np

# TODO: add fully in memory dataloader for smaller scale h5ad datasets
# TODO: handle if batch_size is larger than the given datasize
def get_data_loader(ad_files_path: str,
                    batch_size: int = 1024,
                    shuffle: bool = True,
                    num_workers: int = 1,
                    age_column: str = "age",
                    cell_id: str = "soma_joinid",
                    loader_method: str = "scellage"):
    """
    Given the folder path for the .h5ad files, return torch DataLoader

    :param ad_files_path: folder path that contains the .h5ad files
    :param batch_size: batch size of the data loader
    :param shuffle: boolean value to showing whether to shuffle the data
    :param num_workers: number of works for data loader
    :param age_column: the column name for the age
    :param cell_id: the unique cell ID, which is used to trace back to the donor information
    :param loader_method: "torch" or "scellage" or "scellage_balanced"
    :return: torch DataLoader
    """
    if not loader_method in ["torch","scellage","scellage_balanced"]:
        msg = "Error: loader_method can only be in ['torch','scellage','scellage_balanced']"
        raise ValueError(msg)

    ad_files = get_h5ad_files(ad_files_path)
    if loader_method == "torch":
        print("warning: the torch loader is currently not recommended.")
        dataset = H5ADDataset(ad_files,
                              age_column=age_column,
                              soma_joinid=cell_id)
        ## TODO: re-write DataLoader class to add donor information
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        return dataloader
    elif loader_method == 'scellage':
        dataloader = H5ADDataLoader(file_paths=ad_files,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    age_column=age_column,
                                    cell_id=cell_id
                                    )
        return dataloader
    elif loader_method == 'scellage_balanced':
        dataloader = BalancedH5ADDataLoader(file_paths=ad_files,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    age_column=age_column,
                                    cell_id=cell_id
                                    )
        return dataloader
    else:
        raise ValueError(f"{loader_method} error")


def get_h5ad_files(ad_files_path: str):
    """
    Get a list of .h5ad files in given path
    :param ad_files_path: folder path that contains the .h5ad files
    :return: a list of .h5ad file paths
    """
    if not os.path.exists(ad_files_path) and os.path.isdir(ad_files_path):
        raise FileNotFoundError(f"Folder not found: {ad_files_path}")

    ad_files = [os.path.join(ad_files_path, f) for f in os.listdir(ad_files_path) if f.endswith('.h5ad')]
    return ad_files


class H5ADDataset(Dataset):
    """
    Create torch Dataset based on .h5ad files

    :param file_paths: folder path that contains that .h5ad files
    :param age_column: column name in the .obs of anndata that stores the age information (default: age)
    """

    def __init__(self,
                 file_paths: List[str],
                 age_column: str = "age",
                 soma_joinid: str = "soma_joinid", ## for tracing the data
                 transform=None):
        self.file_paths = file_paths
        self.transform = transform ## for future using
        self.age_column = age_column
        self.soma_joinid = soma_joinid
        # Store the total number of samples across all Parquet files
        self.total_samples = sum(sc.read_h5ad(file, backed="r").shape[0] for file in file_paths)
        # Optionally, store the cumulative sizes of each file to efficiently index
        self.cumulative_sizes = self.compute_cumulative_sizes()

    def compute_cumulative_sizes(self):
        sizes = [sc.read_h5ad(file, backed="r").shape[0] for file in self.file_paths]
        cumulative_sizes = []
        cumulative_sum = 0
        for size in sizes:
            cumulative_sum += size
            cumulative_sizes.append(cumulative_sum)
        return cumulative_sizes

    def __len__(self):
        return self.total_samples

    # TODO: check possible segmentation error
    def __getitem__(self, idx):
        # Find the file and the row within that file corresponding to the index
        file_idx = self.find_file_index(idx)
        file_path = self.file_paths[file_idx]
        ad = sc.read_h5ad(file_path, backed="r")
        row_idx = idx - (self.cumulative_sizes[file_idx - 1] if file_idx > 0 else 0)

        ad_select = ad[row_idx]
        sample = ad_select.X.toarray()
        sample = sample.flatten()
        age = ad_select.obs[self.age_column].values
        age = round(age[0]) # for age like 0.6 year, it will be round to be 1 year

        ## TODO: add soma_joinid to batch information
        soma_id = ad_select.obs[self.soma_joinid].values
        soma_id = soma_id[0]

        ## testing code, for tracing back the file id
        #chunk_id = file_path.split("/")[-1].split(".")[0]
        #chunk_id = re.sub("Chunk","", chunk_id)
        #chunk_id = int(chunk_id)

        if self.transform:
            sample = self.transform(sample)

        # testing code for tracing back
        #y = [age, soma_id, chunk_id, idx, row_idx] ## add soma_id to trace the donor information

        y = [age, soma_id]
        # notice: torch.tensor(y, dtype=torch.float32) might slight change the original value
        # eg:
        # torch.tensor(74041247, dtype=torch.float32).numpy()
        # -> array(74041248., dtype=float32)
        # This will be an issue for soma_id matching
        # torch.tensor(53017291, dtype=torch.float32).numpy()
        # -> array(53017292., dtype=float32)
        #return torch.tensor(sample, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

        ## torch.tensor(53017291, dtype=torch.int32).numpy()
        ##-> array(53017291, dtype=int32)
        return torch.tensor(sample, dtype=torch.float32), torch.tensor(y, dtype=torch.int32)

    def find_file_index(self, idx):
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


# TODO: when num_workers larger than 1, errors: Unexpected segmentation fault encountered in worker.
# Current only allow num_workers=1

# TODO: Error occurred for most recent version of dependencies. Need to check dependencies

'''
# ValueError: setting an array element with a sequence.
# The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (2,) + inhomogeneous part.
basic_loader = BasicDataLoader(anndata_dir_root=ad_dir_root,
                              batch_size=1024,
                              num_workers=1,
                              loader_method="scellage")
val_data_loader = basic_loader.dataloader_val
X_val, y_and_soma = next(iter(val_data_loader))


File ~/github/scAgingClock/scellage/scellage/h5ad_dataloader.py:201, in H5ADDataLoader._process_h5ad_file(self, file_path, index_selected)
    199 ad = sc.read_h5ad(file_path, backed="r")
    200 ad_select = ad[index_selected]
--> 201 sample_X = ad_select.X.toarray()       
'''

class BasicDataLoader:
    def __init__(self,
                 anndata_dir_root: str,
                 var_file_name: str = "h5ad_var.tsv",
                 var_colname: str = "h5ad_var",
                 batch_size_train: int = 1024,
                 batch_size_val: int = 1024,
                 batch_size_test: int = 1024,
                 shuffle: bool = True,
                 num_workers: int = 10,
                 age_column: str = "age",
                 cell_id: str = "soma_joinid",
                 loader_method: str = "torch"
                 ):
        """
        pytorch Dataloader like DataLoader based on .h5ad files

        :param anndata_dir_root: root directory that stores model datasets:  h5ad_var.tsv  test/*.h5ad  train/*.h5ad  val/*.h5ad
        :param var_file_name: file name for the file with the .h5ad shared .var information, with two columns: var name column and var index
        :param var_colname: column name of the var name in var_file_name
        :param batch_size_train: bath size for training DataLoader
        :param batch_size_val: batch size for validation DataLoader
        :param batch_size_test: batch size for testing DataLoader
        :param shuffle: whether to shuffle the DataLoader
        :param num_workers: number of parallel jobs for Data Loading
        :param age_column: age column name in the adata.obs
        :param cell_id: cell id column name in the adata.obs # default using CELLxGENE soma_joinid
        :param loader_method: loader method used: "torch" or "scellage"
        """
        self.anndata_dir_root = anndata_dir_root

        ## tab-delimited file storing the .var information of .h5ad (shared by all .h5ad files)
        self.var_file_name = var_file_name
        self.var_df = self._load_h5ad_var()
        self.var_colname = var_colname

        self.batch_size_val = batch_size_val
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test

        self.shuffle = shuffle
        self.num_workers = num_workers

        self.age_column = age_column
        self.cell_id = cell_id
        self.loader_method = loader_method

        ## torch dataloader for train/val/test
        dataloader_train, dataloader_val, dataloader_test = self._load_data()
        self.dataloader_train = dataloader_train
        self.dataloader_val = dataloader_val
        self.dataloader_test =dataloader_test

    ## get dataloader based on given .h5ad root directory
    ## under .h5ad root directory, there should be three folders "train", "val" and "test"

    def _load_data(self):
        dataloader_train = get_data_loader(ad_files_path=os.path.join(self.anndata_dir_root, "train"),
                                           batch_size=self.batch_size_train,
                                           shuffle=self.shuffle,
                                           num_workers=self.num_workers,
                                           cell_id=self.cell_id,
                                           age_column=self.age_column,
                                           loader_method=self.loader_method
                                           )
        dataloader_val = get_data_loader(ad_files_path=os.path.join(self.anndata_dir_root, "val"),
                                         batch_size=self.batch_size_val,
                                         shuffle=self.shuffle,
                                         num_workers=self.num_workers,
                                         cell_id=self.cell_id,
                                         age_column=self.age_column,
                                         loader_method=self.loader_method
                                         )
        dataloader_test = get_data_loader(ad_files_path=os.path.join(self.anndata_dir_root, "test"),
                                          batch_size=self.batch_size_test,
                                          shuffle=self.shuffle,
                                          num_workers=self.num_workers,
                                          cell_id=self.cell_id,
                                          age_column=self.age_column,
                                          loader_method=self.loader_method
                                          )
        return dataloader_train, dataloader_val, dataloader_test

    ## load the pandas dataframe with .h5ad .var information (feature names for each column)
    def _load_h5ad_var(self):
        return pd.read_csv(os.path.join(self.anndata_dir_root, self.var_file_name), sep="\t")


##  Create a dataset for the torch DataLoader based on anndata format input and np.array Y
# Example:
# age_somaids = np.array(adata_test_neuron.obs[["age","soma_joinid"]])
# test_dataset = SparseDataset(adata_test_neuron.X, age_somaids)
class SparseDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, idx):
        X_row = self.X[idx].toarray().squeeze()  # Convert single row to dense
        y_value = self.y[idx]
        return torch.tensor(X_row, dtype=torch.float32), torch.tensor(y_value, dtype=torch.int32)

### create dataloader based on a single anndata
## anndata .obs should contain age and cell id columns
def create_dataloader_from_anndata(anndata,
                                   age_column_name: str = "age",
                                   cell_id_column_name: str = "soma_joinid",
                                   batch_size: int = 1024,
                                   shuffle: bool = False):
    age_somaids = np.array(anndata.obs[[age_column_name, cell_id_column_name]], dtype=np.int32)
    dataset = SparseDataset(anndata.X, age_somaids)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return data_loader

