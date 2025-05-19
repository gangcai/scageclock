import os
import pandas as pd
from .h5ad_dataloader import H5ADDataLoader, BalancedH5ADDataLoader

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
                 loader_method: str = "scageclock",
                 dataset_folder_dict=None,
                 balanced_dataloader_parameters: dict | None = None,
                 K_fold_mode: bool = False,
                 K_fold_train: tuple[str] = ("Fold1", "Fold2", "Fold3", "Fold4"), # notice: if only one should be like ("Fold1",)
                 K_fold_val: str = "Fold5",
                 **kwargs
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
        :param loader_method: loader method used: "scageclock" or "scageclock_balanced" (only for training datasets)
        :param dataset_folder_dict: the folder name for each type of datasets, default: {"training": "train", "validation": "val", "testing": "test"}
        :param balanced_dataloader_parameters: dictionary for h5ad_dataloader BalancedH5ADDataLoader
        :param K_fold_mode: whether to use K_fold mode. Each-fold datasets should be under one folder
        :param K_fold_train: The K_fold folders under ad_files_path that are used for training
        :param K_fold_val: The K_fold folder under ad_files_path that are sued for validation
        """

        # default value for dataset_folder_dict if it is None
        if K_fold_mode and (dataset_folder_dict is None):
            dataset_folder_dict = {"training_validation": "train_val"}
        elif dataset_folder_dict is None:
            dataset_folder_dict = {"training": "train", "validation": "val", "testing": "test"}

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

        self.dataset_folder_dict = dataset_folder_dict

        self.K_fold_train = K_fold_train
        self.K_fold_val = K_fold_val
        self.K_fold_mode = K_fold_mode

        ## initiate dataloader for
        self.dataloader_train = None
        self.dataloader_val = None
        self.dataloader_test = None

        self.balanced_dataloader_parameters = balanced_dataloader_parameters

        ## loading the datasets
        self._load_data(**kwargs)

    ## get dataloader based on given .h5ad root directory
    ## under .h5ad root directory, there should be three folders "train", "val" and "test"

    def _load_data(self, **kwargs):
        check_tag = False
        if "training_validation" in self.dataset_folder_dict:
            ad_files_path = os.path.join(self.anndata_dir_root, self.dataset_folder_dict["training_validation"])
            if self.K_fold_mode:
                self.dataloader_train = get_data_loader(ad_files_path=ad_files_path,
                                                        batch_size=self.batch_size_train,
                                                        shuffle=self.shuffle,
                                                        num_workers=self.num_workers,
                                                        cell_id=self.cell_id,
                                                        age_column=self.age_column,
                                                        loader_method=self.loader_method,
                                                        balanced_dataloader_parameters=self.balanced_dataloader_parameters,
                                                        sub_folders= self.K_fold_train,
                                                        )

                self.dataloader_val = get_data_loader(ad_files_path=ad_files_path,
                                                        batch_size=self.batch_size_train,
                                                        shuffle=self.shuffle,
                                                        num_workers=self.num_workers,
                                                        cell_id=self.cell_id,
                                                        age_column=self.age_column,
                                                        loader_method=self.loader_method,
                                                        balanced_dataloader_parameters=self.balanced_dataloader_parameters,
                                                        sub_folders=(self.K_fold_val,),
                                                        )
            else:
                self.dataloader_train = get_data_loader(ad_files_path=ad_files_path,
                                                        batch_size=self.batch_size_train,
                                                        shuffle=self.shuffle,
                                                        num_workers=self.num_workers,
                                                        cell_id=self.cell_id,
                                                        age_column=self.age_column,
                                                        loader_method=self.loader_method,
                                                        balanced_dataloader_parameters=self.balanced_dataloader_parameters,
                                                        sub_folders= self.K_fold_train,
                                                        )
            print(f"training/validation datasets are loaded from {self.dataset_folder_dict["training_validation"]}")
            check_tag = True
        if "training" in self.dataset_folder_dict:

            self.dataloader_train = get_data_loader(ad_files_path=os.path.join(self.anndata_dir_root, self.dataset_folder_dict["training"]),
                                                    batch_size=self.batch_size_train,
                                                    shuffle=self.shuffle,
                                                    num_workers=self.num_workers,
                                                    cell_id=self.cell_id,
                                                    age_column=self.age_column,
                                                    loader_method=self.loader_method,
                                                    balanced_dataloader_parameters = self.balanced_dataloader_parameters,
                                                    **kwargs
                                                    )
            print(f"training datasets are loaded from {self.dataset_folder_dict["training"]}")
            check_tag = True

        if "validation" in self.dataset_folder_dict:
            self.dataloader_val = get_data_loader(ad_files_path=os.path.join(self.anndata_dir_root, self.dataset_folder_dict["validation"]),
                                                  batch_size=self.batch_size_val,
                                                  shuffle=self.shuffle,
                                                  num_workers=self.num_workers,
                                                  cell_id=self.cell_id,
                                                  age_column=self.age_column,
                                                  loader_method="scageclock",
                                                  **kwargs
                                                  )
            print(f"validation datasets are loaded from {self.dataset_folder_dict["validation"]}")
            check_tag = True

        if "testing" in self.dataset_folder_dict:
            self.dataloader_test = get_data_loader(ad_files_path=os.path.join(self.anndata_dir_root, self.dataset_folder_dict["testing"]),
                                                   batch_size=self.batch_size_test,
                                                   shuffle=self.shuffle,
                                                   num_workers=self.num_workers,
                                                   cell_id=self.cell_id,
                                                   age_column=self.age_column,
                                                   loader_method="scageclock",
                                                   **kwargs
                                                   )
            print(f"testing datasets are loaded from {self.dataset_folder_dict["testing"]}")
            check_tag = True

        if not check_tag:
            raise ValueError(f"dataset_folder_dict setting is not correct: {self.dataset_folder_dict}, at least one of training, validation and testing should be in it.")

        return True

    ## load the pandas dataframe with .h5ad .var information (feature names for each column)
    def _load_h5ad_var(self):
        return pd.read_csv(os.path.join(self.anndata_dir_root, self.var_file_name), sep="\t")


# TODO: add fully in memory dataloader for smaller scale h5ad datasets
# TODO: handle if batch_size is larger than the given datasize
def get_data_loader(ad_files_path: str,
                    batch_size: int = 1024,
                    shuffle: bool = True,
                    num_workers: int = 1,
                    age_column: str = "age",
                    cell_id: str = "soma_joinid",
                    loader_method: str = "scageclock",
                    balanced_dataloader_parameters: dict | None = None,
                    sub_folders: tuple[str] | None = None,
                    **kwargs):
    """
    Given the folder path for the .h5ad files, return torch DataLoader

    :param ad_files_path: folder path that contains the .h5ad files
    :param batch_size: batch size of the data loader
    :param shuffle: boolean value to showing whether to shuffle the data
    :param num_workers: number of works for data loader
    :param age_column: the column name for the age
    :param cell_id: the unique cell ID, which is used to trace back to the donor information
    :param loader_method: "scageclock" or "scageclock_balanced"
    :param balanced_dataloader_parameters: dictionary for h5ad_dataloader BalancedH5ADDataLoader (on work for when loader_method == 'scageclock_balanced')
    :param sub_folders: use the .h5ad files under sub_folders
    :return: torch DataLoader
    """
    if not loader_method in ["scageclock","scageclock_balanced"]:
        msg = "Error: loader_method can only be in ['scageclock','scageclock_balanced']"
        raise ValueError(msg)
    if sub_folders is None:
        ad_files = get_h5ad_files(ad_files_path)
    else:
        ad_files = []
        for sub_folder in sub_folders:
            print(os.path.join(ad_files_path, sub_folder))
            sub_folder_ad_files = get_h5ad_files(os.path.join(ad_files_path, sub_folder))
            ad_files += sub_folder_ad_files

    if loader_method == 'scageclock':
        dataloader = H5ADDataLoader(file_paths=ad_files,
                                    batch_size=batch_size,
                                    shuffle=shuffle,
                                    num_workers=num_workers,
                                    age_column=age_column,
                                    cell_id=cell_id
                                    )
        return dataloader
    elif loader_method == 'scageclock_balanced':
        if balanced_dataloader_parameters is None:
            raise ValueError("balanced_dataloader_parameters is not set!")

        if not "batch_iter_max" in balanced_dataloader_parameters:
            balanced_dataloader_parameters["batch_iter_max"] = 10000

        if not "meta_cell_id_column" in balanced_dataloader_parameters:
            balanced_dataloader_parameters["meta_cell_id_column"] = "soma_joinid"

        if not "index_cell_id_column" in balanced_dataloader_parameters:
            balanced_dataloader_parameters["index_cell_id_column"] = "cell_id"

        if not "meta_balanced_column" in balanced_dataloader_parameters:
            balanced_dataloader_parameters["meta_balanced_column"] = "tissue_general"

        if not "h5ad_cell_id_column" in balanced_dataloader_parameters:
            balanced_dataloader_parameters["h5ad_cell_id_column"] = "soma_joinid"

        if sub_folders is None:
            is_meta_index_file_name_full_path = False
        else:
            is_meta_index_file_name_full_path = True
        dataloader = BalancedH5ADDataLoader(h5ad_files_folder_path=balanced_dataloader_parameters["h5ad_files_folder_path"],
                                            h5ad_files_index_file=balanced_dataloader_parameters["h5ad_files_index_file"],
                                            h5ad_files_meta_file=balanced_dataloader_parameters["h5ad_files_meta_file"],
                                            batch_size=batch_size,
                                            age_column=age_column,
                                            h5ad_cell_id_column=balanced_dataloader_parameters["h5ad_cell_id_column"],
                                            index_cell_id_column=balanced_dataloader_parameters["index_cell_id_column"],
                                            meta_cell_id_column=balanced_dataloader_parameters["meta_cell_id_column"],
                                            meta_balanced_column=balanced_dataloader_parameters["meta_balanced_column"],
                                            batch_iter_max=balanced_dataloader_parameters["batch_iter_max"],
                                            is_meta_index_file_name_full_path=is_meta_index_file_name_full_path,
                                            **kwargs
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

