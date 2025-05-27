import pandas as pd
import numpy as np
from catboost import Pool, CatBoostRegressor
import time
from ..utility import get_validation_metrics
import torch
from ..dataloader import BasicDataLoader
from ..h5ad_dataloader import fully_loaded, fully_loaded_KFolds
import logging
import os

class CatBoostDataLoader(BasicDataLoader):
    def __init__(self,
                 anndata_dir_root: str,
                 var_file_name: str = "h5ad_var.tsv",
                 var_colname: str = "h5ad_var",
                 batch_size_train: int = 1024,
                 batch_size_val: int = 1024,
                 batch_size_test: int = 1024,
                 shuffle: bool = True,
                 num_workers: int = 10,
                 cat_idx_start: int = 0,
                 cat_idx_end: int = 4,
                 age_column: str = "age",
                 cell_id: str = "soma_joinid",
                 loader_method: str = "scageclock",
                 dataset_folder_dict: dict | None = None,
                 K_fold_mode: bool = False,
                 K_fold_train: tuple[str] = ("Fold1", "Fold2", "Fold3", "Fold4"),
                 K_fold_val: str = "Fold5",
                 ):
        if dataset_folder_dict is None:
            dataset_folder_dict = {"training": "train", "validation": "val", "testing": "test"}
        # Call the parent class's __init__ method using super()
        super().__init__(anndata_dir_root=anndata_dir_root,
                         var_file_name=var_file_name,
                         var_colname=var_colname,
                         batch_size_val=batch_size_val,
                         batch_size_train=batch_size_train,
                         batch_size_test=batch_size_test,
                         shuffle=shuffle,
                         num_workers=num_workers,
                         age_column=age_column,
                         cell_id=cell_id,
                         loader_method=loader_method,
                         dataset_folder_dict=dataset_folder_dict,
                         K_fold_mode=K_fold_mode,
                         K_fold_train=K_fold_train,
                         K_fold_val=K_fold_val
                         )
        self.cat_idx_start = cat_idx_start
        self.cat_idx_end = cat_idx_end

    ## get catboost Pool data
    def get_pool_data(self,
                      X,
                      y):
        X_df = pd.DataFrame(X, columns=list(self.var_df[self.var_colname]))
        y = np.array(y)
        columns = X_df.columns
        categorical_cols = list(columns[self.cat_idx_start:self.cat_idx_end])
        # convert categorical value to int type
        X_df[categorical_cols] = X_df[categorical_cols].astype(int)
        data_pool = Pool(data=X_df, label=y, cat_features=categorical_cols)
        return data_pool


## aging clock based on catboost model
class CatBoostAgeClock:

    def __init__(self,
                 anndata_dir_root: str,
                 dataset_folder_dict: dict | None = None,
                 predict_dataset: str = "validation",
                 validation_during_training: bool = True,
                 iterations: int = 100,
                 learning_rate: float = 0.1,
                 depth: int = 6,
                 bootstrap_type = None,  # Bernoulli, Poisson
                 task_type = None,  # GPU
                 used_ram_limit = "100GB",  # Change this according to the computing resource
                 rsm = None,
                 loss_function: str = "RMSE",
                 eval_metric: str = "RMSE",
                 random_seed: int = 10,
                 verbose: int = 10,
                 allow_writing_files: bool = False,
                 cat_idx_start: int = 0,
                 cat_idx_end: int = 4,
                 var_file_name: str = "h5ad_var.tsv",
                 var_colname: str = "h5ad_var",
                 batch_size_train: int = 1024,
                 batch_size_val: int = 1024,
                 batch_size_test: int = 1024,
                 shuffle: bool = True,
                 num_workers: int = 1,
                 age_column: str = "age",
                 cell_id: str = "soma_joinid",
                 loader_method: str = "scageclock",
                 train_dataset_fully_loaded: bool = False, ## load all .h5ad training files into memory and concatenate into one anndata
                 predict_dataset_fully_loaded: bool = False, ## load all .h5ad prediction files into memory and concatenate into one anndata
                 validation_dataset_fully_loaded: bool = False,
                 train_batch_iter_max: int = 1,  ## maximal number of batch iteration for model training
                 predict_batch_iter_max: int = 20,
                 K_fold_mode: bool = False,
                 K_fold_train: tuple[str] = ("Fold1", "Fold2", "Fold3", "Fold4"),
                 K_fold_val: str = "Fold5",
                 log_file: str = "log.txt",
                 **kwargs
                 ):

        # default value for dataset_folder_dict if it is None
        if K_fold_mode and (dataset_folder_dict is None):
            dataset_folder_dict = {"training_validation": "train_val"}
        elif dataset_folder_dict is None:
            dataset_folder_dict = {"training": "train", "validation": "val", "testing": "test"}

        self.anndata_dir_root = anndata_dir_root
        self.dataset_folder_dict = dataset_folder_dict
        self.predict_dataset = predict_dataset
        self.validation_during_training = validation_during_training

        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.bootstrap_type = bootstrap_type
        self.task_type = task_type
        self.used_ram_limit = used_ram_limit
        self.rsm = rsm
        self.loss_function = loss_function
        self.eval_metric = eval_metric
        self.random_seed = random_seed
        self.verbose = verbose
        self.allow_writing_files = allow_writing_files
        self.cat_idx_start = cat_idx_start
        self.cat_idx_end = cat_idx_end
        self.var_file_name = var_file_name
        self.var_colname = var_colname
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.age_column = age_column
        self.cell_id = cell_id
        self.loader_method = loader_method
        self.train_dataset_fully_loaded = train_dataset_fully_loaded
        self.predict_dataset_fully_loaded = predict_dataset_fully_loaded
        self.validation_dataset_fully_loaded = validation_dataset_fully_loaded
        self.train_batch_iter_max = train_batch_iter_max

        self.K_fold_mode = K_fold_mode
        self.K_fold_train = K_fold_train
        self.K_fold_val = K_fold_val


        # Configure logging
        self.log_file = log_file
        logging.basicConfig(filename=self.log_file, level=logging.INFO)

        ## loading the data (lazy loaded, without loading all into memory)
        self.dataloader = CatBoostDataLoader(anndata_dir_root=self.anndata_dir_root,
                                             cat_idx_start=self.cat_idx_start,
                                             cat_idx_end=self.cat_idx_end,
                                             var_file_name=self.var_file_name,
                                             var_colname=self.var_colname,
                                             batch_size_val=self.batch_size_val,
                                             batch_size_train=self.batch_size_train,
                                             batch_size_test=self.batch_size_test,
                                             shuffle=self.shuffle,
                                             num_workers=self.num_workers,
                                             age_column=self.age_column,
                                             cell_id=self.cell_id,
                                             loader_method=self.loader_method,
                                             dataset_folder_dict=self.dataset_folder_dict,
                                             K_fold_mode=self.K_fold_mode,
                                             K_fold_train=self.K_fold_train,
                                             K_fold_val=self.K_fold_val
                                             )

        ## loading all training data to the memory
        if self.train_dataset_fully_loaded:
            if not self.K_fold_mode:
                print("All training .h5ad files are loaded into memory!")
                train_h5ad_dir = os.path.join(anndata_dir_root, self.dataset_folder_dict["training"])
                self.train_all_data = fully_loaded(train_h5ad_dir,
                                                   age_column=self.age_column,
                                                   cell_id=self.cell_id)
            else:
                train_h5ad_dir = os.path.join(anndata_dir_root, self.dataset_folder_dict["training_validation"])
                self.train_all_data = fully_loaded_KFolds(train_h5ad_dir,
                                                          K_fold_train=self.K_fold_train,
                                                          age_column=self.age_column,
                                                          cell_id=self.cell_id)


        ## create CatBoostRegressor model
        self.model = CatBoostRegressor(iterations=self.iterations,
                                       learning_rate=self.learning_rate,
                                       depth=self.depth,
                                       bootstrap_type=self.bootstrap_type,
                                       task_type=self.task_type,
                                       used_ram_limit=self.used_ram_limit,
                                       rsm=self.rsm,
                                       loss_function=self.loss_function,
                                       eval_metric=self.eval_metric,
                                       random_seed=self.random_seed,
                                       verbose=self.verbose,
                                       allow_writing_files=self.allow_writing_files,
                                       **kwargs)

        self.train_batch_iter_max = train_batch_iter_max
        self.predict_batch_iter_max = predict_batch_iter_max

        if self.validation_during_training:
            self.val_pool, self.val_soma_ids = self._get_val_pool()
        self.eval_metrics = None

    def train(self,):
        start_time = time.time()  # Start timing
        if not self.train_dataset_fully_loaded:
            eval_metrics_list = []
            init_model = None
            print("Start training in dataloader mode")
            logging.info("Start training in dataloader mode")
            for i, (features, labels_soma) in enumerate(self.dataloader.dataloader_train, start=1):
                labels, soma_ids = torch.split(labels_soma, split_size_or_sections=1, dim=1) ## TODO: double check
                train_pool = self.dataloader.get_pool_data(X=features,
                                                           y=labels)
                if self.validation_during_training:
                    self.model.fit(train_pool, eval_set=self.val_pool, init_model=init_model, verbose=10)
                else:
                    print("warning: no validation data")
                    self.model.fit(train_pool,  init_model=init_model, verbose=10)
                init_model = self.model  # update the model
                if self.validation_during_training:
                    eval_metrics_list.append(self.model.evals_result_) ## keep evals_result_ from each model
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Accumulated time cost for iteration {i}: {elapsed_time:.6f} seconds")  # print time relapse
                logging.info(f"Accumulated time cost for iteration {i}: {elapsed_time:.6f} seconds")
                if i >= self.train_batch_iter_max:
                    print(f"Reaching maximal iter number: {self.train_batch_iter_max}")
                    logging.info(f"Reaching maximal iter number: {self.train_batch_iter_max}")
                    break
            if self.validation_during_training:
                self.eval_metrics = self._reformat_eval_metrics(eval_metrics_list)
        else:
            print("Start training in normal mode")
            logging.info("Start training in normal mode")
            features = self.train_all_data[0]
            labels = self.train_all_data[1][:,0]
            train_pool = self.dataloader.get_pool_data(X=features,
                                                       y=labels)
            if self.validation_during_training:
                self.model.fit(train_pool, eval_set=self.val_pool,  verbose=10)
            else:
                print("warning: no validation data")
                self.model.fit(train_pool, verbose=10)
            if self.validation_during_training:
                self.eval_metrics = self._reformat_eval_metrics([self.model.evals_result_])
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time cost for the training: {elapsed_time:.6f} seconds")  # print time relapse
            logging.info(f"Time cost for the training: {elapsed_time:.6f} seconds")
        end_time = time.time()  # End timing
        elapsed_time = end_time - start_time
        print(f"Total time costs: {elapsed_time:.6f} seconds")  # print time relapse
        logging.info(f"Total time costs: {elapsed_time:.6f} seconds")
        return True

    def cell_level_test(self):
        y_test_pred, y_test, soma_ids_all = self.predict()
        test_metrics_df = get_validation_metrics(y_true=y_test,
                               y_pred=y_test_pred,
                               print_metrics=False)
        return test_metrics_df, y_test_pred, y_test, soma_ids_all

    def predict(self):
        predictions, targets_all, soma_ids_all = self._predict_basic()
        return predictions, targets_all, soma_ids_all

    def get_feature_importance(self):
        return self.model.feature_importances_


    ## use the first batch of the validation data loader as the validation pool for the training process evaluation
    def _get_val_pool(self):

        if not self.validation_dataset_fully_loaded:
            print("Only one batch of the validation data is used (memory-saving for large-scale validation datasets)")
            data_iter_val = iter(self.dataloader.dataloader_val)
            X_val, y_and_soma = next(data_iter_val)
            y_val, soma_ids = torch.split(y_and_soma, split_size_or_sections=1, dim=1)
            val_pool = self.dataloader.get_pool_data(X=X_val,
                                                     y=y_val)
        else:
            if not self.K_fold_val:
                print("All validation data is used and loaded into memory")
                X_val, y_and_soma = fully_loaded(os.path.join(self.anndata_dir_root,
                                                          self.dataset_folder_dict["validation"]),
                                             age_column=self.age_column,
                                             cell_id=self.cell_id)
                y_val = y_and_soma[:,0]
                soma_ids = y_and_soma[:,1]
                val_pool = self.dataloader.get_pool_data(X=X_val,
                                                         y=y_val)
            else:
                print("All validation data is used and loaded into memory")
                X_val, y_and_soma = fully_loaded(os.path.join(self.anndata_dir_root,
                                                          self.dataset_folder_dict["training_validation"],
                                                          self.K_fold_val),
                                             age_column=self.age_column,
                                             cell_id=self.cell_id)
                y_val = y_and_soma[:,0]
                soma_ids = y_and_soma[:,1]
                val_pool = self.dataloader.get_pool_data(X=X_val,
                                                         y=y_val)

        return val_pool, soma_ids

    def _predict_basic(self, ):
        if not self.predict_dataset_fully_loaded:
            print("prediction based on multiple batches")
            if self.predict_dataset == "testing":
                if self.dataloader.dataloader_test is None:
                    raise ValueError("testing datasets is not provided!")
                else:
                    predict_dataloader = self.dataloader.dataloader_test
            elif self.predict_dataset == "validation":
                if self.dataloader.dataloader_val is None:
                    raise ValueError("validation datasets is not provided!")
                else:
                    predict_dataloader = self.dataloader.dataloader_val
            elif self.predict_dataset == "training":
                if self.dataloader.dataloader_train is None:
                    raise ValueError("training datasets is not provided!")
                else:
                    predict_dataloader = self.dataloader.dataloader_train
            else:
                raise ValueError("supported datasets for prediction: training, testing, and validation")
            predictions = []
            targets_all = []
            soma_ids_all = []
            iter_num = 0
            test_samples_num = 0
            for inputs, labels_soma in predict_dataloader:
                labels, soma_ids = torch.split(labels_soma, split_size_or_sections=1, dim=1)
                test_pool = self.dataloader.get_pool_data(X=inputs,
                                                          y=labels)
                outputs = self.model.predict(test_pool)
                outputs = outputs.squeeze()
                labels = labels.squeeze()
                labels = labels.to(torch.float32)
                predictions.extend(outputs)
                targets_all.extend(labels.numpy())
                soma_ids_all.extend(soma_ids.numpy())
                test_samples_num += inputs.size(0)
                iter_num += 1
                if iter_num >= self.predict_batch_iter_max:
                    break
        else:
            if not self.K_fold_mode:
                print("prediction based on all prediction datasets, all of which is loaded into memory")
                X, y_and_soma = fully_loaded(os.path.join(self.anndata_dir_root,
                                                          self.dataset_folder_dict[self.predict_dataset]),
                                             age_column=self.age_column,
                                             cell_id=self.cell_id)
                targets_all = y_and_soma[:,0]
                soma_ids_all = y_and_soma[:,1]
                predict_pool = self.dataloader.get_pool_data(X=X,
                                                         y=targets_all)
                predictions = self.model.predict(predict_pool)
                predictions = predictions.squeeze()
            else:
                print("prediction based on all prediction datasets, all of which is loaded into memory")
                X, y_and_soma = fully_loaded(os.path.join(self.anndata_dir_root,
                                                          self.dataset_folder_dict["training_validation"],
                                                          self.K_fold_val),
                                             age_column=self.age_column,
                                             cell_id=self.cell_id)
                targets_all = y_and_soma[:,0]
                soma_ids_all = y_and_soma[:,1]
                predict_pool = self.dataloader.get_pool_data(X=X,
                                                         y=targets_all)
                predictions = self.model.predict(predict_pool)
                predictions = predictions.squeeze()

        return predictions, targets_all, np.array(soma_ids_all).flatten()

    ## process the CatBoost evals_result_ from multiple batch training
    def _reformat_eval_metrics(self,
                              eval_metrics_list):
        if self.validation_during_training:
            all_batch_id = []
            all_train_rmse = []
            all_val_rmse = []
            all_steps = []
            i = 0
            for metric in eval_metrics_list:
                train_metric = list(metric["learn"]["RMSE"])
                val_metric = list(metric["validation"]["RMSE"])
                all_train_rmse = all_train_rmse + train_metric
                all_val_rmse = all_val_rmse + val_metric
                all_batch_id = all_batch_id + list([i] * len(train_metric))
                for val in val_metric:
                    i += 1
                    all_steps.append(i)

            train_metrics_df = pd.DataFrame({"batch_id": all_batch_id * 2,
                                             "step": all_steps * 2,
                                             "RMSE": all_train_rmse + all_val_rmse,
                                             "label": ["train"] * len(all_train_rmse) + ["validation"] * len(all_val_rmse)})
        else:
            all_batch_id = []
            all_train_rmse = []
            all_steps = []
            i = 0
            for metric in eval_metrics_list:
                train_metric = list(metric["learn"]["RMSE"])
                all_train_rmse = all_train_rmse + train_metric
                all_batch_id = all_batch_id + list([i] * len(train_metric))
                for val in train_metric:
                    i += 1
                    all_steps.append(i)

            train_metrics_df = pd.DataFrame({"batch_id": all_batch_id,
                                             "step": all_steps,
                                             "RMSE": all_train_rmse,
                                             "label": ["train"] * len(all_train_rmse)})
        return train_metrics_df



