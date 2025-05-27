import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import time
from ..utility import get_validation_metrics
import torch
from ..dataloader import BasicDataLoader
import logging
from ..h5ad_dataloader import fully_loaded, fully_loaded_KFolds
import os


class XGBoostDataLoader(BasicDataLoader):
    def __init__(self,
                 anndata_dir_root: str,
                 var_file_name: str = "h5ad_var.tsv",
                 var_colname: str = "h5ad_var",
                 batch_size_train: int = 1024,
                 batch_size_val: int = 1024,
                 batch_size_test: int = 1024,
                 shuffle: bool = True,
                 num_workers: int = 10,
                 cat_idx_start: int = 0, # not used
                 cat_idx_end: int = 4, # not used
                 age_column: str = "age",
                 cell_id: str = "soma_joinid",
                 loader_method: str = "scageclock",
                 use_cat: bool = False,  # poor performance when setting category type
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
        self.use_cat = use_cat

    def get_inputs(self,
                   X,
                   y):
        X = pd.DataFrame(X, columns=list(self.var_df[self.var_colname]))

        if self.use_cat:
            columns = X.columns
            categorical_cols = list(columns[self.cat_idx_start:self.cat_idx_end])
            # convert categorical value to category type
            X[categorical_cols] = X[categorical_cols].astype("category")
            y = np.array(y)
        else:
            y = np.array(y)

        return X, y


## aging clock based on catboost model
class XGBoostAgeClock:

    def __init__(self,
                 anndata_dir_root: str,
                 dataset_folder_dict: dict | None = None,
                 predict_dataset: str = "validation",
                 validation_during_training: bool = True,
                 learning_rate: float = 0.3, # eta values
                 n_estimators: int = 100, # number of gradient boosted trees. Equivalent to number of boosting rounds.
                 early_stopping_rounds: int = 20, # stop training if no improvements for this number of rounds
                 max_depth: int = 6,
                 subsample: float = 0.8,
                 reg_alpha: float = 0, # alpha value for L1 regularization
                 reg_lambda: float = 1, # lambda value for L2 regularization
                 device: str = "cuda", # cpu or cuda
                 colsample_bytree: float = 0.8,
                 objective: str = "reg:squarederror", # reg:squarederror--> MSE; reg:absoluteerror--> MAE; reg:tweedie
                 random_seed: int = 10,
                 verbose: int = 10,
                 cat_idx_start: int = 0,
                 cat_idx_end: int = 4,
                 enable_categorical: bool = True,
                 var_file_name: str = "h5ad_var.tsv",
                 var_colname: str = "h5ad_var",
                 batch_size_train: int = 1024,
                 batch_size_val: int = 1024,
                 batch_size_test: int = 1024,
                 shuffle: bool = True,
                 num_workers: int = 1, # for data loader
                 n_jobs: int = 10, # for XGBRegressor n_jobs
                 age_column: str = "age",
                 cell_id: str = "soma_joinid",
                 loader_method: str = "scageclock",
                 train_dataset_fully_loaded: bool = False,
                 ## load all .h5ad training files into memory and concatenate into one anndata
                 predict_dataset_fully_loaded: bool = False,
                 ## load all .h5ad prediction files into memory and concatenate into one anndata
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

        self.learning_rate = learning_rate

        self.n_estimators = n_estimators
        self.early_stopping_rounds= early_stopping_rounds
        self.max_depth = max_depth
        self.subsample = subsample
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.device = device
        self.colsample_bytree = colsample_bytree
        self.objective = objective
        self.random_seed = random_seed
        self.verbose = verbose
        self.cat_idx_start = cat_idx_start
        self.cat_idx_end = cat_idx_end
        self.enable_categorical = enable_categorical
        self.var_file_name = var_file_name
        self.var_colname = var_colname
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.n_jobs = n_jobs
        self.age_column = age_column
        self.cell_id = cell_id
        self.loader_method = loader_method
        self.train_dataset_fully_loaded = train_dataset_fully_loaded
        self.predict_dataset_fully_loaded = predict_dataset_fully_loaded
        self.validation_dataset_fully_loaded = validation_dataset_fully_loaded

        self.train_batch_iter_max = train_batch_iter_max
        self.predict_batch_iter_max = predict_batch_iter_max

        self.K_fold_mode = K_fold_mode
        self.K_fold_train = K_fold_train
        self.K_fold_val = K_fold_val

        # Configure logging
        self.log_file = log_file
        logging.basicConfig(filename=self.log_file, level=logging.INFO)

        ## loading the data (lazy loaded, without loading all into memory)
        self.dataloader = XGBoostDataLoader(anndata_dir_root=self.anndata_dir_root,
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

        ## create XGBoostRegressor model
        # eval_metric will be chosen based on objective parameter setting
        self.model = XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            subsample=self.subsample,
            device=self.device,
            objective=self.objective,
            colsample_bytree=self.colsample_bytree,
            seed=self.random_seed,
            n_jobs=self.n_jobs,
            enable_categorical=self.enable_categorical,
            early_stopping_rounds=self.early_stopping_rounds,
            learning_rate=self.learning_rate,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            **kwargs)

        if self.validation_during_training:
            self.X_val, self.y_val, self.val_soma_ids = self._get_val_data()
        self.eval_metrics = None

    def train(self,):
        start_time = time.time()  # Start timing
        if not self.train_dataset_fully_loaded:
            eval_metrics_list = []
            print("Start training")
            logging.info("Start training")
            for i, (features, labels_soma) in enumerate(self.dataloader.dataloader_train, start=1):
                labels, soma_ids = torch.split(labels_soma, split_size_or_sections=1, dim=1) ## TODO: double check
                X_train, y_train = self.dataloader.get_inputs(X=features,
                                                              y=labels)

                # Train the model on the current batch
                if i == 1:
                    if self.validation_during_training:
                        self.model.fit(X_train, y_train, eval_set=[(X_train, y_train), (self.X_val, self.y_val)])
                    else:
                        self.model.fit(X_train, y_train)
                else:
                    if self.validation_during_training:
                        self.model.fit(X_train, y_train,
                                       eval_set=[(X_train, y_train), (self.X_val, self.y_val)],
                                       xgb_model=self.model.get_booster())
                    else:
                        self.model.fit(X_train, y_train,
                                       xgb_model=self.model.get_booster())

                if self.validation_during_training:
                    eval_metrics_list.append(self.model.evals_result_)  ## keep evals_result_ from each model
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"Accumulated time cost for {i}: {elapsed_time:.6f} seconds")  # print time relapse
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
            X_train, y_train = self.dataloader.get_inputs(X=features,
                                                          y=labels)
            if self.validation_during_training:
                self.model.fit(X_train, y_train, eval_set=[(X_train, y_train), (self.X_val, self.y_val)])
                self.eval_metrics = self._reformat_eval_metrics([self.model.evals_result_])
            else:
                print("warning: no validation data")
                self.model.fit(X_train, y_train, eval_set = [(X_train, y_train)])

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
            for i, (features, labels_soma) in enumerate(predict_dataloader, start=1):
                labels, soma_ids = torch.split(labels_soma, split_size_or_sections=1, dim=1)
                X_test, y_test = self.dataloader.get_inputs(X=features,
                                                            y=labels)
                outputs = self.model.predict(X_test)
                outputs = outputs.squeeze()
                outputs = [float(x) for x in outputs]
                y_test = y_test.squeeze()
                y_test = [float(x) for x in y_test]
                predictions.extend(outputs)
                targets_all.extend(y_test)
                soma_ids_all.extend(soma_ids.numpy())
                test_samples_num += features.size(0)
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
                X_test, y_test = self.dataloader.get_inputs(X=X,
                                                           y=targets_all)
                predictions = self.model.predict(X_test)
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
                X_test, y_test = self.dataloader.get_inputs(X=X,
                                                           y=targets_all)
                predictions = self.model.predict(X_test)
                predictions = predictions.squeeze()

        return predictions, targets_all, np.array(soma_ids_all).flatten()

    def get_feature_importance(self):
        return self.model.feature_importances_

    def write_feature_importance(self,
                               var_file,
                               gene_column_name: str = "h5ad_var",
                               outfile: str = "XGBoostAgeClock_FeatureImportances.tsv"):
        feature_importance = self.model.feature_importances_
        var_df = pd.read_csv(var_file, sep="\t")
        var_df["feature_importance"] = feature_importance
        var_df = var_df.sort_values(by="feature_importance", ascending=False)
        f_importance_df = var_df[[gene_column_name, "feature_importance"]]
        f_importance_df.columns = ["gene", "feature_importance"]
        f_importance_df.to_csv(outfile,
                               sep="\t",
                               index=False)
        return f_importance_df


    def _get_val_data(self):
        if not self.validation_dataset_fully_loaded:
            print("One batch of validation dataset is used")
            data_iter_val = iter(self.dataloader.dataloader_val)
            X_val, y_and_soma = next(data_iter_val)
            y_val, soma_ids = torch.split(y_and_soma, split_size_or_sections=1, dim=1)
            X_val, y_val = self.dataloader.get_inputs(X=X_val,
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
                X_val, y_val = self.dataloader.get_inputs(X=X_val,
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
                X_val, y_val = self.dataloader.get_inputs(X=X_val,
                                                      y=y_val)
        return X_val, y_val, soma_ids

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
                train_metric = list(metric["validation_0"]["rmse"]) # training datasets
                val_metric = list(metric["validation_1"]["rmse"]) # validation datasets
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
                train_metric = list(metric["validation_0"]["rmse"])  # training datasets
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



