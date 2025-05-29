from ..dataloader import BasicDataLoader
import logging
import torch
import time

import torch.nn as nn
import torch.optim as optim
from ..utility import get_validation_metrics
import numpy as np


class TorchElasticNetAgeClock:
    def __init__(self,
                 anndata_dir_root: str,
                 dataset_folder_dict: dict | None = None,
                 feature_size: int = 19183,
                 predict_dataset: str = "testing",
                 validation_during_training: bool = True,
                 cat_card_list: list | None = None,
                 n_embed: int = 4,
                 var_file_name: str = "h5ad_var.tsv",
                 var_colname: str = "h5ad_var",
                 loader_method: str = "torch",
                 batch_size_train: int = 1024,
                 batch_size_val: int = 1024,
                 batch_size_test: int = 1024,
                 shuffle: bool = True,
                 num_workers: int = 10,
                 output_size: int = 1,
                 l1_lambda: float = 0.01,
                 l2_lambda: float = 0.01,
                 dropout_prob: float = 0.2,
                 learning_rate: float = 0.001,
                 epochs: int = 1,
                 weight_decay: float = 0.001,
                 patience: int = 3,
                 scheduler_factor: float = 0.5,
                 train_batch_iter_max: int | None = 100,  ## maximal number of iteration for the DataLoader during training
                 predict_batch_iter_max: int | None = 20,
                 K_fold_mode: bool = False,
                 K_fold_train: tuple[str] = ("Fold1", "Fold2", "Fold3", "Fold4"),
                 K_fold_val: str = "Fold5",
                 device: str = "cpu",
                 log_file: str = "TorchElasticNetAgeClock_log.txt"):

        if cat_card_list is None:
            # ['assay', 'cell_type', 'tissue_general', 'sex'],
            # the cardinalities for each categorical feature column, the first len(cat_car_list) columns
            #cat_card_list = [14, 219, 39, 3]
            cat_card_list = [21, 664, 52, 3]

        # default value for dataset_folder_dict if it is None
        if K_fold_mode and (dataset_folder_dict is None):
            dataset_folder_dict = {"training_validation": "train_val"}
        elif dataset_folder_dict is None:
            dataset_folder_dict = {"training": "train", "validation": "val", "testing": "test"}

        self.anndata_dir_root = anndata_dir_root
        self.dataset_folder_dict = dataset_folder_dict
        self.feature_size = feature_size
        self.predict_dataset = predict_dataset
        self.validation_during_training = validation_during_training
        self.cat_card_list = cat_card_list
        self.n_embed = n_embed
        self.var_file_name = var_file_name
        self.var_colname = var_colname
        self.loader_method = loader_method
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.batch_size_test = batch_size_test
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.output_size = output_size
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weight_decay = weight_decay
        self.device = device
        self.patience = patience
        self.scheduler_factor = scheduler_factor
        self.train_batch_iter_max = train_batch_iter_max
        self.predict_batch_iter_max = predict_batch_iter_max

        ## training loss metrics
        self.epoch_train_loss_list = []
        self.epoch_val_loss_list = []
        self.batch_train_loss_list = []
        self.batch_val_loss_list = []

        self.log_file = log_file
        # Configure logging
        logging.basicConfig(filename=self.log_file, level=logging.INFO)

        ## loading the data
        self.dataloader = BasicDataLoader(anndata_dir_root=self.anndata_dir_root,
                                          var_file_name=self.var_file_name,
                                          var_colname=self.var_colname,
                                          batch_size_val=self.batch_size_val,
                                          batch_size_train=self.batch_size_train,
                                          batch_size_test=self.batch_size_test,
                                          shuffle=self.shuffle,
                                          num_workers=self.num_workers,
                                          loader_method=self.loader_method,
                                          dataset_folder_dict=self.dataset_folder_dict,
                                          K_fold_mode=K_fold_mode,
                                          K_fold_train=K_fold_train,
                                          K_fold_val=K_fold_val
                                          )

        ## checking for validation
        if self.validation_during_training:
            # sampling one batch from validation datasets for validation checking
            if not self.dataloader.dataloader_val is None:
                self.val_X_batch, self.val_y_batch = self.get_val_sample_batch()
            else:
                raise ValueError("validation datasets is not provided!")

        # Initialize the model, loss function, and optimizer
        self.model = TorchElasticNetRegression(cat_cardinalities=self.cat_card_list,
                                               num_numeric_features=feature_size - len(self.cat_card_list),
                                               n_embed=self.n_embed,
                                               output_dim=self.output_size,
                                               l1_lambda=self.l1_lambda,
                                               l2_lambda=self.l2_lambda, ).to(self.device)

        ## Use ElasticNet regularization for the loss calculation
        self.criterion = nn.MSELoss()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                               mode='min',
                                                               patience=self.patience,
                                                               factor=self.scheduler_factor)

    def train(self):
        (self.epoch_train_loss_list, self.epoch_val_loss_list,
         self.batch_train_loss_list, self.batch_val_loss_list) = self._train_basic_batch_level_validation()
        return True

    ## test the age prediction performance at single cell level
    def cell_level_test(self):
        y_test_pred, y_test, soma_ids_all = self.predict()
        test_metrics_df = get_validation_metrics(y_true=y_test,
                               y_pred=y_test_pred,
                               print_metrics=False)
        return test_metrics_df, y_test_pred, y_test, soma_ids_all

    def predict(self):
        predictions, targets_all, soma_ids_all, avg_loss = self._predict_basic()
        return predictions, targets_all, soma_ids_all

    # TODO: improve the model saving and loading
    def save(self,
             saved_model_file_name: str = "scageclock_MLP_Model.pth"):
        torch.save(self.model.state_dict(), saved_model_file_name)
        return True

    def load_model(self,
                   saved_model_file_name:str):
        self.model.load_state_dict(torch.load(saved_model_file_name))
        return True

    def get_val_sample_batch(self):
        # notice: dataloader should be shuffled
        data_iter_val = iter(self.dataloader.dataloader_val)
        X_val, y_and_soma = next(data_iter_val)
        y_val, soma_ids = torch.split(y_and_soma, split_size_or_sections=1, dim=1)
        inputs = X_val
        targets = y_val
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)

        targets = targets.squeeze()
        targets = targets.to(torch.float32)
        return inputs, targets

    ## get validation loss value for each batch
    def _train_basic_batch_level_validation(self):
        ## loss value for each epoch
        epoch_train_loss_list = []
        epoch_val_loss_list = []

        ## loss value for each batch in each epoch
        all_batch_train_loss_list = []
        all_batch_val_loss_list = []

        start_time = time.perf_counter()
        print("Start training")
        logging.info("Start training")
        for epoch in range(self.epochs):
            total_train_loss = 0
            iter_num = 0
            train_samples_num = 0
            total_val_loss = 0
            for inputs, labels_soma in self.dataloader.dataloader_train:
                if iter_num == 0:
                    print("Inside the Training Iteration loop")
                    logging.info("Inside the Training Iteration loop")

                ########## Training phase  ############
                self.model.train() ## re-set to train
                # print(f"Process batch {iter_num}")
                targets, soma_ids = torch.split(labels_soma, split_size_or_sections=1, dim=1)
                self.optimizer.zero_grad()
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                targets = targets.to(self.device)

                targets = targets.squeeze()
                targets = targets.to(torch.float32)
                outputs = outputs.squeeze()
                loss = self.criterion(outputs, targets) + self.model.loss()
                loss.backward()
                self.optimizer.step()
                all_batch_train_loss_list.append(loss.item())
                total_train_loss += loss.item() * inputs.size(0)
                iter_num += 1
                train_samples_num += inputs.size(0) ## TODO: check
                if iter_num % 100 ==1:
                    end_time = time.perf_counter()
                    logging.info(f"accumulated time relapsed for iteration {iter_num}: {end_time - start_time} seconds (training stage)")
                    logging.info(f"training loss: {loss.item()}")

                if self.validation_during_training:
                    ############# evaluation phase for each batch #################
                    self.model.eval() ## re-set to eval
                    with torch.no_grad():
                        targets = self.val_y_batch
                        inputs = inputs.to(self.device)
                        outputs = self.model(self.val_X_batch)
                        targets = targets.to(self.device)
                        outputs = outputs.squeeze()
                        targets = targets.squeeze()
                        targets = targets.to(torch.float32)
                        loss = self.criterion(outputs, targets) + self.model.loss()
                        all_batch_val_loss_list.append(loss.item())
                        total_val_loss += loss.item() * inputs.size(0)
                        if iter_num % 100 == 1:
                            end_time = time.perf_counter()
                            logging.info(f"accumulated time relapsed for iteration {iter_num}: {end_time - start_time} seconds (validation stage)")
                            logging.info(f"validation loss: {loss.item()}")

                ############### end of the loop when reaching train_batch_iter_max ###############
                ## it will take too long to fully iterate the whole batches
                ## only sampling maximal train_batch_iter_max batches for the training process
                if not self.train_batch_iter_max is None:
                    if iter_num >= self.train_batch_iter_max:
                        break
            ## end of batch loop
            end_time = time.perf_counter()
            print(f"accumulated time relapsed for epoch {epoch} : {end_time - start_time} seconds (training stage)")
            logging.info(
                f"accumulated time relapsed for epoch {epoch} : {end_time - start_time} seconds (training stage)")
            # Calculate average training loss (use total number of samples)
            avg_train_loss = total_train_loss / train_samples_num
            print(f"Epoch {epoch + 1}/{self.epochs}, Training Loss: {avg_train_loss:.4f}")
            logging.info(f"Epoch {epoch + 1}/{self.epochs}, Training Loss: {avg_train_loss:.4f}")
            epoch_train_loss_list.append(avg_train_loss)


            if self.validation_during_training:
                print(f"training for epoch {epoch} completed, starting model validation")
                logging.info(f"training for epoch {epoch} completed, starting model validation")
                # Calculate average validation loss (use total number of samples)
                avg_val_loss = total_val_loss / train_samples_num
                epoch_val_loss_list.append(avg_val_loss)
                print(f"Epoch {epoch + 1}/{self.epochs}, Validation Loss: {avg_val_loss:.4f}")
                logging.info(f"Epoch {epoch + 1}/{self.epochs}, Validation Loss: {avg_val_loss:.4f}")

                # Update the learning rate scheduler
                self.scheduler.step(avg_val_loss)
            else:
                print("warning: the validation_during_training is set to be False, and the learning rate scheduler is not used.")
        ## end of epoch loop
        end_time = time.perf_counter()
        print(f"accumulated time relapsed for the model training: {end_time - start_time} seconds (training stage)")
        logging.info(f"accumulated time relapsed for the model training: {end_time - start_time} seconds (training stage)")

        return epoch_train_loss_list, epoch_val_loss_list, all_batch_train_loss_list, all_batch_val_loss_list

    ## making prediction based on the trained MLP model
    def _predict_basic(self, ):
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
        self.model.eval()
        predictions = []
        targets_all = []
        soma_ids_all = []
        total_loss = 0
        with torch.no_grad():
            iter_num = 0
            test_samples_num = 0
            for inputs, labels_soma in predict_dataloader:
                targets, soma_ids = torch.split(labels_soma, split_size_or_sections=1, dim=1)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                outputs = outputs.squeeze()
                targets = targets.squeeze()
                targets = targets.to(torch.float32)
                loss = self.criterion(outputs, targets) + self.model.loss()
                total_loss += loss.item() * inputs.size(0)
                predictions.extend(outputs.cpu().numpy())
                targets_all.extend(targets.cpu().numpy())
                soma_ids_all.extend(soma_ids.cpu().numpy())
                test_samples_num += inputs.size(0)
                iter_num += 1
                if not self.predict_batch_iter_max is None:
                    if iter_num > self.predict_batch_iter_max:
                        break

        avg_loss = total_loss / test_samples_num
        soma_ids_all = np.array(soma_ids_all).flatten()
        return predictions, targets_all, soma_ids_all, avg_loss


class TorchElasticNetRegression(nn.Module):
    def __init__(self,
                 cat_cardinalities,
                 num_numeric_features,
                 n_embed=4,
                 l1_lambda=0.1,
                 l2_lambda=0.1,
                 cat_feature_importance_method: str = "max",  # max, mean, sum
                 output_dim=1):
        super(TorchElasticNetRegression, self).__init__()

        self.n_cats = len(cat_cardinalities)

        self.embed_dim_total = self.n_cats * n_embed
        self.cat_feature_importance_method = cat_feature_importance_method

        # Embedding layers for categorical inputs
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, n_embed) for num_categories in cat_cardinalities
        ])
        # Input size to first hidden layer: embeddings + numeric features
        input_size = len(cat_cardinalities) * n_embed + num_numeric_features

        self.linear = nn.Linear(input_size, output_dim)  # Linear layer
        self.l1_lambda = l1_lambda  # L1 regularization coefficient
        self.l2_lambda = l2_lambda  # L2 regularization coefficient

    def forward(self, x):
        x_cat = x[:, :len(self.embeddings)].long()  # First 4 columns: categorical
        x_num = x[:, len(self.embeddings):]         # Rest: numeric

        # Embed each categorical feature
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embedded, dim=1)  # Shape: (batch_size, 4 * n_embed)

        # Combine embedded categorical and numeric features
        x_combined = torch.cat([x_embed, x_num], dim=1)

        return self.linear(x_combined)

    def loss(self):
        # L1 regularization term
        l1_loss = self.l1_lambda * torch.norm(self.linear.weight, p=1)
        # L2 regularization term
        l2_loss = self.l2_lambda * torch.norm(self.linear.weight, p=2)
        return l1_loss +  l2_loss
