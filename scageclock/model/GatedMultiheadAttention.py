import os.path

import torch
import torch.nn as nn
from ..dataloader import BasicDataLoader
import time
import logging
from ..utility import get_validation_metrics
import numpy as np

# TODO: add model checkpoint
class GatedMultiheadAttentionAgeClock:

    def __init__(self,
                 anndata_dir_root: str,
                 dataset_folder_dict: dict | None = None,
                 predict_dataset: str = "testing",
                 validation_during_training: bool = True,
                 feature_size: int = 19183, # 19031,
                 cat_card_list: list[int] | None = None,
                 n_embed: int = 4,
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
                 l1_lambda: float = 0.01,
                 l2_lambda: float = 0,
                 dropout_prob: float = 0.2,
                 learning_rate: float = 0.001,
                 epochs: int = 1,
                 weight_decay: float = 0.001,
                 patience: int = 3,
                 scheduler_factor: float = 0.5,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 device: str = "cpu",
                 train_batch_iter_max: int | None = 100,  ## maximal number of iteration for the DataLoader during training
                 predict_batch_iter_max: int | None = 20,
                 initial_model: str | None = None,
                 cat_feature_importance_method: str = "max",  # max, mean, sum
                 balanced_dataloader_parameters: dict | None = None,
                 K_fold_mode: bool = False,
                 K_fold_train: tuple[str] = ("Fold1", "Fold2", "Fold3", "Fold4"),
                 K_fold_val: str = "Fold5",
                 save_checkpoint: bool = False,
                 checkpoint_outdir: str = "./checkpoints_saved", # only used when save_checkpoint is true
                 checkpoint_file_prefix: str = "GMA",
                 log_file: str = "log.txt"):
        """
        Aging Clock based on Gated (Elastic Net) Multi-head Attention Neural Network

        :param anndata_dir_root: root directory that stores model datasets:  h5ad_var.tsv  test/*.h5ad  train/*.h5ad  val/*.h5ad
        :param dataset_folder_dict: the folder name for each type of datasets, default: {"training": "train", "validation": "val", "testing": "test"}
        :param predict_dataset: datasets used for making prediction ,  'validation' or 'testing' (default) or 'training' datasets
        :param validation_during_training: boolean value whether to do validation during training processes, if true, validation loss is recorded during each training batch
        :param feature_size: number of features (categorical features + numeric gene features)
        :param cat_card_list: number of cardinalities for each categorical feature
        :param n_embed: dimension size of the embedding for the categorical feature
        :param var_file_name: file name for the file with the .h5ad shared .var information, with two columns: var name column and var index
        :param var_colname: column name of the var name in var_file_name
        :param batch_size_train: bath size for training DataLoader
        :param batch_size_val: batch size for validation DataLoader
        :param batch_size_test: batch size for testing DataLoader
        :param shuffle: whether to shuffle the DataLoader
        :param num_workers: number of parallel jobs for Data Loading
        :param age_column: age column name in the adata.obs
        :param cell_id:cell id column name in the adata.obs # default using CELLxGENE soma_joinid
        :param loader_method: loader method used: "scageclock" or "scageclock_balanced"
        :param l1_lambda: L1 regularization parameter
        :param l2_lambda: L2 regularization parameter
        :param dropout_prob: dropout probability
        :param learning_rate: learning rate for model training
        :param epochs: number of epochs for model training (normally 1 for large-scale scRNA-seq datasets)
        :param weight_decay:  weight_decay value for Adam optimizer
        :param patience: The number of allowed epochs with no improvement after
                        which the learning rate will be reduced.
        :param scheduler_factor: Factor by which the learning rate will be
                                reduced. new_lr = lr * factor.
        :param hidden_dim: dimension of the hidden layer for the Fully Connected Network (FC)
        :param num_heads: number of heads for Multi-head Attention
        :param device: device used for training: cpu , cuda
        :param train_batch_iter_max: early stop of batch iteration when reaching this number of batches for the training processes, not used if None
        :param predict_batch_iter_max: early stop of batch iteration when reaching this number of batches for the prediction processes, not used if None
        :param initial_model: default None. Load the trained model as the initial model.
        :param balanced_dataloader_parameters: dictionary for h5ad_dataloader BalancedH5ADDataLoader
        :param K_fold_mode: whether to use K_fold mode. Each-fold datasets should be under one folder
        :param K_fold_train: The K_fold folders under ad_files_path that are used for training
        :param K_fold_val: The K_fold folder under ad_files_path that are sued for validation
        :param save_checkpoint: whether to save the checkpoints for each epoch
        :param checkpoint_outdir: path to the outputs of checkpoint files, only works when save_checkpoint is True
        :param checkpoint_file_prefix: prefix for the output files of checkpoint
        :param log_file: log file
        """

        # default cardinalities for each categorical feature column
        if cat_card_list is None:
            # default order of categorical columns: ['assay', 'cell_type', 'tissue_general', 'sex'],
            # the cardinalities for each categorical feature column, the first len(cat_car_list) columns
            # cat_card_list = [14, 219, 39, 3] # old version
            cat_card_list = [21, 664, 52, 3]

        # default value for dataset_folder_dict if it is None
        if K_fold_mode and (dataset_folder_dict is None):
            dataset_folder_dict = {"training_validation": "train_val"}
        elif dataset_folder_dict is None:
            dataset_folder_dict = {"training": "train", "validation": "val", "testing": "test"}

        self.anndata_dir_root = anndata_dir_root
        self.dataset_folder_dict = dataset_folder_dict
        self.predict_dataset = predict_dataset
        self.validation_during_training = validation_during_training
        self.feature_size = feature_size
        self.cat_card_list = cat_card_list
        self.n_embed = n_embed
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
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.scheduler_factor = scheduler_factor
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.epochs = epochs
        self.device = device
        self.input_dim = self.feature_size # set input dim as feature size
        self.train_batch_iter_max = train_batch_iter_max
        self.predict_batch_iter_max = predict_batch_iter_max

        self.cat_feature_importance_method = cat_feature_importance_method

        self.K_fold_train = K_fold_train
        self.K_fold_val = K_fold_val

        ## training loss metrics
        self.epoch_train_loss_list = []
        self.epoch_val_loss_list = []
        self.batch_train_loss_list = []
        self.batch_val_loss_list = []
        self.initial_model = initial_model
        self.log_file = log_file


        self.save_checkpoint = save_checkpoint
        self.checkpoint_outdir = checkpoint_outdir
        self.checkpoint_file_prefix = checkpoint_file_prefix

        if self.save_checkpoint:
            if not os.path.exists(self.checkpoint_outdir):
                os.makedirs(self.checkpoint_outdir)

        self.balanced_dataloader_parameters = balanced_dataloader_parameters

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
                                          age_column=self.age_column,
                                          cell_id=self.cell_id,
                                          loader_method=self.loader_method,
                                          dataset_folder_dict=self.dataset_folder_dict,
                                          balanced_dataloader_parameters=self.balanced_dataloader_parameters,
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
        self.model = GatedMultiheadAttentionFCNet(cat_cardinalities=self.cat_card_list,
                                                  num_numeric_features= feature_size-len(self.cat_card_list),
                                                  n_embed=self.n_embed,
                                                  hidden_dim=self.hidden_dim,
                                                  l1_lambda=self.l1_lambda,
                                                  l2_lambda=self.l2_lambda,
                                                  num_heads=self.num_heads,
                                                  cat_feature_importance_method=self.cat_feature_importance_method).to(self.device)

        if self.initial_model:
            self.model.load_state_dict(torch.load(self.initial_model))
            print(f"initial model :{self.initial_model} is used.")
            logging.info(f"initial model :{self.initial_model} is used.")


        self.criterion = nn.MSELoss()  # Mean squared error for regression
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate,
                                          weight_decay=self.weight_decay)

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
        ## loss value for each batch in each epoch
        all_batch_train_loss_list = []

        epoch_val_loss_list = []
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
                loss = self.criterion(outputs, targets) + self.model.feature_gate.loss() ## use total loss
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
                    ############# validation phase for each batch #################
                    self.model.eval() ## re-set to eval
                    with torch.no_grad():
                        targets = self.val_y_batch
                        inputs = inputs.to(self.device)
                        outputs = self.model(self.val_X_batch)
                        targets = targets.to(self.device)
                        outputs = outputs.squeeze()
                        targets = targets.squeeze()
                        targets = targets.to(torch.float32)
                        loss = self.criterion(outputs, targets) + self.model.feature_gate.loss() ## use total loss
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
            print(f"training for epoch {epoch} completed")
            print(f"number of batches iterated: {iter_num}")


            logging.info(f"training for epoch {epoch} completed")
            logging.info(f"number of batches iterated: {iter_num}")
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

            if self.save_checkpoint:
                cp_file = os.path.join(self.checkpoint_outdir, f"{self.checkpoint_file_prefix}_epoch{epoch}.pth" )
                # saving the checkpoint
                torch.save({'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': self.optimizer.state_dict(),
                            'train_avg_loss': avg_train_loss},
                           cp_file)
        ## end of epoch loop
        end_time = time.perf_counter()
        print(f"accumulated time relapsed for the model training: {end_time - start_time} seconds (training stage)")
        logging.info(f"accumulated time relapsed for the model training: {end_time - start_time} seconds (training stage)")

        return epoch_train_loss_list, epoch_val_loss_list, all_batch_train_loss_list, all_batch_val_loss_list

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
                loss = self.criterion(outputs, targets) + self.model.feature_gate.loss() ## use total loss
                total_loss += loss.item() * inputs.size(0)
                predictions.extend(outputs.cpu().numpy())
                targets_all.extend(targets.cpu().numpy())
                soma_ids_all.extend(soma_ids.cpu().numpy())
                test_samples_num += inputs.size(0)
                iter_num += 1
                if not self.predict_batch_iter_max is None:
                    if iter_num >= self.predict_batch_iter_max:
                        break

        avg_loss = total_loss / test_samples_num

        soma_ids_all = np.array(soma_ids_all).flatten()
        return predictions, targets_all, soma_ids_all, avg_loss

    def get_feature_importance(self):
        return self.model.get_feature_importance()


class GatedMultiheadAttentionFCNet(nn.Module):
    def __init__(self,
                 cat_cardinalities,
                 num_numeric_features,
                 n_embed=4,
                 hidden_dim = 256,
                 embed_dim=512,
                 l1_lambda=0.1,
                 l2_lambda=0.1,
                 num_heads=8,
                 cat_feature_importance_method: str = "max",  # max, mean, sum
                 ):
        super().__init__()

        self.n_cats = len(cat_cardinalities)

        self.embed_dim_total = self.n_cats * n_embed
        self.cat_feature_importance_method = cat_feature_importance_method

        # Embedding layers for categorical inputs
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, n_embed) for num_categories in cat_cardinalities
        ])
        # Input size to first hidden layer: embeddings + numeric features
        input_size = len(cat_cardinalities) * n_embed + num_numeric_features


        self.feature_gate = FeatureGate(input_size,
                                        l1_lambda=l1_lambda,
                                        l2_lambda=l2_lambda)
        self.projection = nn.Linear(input_size, embed_dim)
        self.attention = MultiheadAttention(embed_dim, num_heads)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.norm = nn.LayerNorm(embed_dim)  # Normalization for stability

    def forward(self, x):
        x_cat = x[:, :len(self.embeddings)].long()  # First 4 columns: categorical
        x_num = x[:, len(self.embeddings):]         # Rest: numeric

        # Embed each categorical feature
        embedded = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x_embed = torch.cat(embedded, dim=1)  # Shape: (batch_size, 4 * n_embed)

        # Combine embedded categorical and numeric features
        x_combined = torch.cat([x_embed, x_num], dim=1)

        x_combined = self.feature_gate(x_combined)  # Feature selection
        x_combined = self.projection(x_combined)  # Dimensionality reduction
        x_combined = self.norm(x_combined)  # Normalize features
        x_combined = self.attention(x_combined.unsqueeze(1)).squeeze(1)  # Attention module
        return self.fc(x_combined)  # Regression output

    def get_feature_importance(self):
        with torch.no_grad():
            weight_abs = self.feature_gate.weights.detach().abs().cpu()

            # Categorical part
            cat_importance = []
            for i in range(self.n_cats):
                start = i * self.embeddings[i].embedding_dim
                end = start + self.embeddings[i].embedding_dim
                if self.cat_feature_importance_method == "sum":
                    score = weight_abs[start:end].sum()
                elif self.cat_feature_importance_method == "max":
                    score = weight_abs[start:end].max()
                elif self.cat_feature_importance_method == "mean":
                    score = weight_abs[start:end].mean()
                else:
                    raise ValueError(f"{self.cat_feature_importance_method} is not in ['sum', 'max','mean']")
                cat_importance.append(score.item())

            # Numeric part
            start_num = self.embed_dim_total
            num_importance = weight_abs[start_num:].tolist()

        return cat_importance + num_importance

    def get_feature_importance_full(self):
        with torch.no_grad():
            weight_abs = self.feature_gate.weights.detach().abs().cpu()

            # Categorical part
            cat_importance = []
            for i in range(self.n_cats):
                start = i * self.embeddings[i].embedding_dim
                end = start + self.embeddings[i].embedding_dim
                if self.cat_feature_importance_method == "sum":
                    score = weight_abs[start:end].sum()
                elif self.cat_feature_importance_method == "max":
                    score = weight_abs[start:end].max()
                elif self.cat_feature_importance_method == "mean":
                    score = weight_abs[start:end].mean()
                else:
                    raise ValueError(f"{self.cat_feature_importance_method} is not in ['sum', 'max','mean']")
                cat_importance.append(score.item())

            # Numeric part
            start_num = self.embed_dim_total
            num_importance = weight_abs[start_num:].tolist()

        return cat_importance, num_importance, weight_abs.numpy()

"""
Pure L1: Set l2_lambda=0.
Pure L2: Set l1_lambda=0. ## not suggested for sparse feature selection
Elastic: Set both l1_lambda and l2_lambda to positive values.
"""
class FeatureGate(nn.Module):
    def __init__(self, input_dim, l1_lambda=0.1, l2_lambda=0.1):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(input_dim, requires_grad=True))
        self.l1_lambda = l1_lambda  # L1 regularization strength
        self.l2_lambda = l2_lambda  # L2 regularization strength

    def forward(self, x):
        return x * self.weights

    def loss(self):
        # L1 regularization term
        l1_loss = self.l1_lambda * torch.norm(self.weights, p=1)
        # L2 regularization term
        l2_loss = self.l2_lambda * torch.norm(self.weights, p=2)
        # Total loss combines both terms (elastic regularization if both are non-zero)
        return l1_loss + l2_loss

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn_weights = None  # Store attention weights

    def forward(self, x):
        x, attn_weights = self.attn(x, x, x)  # Compute attention
        return x
