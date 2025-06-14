import os
import glob
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from .visualization import plot_age_prediction_performance as papp
from .visualization import plot_catboost_eval
from .utility import donor_level_test, check_model_device, check_tensor_device
import time
import pickle
import joblib

## load different models
from .model.GatedMultiheadAttention import GatedMultiheadAttentionAgeClock as GMA
from .model.GatedMultiheadAttention import GatedMultiheadAttentionFCNet as GMANet
from .model.CatBoost import CatBoostAgeClock
from .model.Linear import TorchElasticNetAgeClock
from .model.MLP import MLPAgeClock
from .model.XGBoost import XGBoostAgeClock


# pipeline for model training
# TODO: optimize ad_dir_root and meta_file_path default settings
# /mnt/DB/gangcai/database/public_db/CZCELLxGENE/whole_datasets/CZCELLxGENE_Human_All/normal/select_protein_coding_genes/H5AD_CountsNormalized_ProteinCoding/
# /mnt/DB/gangcai/database/public_db/CZCELLxGENE/whole_datasets/CZCELLxGENE_Human_All/normal/metadata/meta_testdata.parquet
# /mnt/DB/gangcai/database/public_db/CZCELLxGENE/whole_datasets/CZCELLxGENE_Human_All/normal/metadata/meta_valdata.parquet
# TODO: change the saving method for xgboost
def training_pipeline(model_name: str = "GMA",
                      dataset_folder_dict: dict | None = None,
                      feature_size: int = 19183, # 19031,
                      out_root_dir: str = "./",
                      suffix: str = "pb",
                      run_id: str = "v1",
                      ad_dir_root: str = "./db/",
                      meta_file_path: str = "./db/meta.parquet", # meta file for predict_dataset
                      predict_dataset: str = "validation",
                      model_eval: bool = True, ## whether to evaluation of the model, such as based on validation datasets or testing datasets
                      validation_during_training: bool = True,
                      batch_size_train: int = 1024,
                      train_batch_iter_max:  int | None = 1000,
                      batch_size_val: int = 10240,
                      batch_size_test: int = 10240,
                      l1_lambda: float = 0.01,
                      l2_lambda: float = 0.01,
                      predict_batch_iter_max:  int | None = 20,
                      learning_rate: float = 0.001,
                      nn_weight_decay: float = 0.0001,  # weight decay for neural network
                      epochs: int = 1,
                      loader_method: str = "scageclock",
                      num_workers: int = 1,
                      device: str = "cuda",
                      loss: str = "MSE",
                      model_save_method: str = "stat_dict",
                      # supports one of [stat_dict, pkl, joblib, cbm] , cbm (for catboost)
                      get_feature_importance: bool = False,
                      boost_depth: int = 6,
                      boost_iteration: int = 100,
                      cat_used_ram_limit: str = "100GB",
                      cat_n_embed: int = 4,  ## number of embedding for the categorical feature, only used in eGMA
                      K_fold_mode: bool = False,
                      K_fold_train: tuple[str] = ("Fold1", "Fold2", "Fold3", "Fold4"),
                      K_fold_val: str = "Fold5",
                      **kwargs
                      ):
    start_time = time.time()

    if (device == 'cuda') and (not torch.cuda.is_available()):
        if torch.backends.mps.is_available():
            # exclude xgboost and catboost to use mps device
            if model_name in ["GMA","linear","MLP"]:
                print("warning: cuda is not available, and found MAC mps, which is used instead")
                device = 'mps'
            else:
                device = 'cpu'
        else:
            print("warning: cuda is not available, and the cpu is used instead")
            device = "cpu"


    # default value for dataset_folder_dict if it is None
    if K_fold_mode and (dataset_folder_dict is None):
        dataset_folder_dict = {"training_validation": "train_val"}
    elif dataset_folder_dict is None:
        dataset_folder_dict = {"training": "train", "validation": "val", "testing": "test"}

    ## checking the inputs
    inputs_checker(dataset_folder_dict,
                   ad_dir_root=ad_dir_root,
                   meta_file_path=meta_file_path,
                   K_fold_mode=K_fold_mode)

    ## checking the models
    available_models = list_available_models(print_model_name=False)
    if model_name not in available_models:
        print(f"available models: {available_models}")
        raise ValueError(f"Input model not supported: {model_name}!")

    #settings
    outdir = os.path.join(out_root_dir,f"{model_name}_out_{suffix}/")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
        print(f"create folder {outdir}")
    prefix = f"{model_name}_{suffix}_{run_id}"

    log_file = os.path.join(outdir, prefix + "_log.txt")
    if os.path.exists(log_file):
        os.remove(log_file)

    print(f"h5ad file root directory: {ad_dir_root}")

    # load the aging clock model
    print("loading the aging clock model")
    if model_name == "GMA":
        age_clock = GMA(anndata_dir_root=ad_dir_root,
                        dataset_folder_dict=dataset_folder_dict,
                        predict_dataset=predict_dataset,
                        validation_during_training=validation_during_training,
                        feature_size=feature_size,
                        n_embed=cat_n_embed,
                        batch_size_train=batch_size_train,
                        train_batch_iter_max=train_batch_iter_max,
                        batch_size_val=batch_size_val,
                        batch_size_test=batch_size_test,
                        l1_lambda=l1_lambda,
                        l2_lambda=l2_lambda,
                        predict_batch_iter_max=predict_batch_iter_max,
                        epochs=epochs,
                        loader_method=loader_method,
                        num_workers=num_workers,
                        device=device,
                        log_file=log_file,
                        learning_rate=learning_rate,
                        weight_decay=nn_weight_decay,
                        K_fold_mode=K_fold_mode,
                        K_fold_train=K_fold_train,
                        K_fold_val=K_fold_val,
                        **kwargs)
    elif model_name == "catboost":
        if device == "cuda":
            device = "GPU"
        elif device == "gpu":
            device = "GPU"
        else:
            device = None
        age_clock = CatBoostAgeClock(anndata_dir_root=ad_dir_root,
                                     dataset_folder_dict=dataset_folder_dict,
                                     predict_dataset=predict_dataset,
                                     validation_during_training=validation_during_training,
                                     batch_size_train=batch_size_train,
                                     train_batch_iter_max=train_batch_iter_max,
                                     batch_size_val=batch_size_val,
                                     batch_size_test=batch_size_test,
                                     predict_batch_iter_max=predict_batch_iter_max,
                                     task_type=device,
                                     loader_method=loader_method,
                                     learning_rate=learning_rate,
                                     log_file=log_file,
                                     depth=boost_depth,
                                     used_ram_limit=cat_used_ram_limit,
                                     iterations=boost_iteration,
                                     K_fold_mode=K_fold_mode,
                                     K_fold_train=K_fold_train,
                                     K_fold_val=K_fold_val,
                                     **kwargs
                                     )
    elif model_name == "xgboost":
        age_clock = XGBoostAgeClock(anndata_dir_root=ad_dir_root,
                                    dataset_folder_dict=dataset_folder_dict,
                                    predict_dataset=predict_dataset,
                                    validation_during_training=validation_during_training,
                                    batch_size_train=batch_size_train,
                                    train_batch_iter_max=train_batch_iter_max,
                                    batch_size_val=batch_size_val,
                                    batch_size_test=batch_size_test,
                                    predict_batch_iter_max=predict_batch_iter_max,
                                    device=device,
                                    loader_method=loader_method,
                                    learning_rate=learning_rate,
                                    log_file=log_file,
                                    max_depth=boost_depth,
                                    n_estimators=boost_iteration,
                                    K_fold_mode=K_fold_mode,
                                    K_fold_train=K_fold_train,
                                    K_fold_val=K_fold_val,
                                    **kwargs
                                    )
    elif model_name == "linear":
        age_clock = TorchElasticNetAgeClock(anndata_dir_root=ad_dir_root,
                                            dataset_folder_dict=dataset_folder_dict,
                                            predict_dataset=predict_dataset,
                                            validation_during_training=validation_during_training,
                                            feature_size=feature_size,
                                            batch_size_train=batch_size_train,
                                            train_batch_iter_max=train_batch_iter_max,
                                            batch_size_val=batch_size_val,
                                            batch_size_test=batch_size_test,
                                            l1_lambda=l1_lambda,
                                            l2_lambda=l2_lambda,
                                            predict_batch_iter_max=predict_batch_iter_max,
                                            epochs=epochs,
                                            loader_method=loader_method,
                                            num_workers=num_workers,
                                            device=device,
                                            log_file=log_file,
                                            learning_rate=learning_rate,
                                            weight_decay=nn_weight_decay,
                                            K_fold_mode=K_fold_mode,
                                            K_fold_train=K_fold_train,
                                            K_fold_val=K_fold_val,
                                            **kwargs)
    elif model_name == "MLP":
        age_clock = MLPAgeClock(anndata_dir_root=ad_dir_root,
                                dataset_folder_dict=dataset_folder_dict,
                                predict_dataset=predict_dataset,
                                validation_during_training=validation_during_training,
                                feature_size=feature_size,
                                batch_size_train=batch_size_train,
                                train_batch_iter_max=train_batch_iter_max,
                                batch_size_val=batch_size_val,
                                batch_size_test=batch_size_test,
                                predict_batch_iter_max=predict_batch_iter_max,
                                epochs=epochs,
                                loader_method=loader_method,
                                num_workers=num_workers,
                                device=device,
                                log_file=log_file,
                                learning_rate=learning_rate,
                                weight_decay=nn_weight_decay,
                                K_fold_mode=K_fold_mode,
                                K_fold_train=K_fold_train,
                                K_fold_val=K_fold_val,
                                **kwargs)
    else:
        raise ValueError(f"Input model not supported: {model_name}!")

    # record time elapsed
    end_time = time.time()
    print(f"Time elapsed for model loading: {end_time - start_time} seconds")

    # train the model
    age_clock.train()

    # record time elapsed
    end_time = time.time()
    print(f"Time elapsed for training: {end_time - start_time} seconds")

    ###### plot the training and validation loss values #########
    if validation_during_training:
        print("start training loss values generation on training datasets and validation datasets")
        if model_name in ["GMA","linear","MLP"]:
            train_steps = np.arange(len(age_clock.batch_train_loss_list))
            train_labels = ["train"] * len(age_clock.batch_train_loss_list)
            val_labels = ["validation"] * len(age_clock.batch_val_loss_list)
            train_df = pd.DataFrame({"label": train_labels + val_labels,
                                     "steps": list(train_steps) + list(train_steps),
                                     loss: age_clock.batch_train_loss_list + age_clock.batch_val_loss_list})

            ## save training loss to local file
            train_df.to_csv(os.path.join(outdir, f"{prefix}_traning_process_{loss}.tsv"),
                            sep="\t",
                            index=False)

            # Create a figure for better control over the output
            plt.figure(figsize=(5, 3))
            sns.scatterplot(data=train_df, x="steps", y=loss, hue="label")
            # Adjust layout and legend to prevent clipping
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Legend outside plot
            plt.tight_layout()

            # Save high-quality image
            plt.savefig(os.path.join(outdir, f"{prefix}_loss_values_comparison.png"),
                        dpi=300,  # High resolution
                        bbox_inches='tight',  # Prevent legend cutoff
                        transparent=False)  # White background

            # Clear the figure to free memory
            plt.close()

        if model_name in ["catboost","xgboost"]:
            eval_metrics_df = age_clock.eval_metrics
            eval_metrics_df.to_csv(os.path.join(outdir, f"{prefix}_traning_process_{loss}.tsv"),
                                   sep="\t",
                                   index=False)
            plot_catboost_eval(eval_metrics_df,
                               save_path=os.path.join(outdir, f"{prefix}_loss_values_comparison.png"))


    ### saving the model #####
    print("start saving the model")
    if model_name == "xgboost":
        model_save_method = "bin"
        print(f"{model_name} only support bin model saving")
    elif model_name == "catboost":
        model_save_method = "cbm"
        print(f"{model_name} only support cbm model saving")
    else:
        if model_save_method not in ["stat_dict","pkl","joblib"]:
            model_save_method = "stat_dict"
            print(f"neural network model supports one of stat_dict, pkl or joblib format model saving")

    if model_save_method == "stat_dict":
        saved_model_file_name = os.path.join(outdir, prefix + ".pth")
        torch.save(age_clock.model.state_dict(), saved_model_file_name)
    elif model_save_method == "pkl":
        with open(os.path.join(outdir,f"{prefix}.pkl"), 'wb') as file:
            pickle.dump(age_clock, file)
    elif model_save_method == "cbm": # For Catboost
        age_clock.model.save_model(os.path.join(outdir, f"{prefix}.cbm"))
    elif model_save_method == "bin":
        age_clock.model.save_model(os.path.join(outdir, f"{prefix}.bin"))
    elif model_save_method == "joblib":
        # joblib.dump(age_clock, os.path.join(outdir,f"{prefix}.joblib")) # don't save the whole age_clock, which can be huge
        joblib.dump(age_clock.model, os.path.join(outdir,f"{prefix}.joblib"))
    else:
        print(f"Warning: only [stat_dict, pkl, joblib, cbm] are supported for model_save_method parameter settings")

    ### model evaluation based on predict_dataset####
    if model_eval:
        print("start model evaluation")
        print(f"datasets used for cell_level_test: {predict_dataset}")
        cell_level_eval_metrics_dict, y_eval_pred, y_eval_true, soma_ids_all = age_clock.cell_level_test()
        print(f"cell id check: {soma_ids_all[:3]}")
        cell_level_m_df = pd.DataFrame(cell_level_eval_metrics_dict, index=[0])
        cell_level_m_df.to_csv(os.path.join(outdir, f"{prefix}_cell_level_evaluation_metrics.tsv"), index=False, sep="\t")

        cell_level_eval_df = pd.DataFrame({"y_eval_true": y_eval_true,
                                           "y_eval_predicted": y_eval_pred,
                                           "cell_id": soma_ids_all, })

        cell_level_eval_df.to_csv(os.path.join(outdir, f"{prefix}_cell_level_evaluation_predictions.tsv"),sep="\t")

        donor_true_ages, donor_predicted_ages, donor_level_eval_metrics_dict = donor_level_test(meta_file_path=meta_file_path,
                                                                                                test_soma_joinids=soma_ids_all,
                                                                                                y_test_true=y_eval_true,
                                                                                                y_test_predict=y_eval_pred,
                                                                                                method="mean")


        donor_level_m_df = pd.DataFrame(donor_level_eval_metrics_dict, index=[0])
        donor_level_m_df.to_csv(os.path.join(outdir, f"{prefix}_donor_level_evaluation_metrics.tsv"), index=False, sep="\t")


        donor_level_eval_df = pd.DataFrame({"y_eval_true": donor_true_ages,
                                               "y_eval_predicted": donor_predicted_ages})

        donor_level_eval_df.to_csv(os.path.join(outdir, f"{prefix}_donor_level_evaluation_predictions.tsv"),sep="\t")


        papp(real_age=donor_true_ages.values,
             predicted_age=donor_predicted_ages.values,
             outdir=outdir,
             method_name=model_name,
             level_type="donor",
             filename=f"{prefix}_donor_level_evaluation_plot.jpg")


    #### getting the feature importance #####
    if get_feature_importance:
        print("start feature importance calculation")
        fi_supported_models = ["xgboost","catboost","GMA"]
        if model_name in fi_supported_models:
            feature_importances = age_clock.get_feature_importance()
            var_file = ad_dir_root + "/h5ad_var.tsv"

            var_df = pd.read_csv(var_file, sep="\t")
            var_df["feature_importance"] = feature_importances
            var_df = var_df.sort_values(by="feature_importance", ascending=False, ignore_index=True)
            var_df.to_csv(os.path.join(outdir, f"{prefix}_featureImportances.tsv"),
                          sep="\t")
        else:
            print(f"feature importance only support {fi_supported_models}")

    end_time = time.time()
    print("Training completed!")
    print(f"Time elapsed for the whole pipeline: {end_time - start_time} seconds")
    return True

## given a single cell matrix inputs (numpy array or tensor), predict the donor/individual age of the cells
## TODO: merge with the prediction function in evaluation
def predict(model,
            inputs,
            device='cpu'):

    if not device in ['cpu',"cuda"]:
        raise ValueError(f"only two devices supported: cpu or cuda")

    if not (isinstance(inputs, np.ndarray) or isinstance(inputs, torch.Tensor)):
        raise  ValueError("inputs type should be either of numpy.ndarray or torch.Tensor")

    # format the dtype of the inputs
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs)
    inputs = inputs.to(torch.float32)

    model_device_name = check_model_device(model)
    if isinstance(inputs,np.ndarray):
        inputs = torch.tensor(inputs)

    inputs_device_name = check_tensor_device(inputs)

    if (device == 'cpu') and (model_device_name == 'cuda'):
        model = model.cpu()
    elif (device == 'cuda') and (model_device_name == 'cpu'):
        model = model.to("cuda")

    if (device == 'cpu') and (inputs_device_name == 'cuda'):
        inputs = inputs.cpu()
    elif (device == 'cuda') and (inputs_device_name == 'cpu'):
        inputs = inputs.to("cuda")

    age_predicted = model(inputs)
    return age_predicted.squeeze().detach()


def get_feature_importance(model,
                           feature_file: str | None = None,
                           feature_colname: str = "h5ad_var"):
    if feature_file is None:
        return model.get_feature_importance()
    else:
        if os.path.exists(feature_file):
            feature_importance = model.get_feature_importance()
            feature_id_df = pd.read_csv(feature_file, sep="\t")
            feature_id_lst = list(feature_id_df[feature_colname])
            results = pd.DataFrame({"feature_name":feature_id_lst,
                                    "feature_importance":feature_importance})
            return results


def load_GMA_model(model_file,
                   model_file_type: str = "pth",
                   cat_cardinalities: list[int] | None = None,
                   num_numeric_features: int = 19179, # 19183-4,
                   hidden_dim: int = 128,
                   l1_lambda: float = 0.001,
                   l2_lambda: float = 0,
                   num_heads: int = 8,):
    if cat_cardinalities is None:
        # ['assay', 'cell_type', 'tissue_general', 'sex'],
        # the cardinalities for each categorical feature column, the first len(cat_car_list) columns
        cat_cardinalities = [21, 664, 52, 3]

    if model_file_type == "pth":
        GMA_model = GMANet(cat_cardinalities=cat_cardinalities,
                           num_numeric_features=num_numeric_features,
                           hidden_dim=hidden_dim,
                           l1_lambda=l1_lambda,
                           l2_lambda=l2_lambda,
                           num_heads=num_heads)
        if torch.backends.mps.is_available():
            print("Mac mps is found, and device is set to be mps")
            device = 'mps'
        elif torch.cuda.is_available():
            print("Cuda is found, and device is set to be cuda")
            device = "cuda"
        else:
            print("warning: both of cuda and mps are not available, and the cpu is used instead")
            device = "cpu"
        GMA_model.load_state_dict(torch.load(model_file, map_location=torch.device(device)))
        return GMA_model
    elif model_file_type == "pkl":
        with open(model_file, 'rb') as file:  # 'rb' stands for read binary
            GMA_model = pickle.load(file)
        return GMA_model
    else:
        raise ValueError(f"{model_file_type} not supported")

def list_available_models(print_model_name=True):
    available_models = ["linear", "xgboost", "catboost", "MLP", "GMA"]
    if print_model_name:
        print(available_models)

    return available_models

def inputs_checker(dataset_folder_dict,
                   ad_dir_root,
                   meta_file_path,
                   K_fold_mode: bool = False,):
    # check the existence of the folders and files
    for sub_folder in dataset_folder_dict.values():
        if not os.path.exists(os.path.join(ad_dir_root, sub_folder)):
            raise ValueError(f"sub folder not exist in the inputs root directory: {sub_folder}!")
        if K_fold_mode:
            sub_folder_h5ad_files = glob.glob(os.path.join(ad_dir_root, f"{sub_folder}/*/*.h5ad"))
        else:
            sub_folder_h5ad_files = glob.glob(os.path.join(ad_dir_root, f"{sub_folder}/*.h5ad"))
        if len(sub_folder_h5ad_files) < 1:
            raise ValueError(f"NO h5ad file under the  {sub_folder}/ in the root directory")

    if not os.path.exists(meta_file_path):
        raise ValueError(f"NO {meta_file_path} found")

    pass

