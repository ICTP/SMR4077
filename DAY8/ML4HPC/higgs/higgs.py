# Import all the relevant dependencies
import os
from xgboost import dask as dxgb
from dask_ml.model_selection import train_test_split
import numpy as np
import numba, socket, time
import dask, dask_cudf
import xgboost as xgb
from dask.distributed import Client, wait
import dask.array as da
from cuml.metrics import confusion_matrix, accuracy_score
import cupy as cp
import cudf
import pandas as pd
import cupy as cp
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import classification_report
import warnings
from cuml.metrics import roc_auc_score
warnings.filterwarnings('ignore')
matplotlib.use("Agg")


# Define a function that, given the dataset dask_cudf.core.DataFrame, returns a train and validation sets for the training with XGBoost
def load_higgs(client, ddf) -> tuple[dask_cudf.core.DataFrame,dask_cudf.core.Series,dask_cudf.core.DataFrame,dask_cudf.core.Series,]:
    y = ddf["process"]
    X = ddf[ddf.columns.difference(["process"])]
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=False)
    X_train, X_valid, y_train, y_valid = client.persist([X_train, X_valid, y_train, y_valid])
    wait([X_train, X_valid, y_train, y_valid])
    return X_train, X_valid, y_train, y_valid

# Given the train and validation sets, we train the XGBoost using a binary:logistic loss
# By setting eval_metric='error', we can monitor the model's performance during training and enable early stopping to prevent overfitting. 
# This metric is particularly useful when you want to minimize the overall misclassification rate.
def fit_model_es(client, X, y, X_valid, y_valid) -> dxgb.Booster:
    # Firstly, we specify the number of rounds to trigger early stopping for training.
    # XGBoost will stop the training once the validation metric fails to improve in consecutive X rounds, where X is the number of rounds specified for early stopping.
    # early_stopping_rounds=10 to enable early stopping if the error rate doesn't improve for 10 consecutive rounds.
    early_stopping_rounds = 10 
    # Secondly, we use a data type called DaskDeviceQuantileDMatrix for training but DaskDMatrix for validation.
    # DaskDeviceQuantileDMatrix is a drop-in replacement of DaskDMatrix for GPU-based training inputs that avoids extra data copies.
    # DaskDMatrix acts like normal DMatrix, works as a proxy for local DMatrix scatter around workers.
    Xy_valid = dxgb.DaskDMatrix(client, X_valid, y_valid)
    # DaskDeviceQuantileDMatrix is a data type specialized for gpu_hist, tree method that reduces memory overhead. 
    # When training on GPU pipeline, it'spreferred over DaskDMatrix
    Xy = dxgb.DaskDeviceQuantileDMatrix(client, X, y)
    # DaskDeviceQuantileDMatrix is used instead of DaskDMatrix, be careful that it can not be used for anything else other than training.
    # train the model
    # Use train method from xgboost.dask instead of xgboost.  
    # This distributed version of train returns a dictionary containing the resulting booster and evaluation history obtained from evaluation metrics.
    # XGBoost has 3 builtin tree methods, namely exact, approx and hist
    # exact:
    # During split-finding, it iterates over all entries of input data. 
    # It's more accurate (among other greedy methods) but computationally slower in compared to other tree methods.
    # Features like distributed training and external memory that require approximated quantiles are not supported. 
    # As exact tree method is slow in computation performance and difficult to scale, we often employ approximated training algorithms.
    # approx: 
    # An approximation tree method that runs sketching before building each tree using all the rows (rows belonging to the root).
    # Hessian is used as weights during sketch. 
    # hist: 
    # An approximation tree method that runs sketching before training using only user provided weights instead of hessian.
    # The subsequent per-node histogram is built upon this global sketch.
    # This is the fastest algorithm as it runs sketching only once. 
    booster = dxgb.train(
        client,
        {"objective": "binary:logistic", "eval_metric": "error", "tree_method": "hist", "device" : "cuda",},
        Xy,
        evals=[(Xy_valid, "Valid")],
        num_boost_round=2000, # Number of boosting iterations
        early_stopping_rounds=early_stopping_rounds,
    )["booster"]
    return booster


# Train with Customized objective and evaluation metric
# In the example below the XGBoost model is trained using
# - a custom logistic regression-based objective function (logit) and
# - a custom evaluation metric (error) along with early stopping.
# Note that the function returns both gradient and hessian, which XGBoost uses to optimize the model.
# Also, the parameter named metric_name needs to be specified in our callback.
# It is used to inform XGBoost that the custom error function should be used for evaluating early stopping criteria.
def fit_model_customized_objective(client, X, y, X_valid, y_valid) -> dxgb.Booster:
    def logit(predt: np.ndarray, Xy: xgb.DMatrix) -> tuple[np.ndarray, np.ndarray]:
        predt = 1.0 / (1.0 + np.exp(-predt))
        labels = Xy.get_label()
        grad = predt - labels
        hess = predt * (1.0 - predt)
        return grad, hess

    def error(predt: np.ndarray, Xy: xgb.DMatrix) -> tuple[str, float]:
        label = Xy.get_label()
        r = np.zeros(predt.shape)
        predt = 1.0 / (1.0 + np.exp(-predt))
        gt = predt > 0.5
        r[gt] = 1 - label[gt]
        le = predt <= 0.5
        r[le] = label[le]
        return "CustomErr", float(np.average(r))

    # Use early stopping with custom objective and metric.
    early_stopping_rounds = 10
    # Specify the metric we want to use for early stopping.
    es = xgb.callback.EarlyStopping(rounds=early_stopping_rounds, save_best=True, metric_name="CustomErr")
    Xy = dxgb.DaskDeviceQuantileDMatrix(client, X, y)
    Xy_valid = dxgb.DaskDMatrix(client, X_valid, y_valid)
    booster = dxgb.train(
        client,
        {"eval_metric": "error", "tree_method": "hist", "device" : "cuda"},
        Xy,
        evals=[(Xy_valid, "Valid")],
        num_boost_round=2000, # Number of boosting iterations
        obj=logit,  # pass the custom objective
        custom_metric=error,  # pass the custom metric
        callbacks=[es],
    )["booster"]
    return booster

# Running inference
# After some tuning, we arrive at the final model for performing inference on new data.
def predict(client, model, X):
    predt = dxgb.predict(client, model, X)
    return predt


def _sensitivity(y_actual, y_pred):
    """Calculate the sensitivity score per class for a model"""
    cm = confusion_matrix(y_actual, y_pred)
    FN = cm[1, 0].item()
    TP = cm[1, 1].item()
    if(not TP): # avoid Nan values if both TP and FN are 0s
        sensitivity = 0.0
    else:
        sensitivity = round((TP / (TP + FN)), 5)
    return sensitivity


def _specificity(y_actual, y_pred):
    """Calculate the specificity score per class for a model"""
    cm = confusion_matrix(y_actual, y_pred)
    TN = cm[0, 0].item()
    FP = cm[0, 1].item()
    if(not TN): # avoid Nan values if both TN and FP are 0s
        specificity = 0.0
    else:
        specificity = round((TN / (TN + FP)), 5)
    return specificity


def report(y_actual, y_pred, filename):
    """Print a report with all evaluation metrics for a model"""
    print(classification_report(y_actual.get(), y_pred.get()))
    sensitivity = _sensitivity(y_actual, y_pred)
    print('Sensitivity: ', sensitivity)
    specificity = _specificity(y_actual, y_pred)
    print('Specificity: ', specificity)
    cm_matrix = confusion_matrix(y_actual,y_pred)
    sns_plot=sns.heatmap(cm_matrix.get(), annot=True)
    sns_plot.figure.savefig(filename,dpi=600)


def main():
    # Set up and connect to Dask Client
    current_dir=os.environ.get('SLURM_SUBMIT_DIR')
    dask.config.set({"dataframe.backend": "cudf"})
    client = Client(scheduler_file=current_dir+'/out/scheduler.json')
    client.wait_for_workers(8)
    # print client informations
    print("\nclient:\n", client, flush=True)
    # Read the dataset
    # Now, we load the data using dask-cudf
    ddf = dask_cudf.read_csv(current_dir + "/data/HIGGS.csv")
    colnames = ['process',     'lepton_pT',   'lepton_eta',  'lepton_phi',  
                'missing_energy_magnitude',   'missing_energy_phi', 
                'jet_1_pt',    'jet_1_eta',   'jet_1_phi',   'jet_1_b_tag', 
                'jet_2_pt',    'jet_2_eta',   'jet_2_phi',   'jet_2_b_tag',
                'jet_3_pt',    'jet_3_eta',   'jet_3_phi',   'jet_3_b_tag', 
                'jet_4_pt',    'jet_4_eta',   'jet_4_phi',   'jet_4_b_tag',
                'm_jj','m_jjj','m_lv','m_jlv','m_bb','m_wbb','m_wwbb']

    ddf.columns = colnames
    print("\nddf.head():\n", ddf.head(), flush=True)
    print("\nddf.shape:\n",  ddf.shape,  flush=True)
    # Split data
    # Now use train_test_split() function from dask-ml to split up the dataset.
    # Most of the time, the GPU backend of Dask works seamlessly with utilities in dask-ml and we can accelerate the entire ML pipeline as such:
    X_train, X_valid, y_train, y_valid = load_higgs(client, ddf)
    print("\nX_train.head():\n", X_train.head(), flush=True)
    print("\nX_train.shape:\n",  X_train.shape,  flush=True)
    print("\ny_train.head():\n", y_train.head(), flush=True)
    print("\ny_train.shape:\n",  y_train.shape,  flush=True)
    # Model training
    booster = fit_model_es(client, X=X_train, y=y_train, X_valid=X_valid, y_valid=y_valid)
    print("\nbooster:\n", booster, flush=True)
    # Train with Customized objective and evaluation metric
    booster_custom = fit_model_customized_objective(client, X=X_train, y=y_train, X_valid=X_valid, y_valid=y_valid)
    print("\nbooster_custom:\n", booster_custom, flush=True)
    # Running inference
    y_preds = predict(client, booster, X_valid)
    y_preds_custom = predict(client, booster_custom, X_valid)
    y_preds_custom = 1.0 / (1.0 + np.exp(-y_preds_custom))
    y_valid = y_valid.persist()
    y_preds = y_preds.persist()
    y_preds_custom = y_preds_custom.persist()
    # The prediction is a probability between 0 and 1 that the event is a Higgs boson.
    # By rounding and converting to int, probabilities < 0.5 becomes 0 and probabilities > 0.5 becomes 1 
    # In this case we use a simple threshold of 0.5, but other possibilities are allowed 
    y_valid_label = y_valid.round().astype(int)
    y_preds_label = y_preds.round().astype(int)
    y_preds_custom_label = y_preds_custom.round().astype(int)
    wait([y_valid_label, y_preds_label, y_preds_custom_label])
    score = accuracy_score(y_valid_label, y_preds_label)
    score_custom = accuracy_score(y_valid_label, y_preds_custom_label)
    print(f"\nmodel accuracy: {score}\n", flush=True)
    print(f"\ncustom model accuracy: {score_custom}\n", flush=True)

    print("\ny_valid_label.head():\n", y_valid_label.head(), flush=True)
    print("\ny_valid_label.shape:\n",  y_valid_label.shape,  flush=True)
    print("\ny_preds_label.head():\n", y_preds_label.head(), flush=True)
    print("\ny_preds_label.shape:\n",  y_preds_label.shape,  flush=True)
    print("\ny_preds_custom_label.head():\n", y_preds_custom_label.head(), flush=True)
    print("\ny_preds_custom_label.shape:\n",  y_preds_custom_label.shape,  flush=True)
    
    y_valid_c = y_valid.compute()
    y_preds_c = y_preds.compute()
    y_preds_custom_c = y_preds_custom.compute()
    # By rounding and converting to int, <0.5 becomes 0, >0.5 becomes 1 
    y_valid_c_label = y_valid_c.round().astype(int)
    y_preds_c_label = y_preds_c.round().astype(int)
    y_preds_custom_c_label = y_preds_custom_c.round().astype(int)    
    print("\nAfter calling compute():\n", flush=True)
    print("\ny_valid_c.shape:\n",  y_valid_c.shape,  flush=True)
    print("\ny_preds_c.shape:\n",  y_preds_c.shape,  flush=True)
    print("\ny_preds_custom_c.shape:\n",  y_preds_custom_c.shape,  flush=True)
    print("\nmodel results:\n", flush=True)
    report(y_valid_c_label.to_cupy(), y_preds_c_label.to_cupy(), current_dir + '/cm_matrix.png')
    print("\ncustom model results:\n", flush=True)
    report(y_valid_c_label.to_cupy(), y_preds_custom_c_label.to_cupy(), current_dir + '/cm_matrix_custom.png')
    roc_auc = roc_auc_score(y_valid_c_label.to_cupy(), y_preds_c.to_cupy())
    roc_auc_custom = roc_auc_score(y_valid_c_label.to_cupy(), y_preds_custom_c.to_cupy())
    print(f"\nmodel roc_auc: {roc_auc}\n", flush=True)
    print(f"\ncustom model roc_auc: {roc_auc_custom}\n", flush=True)
    # Clean up
    # When finished, be sure to destroy your cluster to avoid incurring extra costs for idle resources.
    client.close()

if __name__ == "__main__":
    main()


