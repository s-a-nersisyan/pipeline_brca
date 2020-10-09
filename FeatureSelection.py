import pandas as pd
import numpy as np

import xgboost
import shap


def boosting_shapley(df, n, **kwargs):
    '''
    Input expression dataframe and number n, return list
    of n selected features
    TODO: for supervised feature selection one should also pass
    subset of datasets for feature selection
    '''
    datasets = kwargs["datasets"]
    eta = kwargs.get("eta", 0.001)
    num_rounds = kwargs.get("num_rounds", 3000)
    early_stopping_rounds = kwargs.get("early_stopping_rounds", 40)
    subsample = kwargs.get("subsample", 0.8)

    df_subset = df.loc[df["Dataset"].isin(datasets)]
    X = df_subset.drop(columns=["Class", "Dataset", "Dataset type"]).to_numpy()
    y = df_subset["Class"].to_numpy()
    xgboost_input = xgboost.DMatrix(X, label=y)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": eta,
        "subsample": subsample,
        "base_score": np.mean(y)
    }
    model = xgboost.train(
        params,
        xgboost_input,
        num_rounds,
        evals = [(xgboost_input, "test")],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False
    )

    shap_values = shap.TreeExplainer(model).shap_values(X)
    feature_importances = np.mean(np.abs(shap_values), axis=0)
    features = df_subset.drop(columns=["Class", "Dataset", "Dataset type"]).columns

    return [feature for feature, importance in sorted(zip(features, feature_importances), key=lambda x: x[1], reverse=True)][0:n]


if __name__ == "__main__":
    df = pd.read_csv("test_data/data.csv", index_col=0)
    df["Dataset type"] = "Somevalue"
    print(df)

    result = boosting_shapley(df, 10, datasets=["GSE3494", "GSE6532", "GSE12093", "GSE17705", "GSE1456"], num_rounds=10)
    print(result)
