"""
This file contains data preprocessing and feature engineering
"""

import pandas as pd
import util as utils
from imblearn.over_sampling import SMOTEN
from sklearn.preprocessing import StandardScaler

def load_dataset(config_data: dict):
    """
    Load dataset
    """
    x_train = utils.pickle_load(config_data["train_set_path"][0])
    y_train = utils.pickle_load(config_data["train_set_path"][1])
    x_valid = utils.pickle_load(config_data["valid_set_path"][0])
    y_valid = utils.pickle_load(config_data["valid_set_path"][1])
    x_test = utils.pickle_load(config_data["test_set_path"][0])
    y_test = utils.pickle_load(config_data["test_set_path"][1])

    return x_train, x_valid, x_test, y_train, y_valid, y_test

def fill_missing_values(df):
    """
    fill missing values in the dataset
    """
    data = df.copy()
    for col in data.columns:
        if data[col].isna().sum() > 1:
            mean_value = data[col].mean()
            data[col].fillna(value=mean_value, inplace=True)

    return data

def column_encoder(df):
    """
    encode the categorical columns
    """
    data = pd.get_dummies(df)

    return data

def scaler_model(df, config):
    """
    save scaler model
    """
    scaler = StandardScaler()
    scaler.fit(df)
    utils.pickle_dump(scaler, config["standard_scaler"]) 

def scale_data(df, config):
    """
    scale the dataset
    """
    scaler = utils.pickle_load(config["standard_scaler"])
    set_x_scaled = scaler.transform(df)

    return set_x_scaled

def resample_data(set_x, set_y):
    """
    Oversampling the dataset
    """
    sm = SMOTEN(k_neighbors=20, n_jobs=-1)
    X_train_res, y_train_res = sm.fit_resample(set_x, set_y)

    return X_train_res, y_train_res

if __name__ == "__main__":
    # 1. Load configuration file
    config = utils.load_config()

    # 2. Load dataset
    x_train, x_valid, x_test, \
        y_train, y_valid, y_test = load_dataset(config)

    # 3. Imputation of missing values
    X_train_imp = fill_missing_values(x_train)
    X_valid_imp = fill_missing_values(x_valid)
    X_test_imp = fill_missing_values(x_test)

    # 4. Encode column with categorical type
    X_train_imp_encode = column_encoder(X_train_imp)
    X_valid_imp_encode = column_encoder(X_valid_imp)
    X_test_imp_encode = column_encoder(X_test_imp)

    # 5. Scale data
    scaler_model(X_test_imp_encode, config)
    X_train_imp_encode_scaled = scale_data(X_train_imp_encode, config)
    X_valid_imp_encode_scaled = scale_data(X_valid_imp_encode, config)
    X_test_imp_encode_scaled = scale_data(X_test_imp_encode, config)

    # 6. Oversampling data
    X_train_imp_encode_scaled_bal, y_train_bal = resample_data(
        X_train_imp_encode_scaled, 
        y_train
    )
    print("train:", X_train_imp_encode_scaled_bal.shape)
    print("valid:", X_valid_imp_encode_scaled.shape)
    print("test:", X_test_imp_encode_scaled.shape)

    # 7. Dump set data
    utils.pickle_dump(
            X_train_imp_encode_scaled_bal,
            config["train_feng_set_path"][0]
    )
    utils.pickle_dump(
            y_train_bal,
            config["train_feng_set_path"][1]
    )
    utils.pickle_dump(
            X_valid_imp_encode_scaled,
            config["valid_feng_set_path"][0]
    )
    utils.pickle_dump(
            y_valid,
            config["valid_feng_set_path"][1]
    )
    utils.pickle_dump(
            X_test_imp_encode_scaled,
            config["test_feng_set_path"][0]
    )
    utils.pickle_dump(
            y_test,
            config["test_feng_set_path"][1]
    )

    