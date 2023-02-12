"""
This file contains training model
"""

import util as utils
from xgboost import XGBClassifier
from sklearn.metrics import classification_report


def load_train_feng(params: dict):
    """
    Load train set
    """
    x_train = utils.pickle_load(params["train_feng_set_path"][0])
    y_train = utils.pickle_load(params["train_feng_set_path"][1])

    return x_train, y_train

def load_valid(params: dict):
    """
    Load valid set
    """
    x_valid = utils.pickle_load(params["valid_feng_set_path"][0])
    y_valid = utils.pickle_load(params["valid_feng_set_path"][1])

    return x_valid, y_valid

def load_test(params: dict):
    """
    Load tets set
    """
    x_test = utils.pickle_load(params["test_feng_set_path"][0])
    y_test = utils.pickle_load(params["test_feng_set_path"][1])

    return x_test, y_test

def train_model(x_train, y_train, x_valid, y_valid, x_test, y_test):
    """
    Train model using dataset
    """
    eval_set = [(x_train, y_train), (x_valid, y_valid)]
    model = XGBClassifier(colsample_bytree=0.9215312945790399, gamma=8.579361690309746, max_depth=9, min_child_weight=10.0, 
                          reg_alpha=100.0, reg_lambda=0.553734864557859)
    model.fit(x_train, y_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=True)

    y_pred = model.predict(x_test)
    print(classification_report(y_test, y_pred))

    return model

if __name__ == "__main__" :
    # 1. Load config file
    config = utils.load_config()

    # 2. Load set data
    x_train, y_train = load_train_feng(config)
    x_valid, y_valid = load_valid(config)
    x_test, y_test = load_test(config)

    # 3. Train model
    model = train_model(x_train, y_train, x_valid, y_valid, x_test, y_test)

    # 4. Dump model
    utils.pickle_dump(model, config["production_model_path"])