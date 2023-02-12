"""
This file contains data pipeline
"""

import pandas as pd
import util as utils
import copy
from sklearn.model_selection import train_test_split

def read_raw_data(config: dict) -> pd.DataFrame:
    """
    Load dataset
    """
    return pd.read_csv(config["dataset_path"])

def check_data(input_data: pd.DataFrame, config: dict, api: bool = False):
        """
        Check types of data
        """
        input_data = copy.deepcopy(input_data)
        config = copy.deepcopy(config)

        if not api:
                assert input_data.select_dtypes("int").columns.to_list() == config["int_columns"], "an error occurs in int column(s)."
                assert input_data.select_dtypes("float").columns.to_list() == config["float_columns"], "an error occurs in float column(s)."
                assert input_data.select_dtypes("object").columns.to_list() == config["object_columns"], "an error occurs in object column(s)."
                
                # Check range of OUTCOME
                assert input_data[config["float_columns"][5]].between(
                        config["range_outcome"][0],
                        config["range_outcome"][1]
                        ).sum() == len(input_data), "an error occurs in range_outcome."
                # Check range of ID
                assert input_data[config["int_columns"][0]].between(
                        config["range_id"][0],
                        config["range_id"][1]
                        ).sum() == len(input_data), "an error occurs in range_id."
                # Check range of POSTAL_CODE
                assert input_data[config["int_columns"][1]].between(
                        config["range_postal_code"][0],
                        config["range_postal_code"][1]
                        ).sum() == len(input_data), "an error occurs in range_postal_code."
                # Check value of RACE
                assert input_data[config["object_columns"][2]].isin(
                        config["value_race"]
                        ).sum() == len(input_data), "an error occurs in value_race."
                # Check value of INCOME
                assert input_data[config["object_columns"][5]].isin(
                        config["value_income"]
                        ).sum() == len(input_data), "an error occurs in value_income."
                # Check value of VAHICLE_TYPE
                assert input_data[config["object_columns"][7]].isin(
                        config["value_vahicle_type"]
                        ).sum() == len(input_data), "an error occurs in value_vahicle_type."
                # Check range of DUIS
                assert input_data[config["int_columns"][3]].between(
                        config["range_duis"][0],
                        config["range_duis"][1]
                        ).sum() == len(input_data), "an error occurs in range_duis."
                # Check range of SPEEDING_VIOLATIONS
                assert input_data[config["int_columns"][2]].between(
                        config["range_speeding_violation"][0],
                        config["range_speeding_violation"][1]
                        ).sum() == len(input_data), "an error occurs in range_speeding_violation."
                # Check range of PAST_ACCIDENTS
                assert input_data[config["int_columns"][4]].between(
                        config["range_past_accidents"][0],
                        config["range_past_accidents"][1]
                        ).sum() == len(input_data), "an error occurs in range_past_accidents."
                # Check value of VEHICLE_YEAR
                assert input_data[config["object_columns"][6]].isin(
                        config["value_vehicle_year"]
                        ).sum() == len(input_data), "an error occurs in value_vehicle_year."
                # Check value of GENDER
                assert input_data[config["object_columns"][1]].isin(
                        config["value_gender"]
                        ).sum() == len(input_data), "an error occurs in value_gender."
                # Check value of EDUCATION
                assert input_data[config["object_columns"][4]].isin(
                        config["value_education"]
                        ).sum() == len(input_data), "an error occurs in value_education."
                # Check value of AGE
                assert input_data[config["object_columns"][0]].isin(
                        config["value_age"]
                        ).sum() == len(input_data), "an error occurs in value_age."
                # Check value of DRIVING_EXPERIENCE
                assert input_data[config["object_columns"][3]].isin(
                        config["value_driving_experience"]
                        ).sum() == len(input_data), "an error occurs in value_driving_experience."
                # Check range of CHILDREN
                assert input_data[config["float_columns"][3]].between(
                        config["range_children"][0],
                        config["range_children"][1]
                        ).sum() == len(input_data), "an error occurs in range_children."
                # Check range of MARRIED
                assert input_data[config["float_columns"][2]].between(
                        config["range_married"][0],
                        config["range_married"][1]
                        ).sum() == len(input_data), "an error occurs in range_married."
                # Check range of VEHICLE_OWNERSHIP
                assert input_data[config["float_columns"][1]].between(
                        config["range_vehicle_ownership"][0],
                        config["range_vehicle_ownership"][1]
                        ).sum() == len(input_data), "an error occurs in range_vehicle_ownership."
                
        else:
                # In case checking data from api
                # First 2 column names in list of int columns are not used as predictor (ID and Postal Code)
                int_columns = config["int_columns"]
                del int_columns[:2]

                # Last 1 column names in list of int columns are not used as predictor (Outcome)
                float_columns = config["float_columns"]
                del float_columns[-1:]

                # Last 3 column names in list of int columns are not used as predictor (Race, Income, and Vehicle Type)
                object_columns = config["object_columns"]
                not_object_columns = ["RACE", "INCOME", "VEHICLE_TYPE"]

                object_columns = [x for x in object_columns if x not in not_object_columns]
                assert input_data.select_dtypes("int").columns.to_list() == int_columns, "an error occurs in int column(s)."
                assert input_data.select_dtypes("float").columns.to_list() == float_columns, "an error occurs in float column(s)."
                assert input_data.select_dtypes("object").columns.to_list() == object_columns, "an error occurs in object column(s)."
                # Check range of ANNUAL_MILEAGE
                assert input_data[config["float_columns"][4]].between(
                        config["range_annual_mileage"][0],
                        config["range_annual_mileage"][1]
                        ).sum() == len(input_data), "an error occurs in range_annual_mileage."
                # Check range of CREDIT_SCORE
                assert input_data[config["float_columns"][0]].between(
                        config["range_credit_score"][0],
                        config["range_credit_score"][1]
                        ).sum() == len(input_data), "an error occurs in range_credit_score."
                 # Check range of CHILDREN
                assert input_data[config["float_columns"][3]].between(
                        config["range_children"][0],
                        config["range_children"][1]
                        ).sum() == len(input_data), "an error occurs in range_children."
                # Check range of MARRIED
                assert input_data[config["float_columns"][2]].between(
                        config["range_married"][0],
                        config["range_married"][1]
                        ).sum() == len(input_data), "an error occurs in range_married."
                # Check range of VEHICLE_OWNERSHIP
                assert input_data[config["float_columns"][1]].between(
                        config["range_vehicle_ownership"][0],
                        config["range_vehicle_ownership"][1]
                        ).sum() == len(input_data), "an error occurs in range_vehicle_ownership."
                 # Check range of DUIS
                assert input_data[config["int_columns"][1]].between(
                        config["range_duis"][0],
                        config["range_duis"][1]
                        ).sum() == len(input_data), "an error occurs in range_duis."
                # Check range of SPEEDING_VIOLATIONS
                assert input_data[config["int_columns"][0]].between(
                        config["range_speeding_violation"][0],
                        config["range_speeding_violation"][1]
                        ).sum() == len(input_data), "an error occurs in range_speeding_violation."
                # Check range of PAST_ACCIDENTS
                assert input_data[config["int_columns"][2]].between(
                        config["range_past_accidents"][0],
                        config["range_past_accidents"][1]
                        ).sum() == len(input_data), "an error occurs in range_past_accidents."
                # Check value of VEHICLE_YEAR
                assert input_data[config["object_columns"][6]].isin(
                        config["value_vehicle_year"]
                        ).sum() == len(input_data), "an error occurs in value_vehicle_year."
                # Check value of GENDER
                assert input_data[config["object_columns"][1]].isin(
                        config["value_gender"]
                        ).sum() == len(input_data), "an error occurs in value_gender."
                # Check value of EDUCATION
                assert input_data[config["object_columns"][4]].isin(
                        config["value_education"]
                        ).sum() == len(input_data), "an error occurs in value_education."
                # Check value of AGE
                assert input_data[config["object_columns"][0]].isin(
                        config["value_age"]
                        ).sum() == len(input_data), "an error occurs in value_age."
                # Check value of DRIVING_EXPERIENCE
                assert input_data[config["object_columns"][3]].isin(
                        config["value_driving_experience"]
                        ).sum() == len(input_data), "an error occurs in value_driving_experience."
        
        
        

def split_data(input_data: pd.DataFrame, config: dict):
    """
    Split from dataset into train, valid, and test set
    """
    # Split predictor and label
    x = input_data[config["predictors"]].copy()
    y = input_data[config["label"]].copy()
    # 1st split train and test
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size = config["test_size"],
        random_state = 42,
        stratify = y
    )
    # 2nd split test and valid
    x_valid, x_test, y_valid, y_test = train_test_split(
        x_test, y_test,
        test_size = config["valid_size"],
        random_state = 42,
        stratify = y_test
    )

    return x_train, x_valid, x_test, y_train, y_valid, y_test


if __name__ == "__main__":
    # 1. Load configuration file
    config = utils.load_config()
    # 2. Read all raw dataset
    raw_dataset = read_raw_data(config)
    # 3. Data defense for non API data
    check_data(raw_dataset, config)
    # 4. Splitting train, valid, and test set
    x_train, x_valid, x_test, \
        y_train, y_valid, y_test = split_data(raw_dataset, config)
    # 6. Save train, valid and test set
    utils.pickle_dump(x_train, config["train_set_path"][0])
    utils.pickle_dump(y_train, config["train_set_path"][1])
    utils.pickle_dump(x_valid, config["valid_set_path"][0])
    utils.pickle_dump(y_valid, config["valid_set_path"][1])
    utils.pickle_dump(x_test, config["test_set_path"][0])
    utils.pickle_dump(y_test, config["test_set_path"][1])
    utils.pickle_dump(raw_dataset, config["dataset_cleaned_path"])