from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import util as utils
import data_pipeline as data_pipeline
from preprocessing import scale_data

config = utils.load_config()
model_data = utils.pickle_load(config["production_model_path"])

class api_data(BaseModel):
    SPEEDING_VIOLATIONS : int
    DUIS : int 
    PAST_ACCIDENTS : int
    CREDIT_SCORE : float
    VEHICLE_OWNERSHIP : float
    MARRIED : float
    CHILDREN : float
    ANNUAL_MILEAGE : float
    AGE : object
    GENDER : object
    DRIVING_EXPERIENCE : object
    EDUCATION : object
    VEHICLE_YEAR : object


app = FastAPI()

@app.get("/")
def home():
    return "Cari apa?"

@app.post("/predict/")
def predict(data: api_data):    
    # Convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)  # type: ignore
    data.columns = config["predictors"]

    # Convert dtype
    convert_type = {
        "SPEEDING_VIOLATIONS" : int,
        "DUIS" : int,
        "PAST_ACCIDENTS" : int,
        "CREDIT_SCORE" : float,
        "VEHICLE_OWNERSHIP" : float,
        "MARRIED" : float,
        "CHILDREN" : float,
        "ANNUAL_MILEAGE" : float,
        "AGE" : object,
        "GENDER" : object,
        "DRIVING_EXPERIENCE" : object,
        "EDUCATION" : object,
        "VEHICLE_YEAR" : object
    }

    data = data.astype(convert_type)
    # Check range data
    try:
        data_pipeline.check_data(data, config, True)  # type: ignore
    except AssertionError as ae:
        return {"res": [], "error_msg": str(ae)}

    # Preprocess data
    data_encode = {
        "SPEEDING_VIOLATIONS": data["SPEEDING_VIOLATIONS"],
        "DUIS": data["DUIS"], 
        "PAST_ACCIDENTS": data["PAST_ACCIDENTS"], 
        "CREDIT_SCORE": data["CREDIT_SCORE"],
        "VEHICLE_OWNERSHIP": data["VEHICLE_OWNERSHIP"], 
        "MARRIED": data["MARRIED"], 
        "CHILDREN": data["CHILDREN"], 
        "ANNUAL_MILEAGE": data["ANNUAL_MILEAGE"], 
    }

    age = data["AGE"][0]
    data_encode["AGE_16-25"] = 0
    data_encode["AGE_26-39"] = 0
    data_encode["AGE_40-64"] = 0
    data_encode["AGE_65+"] = 0
    if age == "16-25":
        data_encode["AGE_16-25"] = 1
    elif age == "26-39":
        data_encode["AGE_26-39"] = 1
    if age == "40-64":
        data_encode["AGE_40-64"] = 1
    else:
        data_encode["AGE_65+"] = 1

    gender = data["GENDER"][0]
    if gender == "female":
        data_encode["GENDER_female"] = 1
        data_encode["GENDER_male"] = 0
    else:
        data_encode["GENDER_female"] = 0
        data_encode["GENDER_male"] = 1

    driving_experience = data["DRIVING_EXPERIENCE"][0]
    data_encode["DRIVING_EXPERIENCE_0-9y"] = 0 
    data_encode["DRIVING_EXPERIENCE_10-19y"] = 0
    data_encode["DRIVING_EXPERIENCE_20-29y"] = 0 
    data_encode["DRIVING_EXPERIENCE_30y+"] = 0
    if driving_experience == "0-9y":
        data_encode["DRIVING_EXPERIENCE_0-9y"] = 1
    elif driving_experience == "10-19y":
        data_encode["DRIVING_EXPERIENCE_10-19y"] = 1 
    elif driving_experience == "20-29y":
        data_encode["DRIVING_EXPERIENCE_20-29y"] = 1 
    else:
        data_encode["DRIVING_EXPERIENCE_30y+"] = 1

    education = data["EDUCATION"][0]
    data_encode["EDUCATION_high school"] = 0
    data_encode["EDUCATION_none"] = 0
    data_encode["EDUCATION_university"] = 0
    if education == "high school":
        data_encode["EDUCATION_high school"] = 1
    elif education == "university":
        data_encode["EDUCATION_university"] = 1
    else:
        data_encode["EDUCATION_none"] = 1

    vehicle_year = data["VEHICLE_YEAR"][0]
    if vehicle_year == "after 2015":
        data_encode["VEHICLE_YEAR_after 2015"] = 1
        data_encode["VEHICLE_YEAR_before 2015"] = 0
    else:
        data_encode["VEHICLE_YEAR_after 2015"] = 0
        data_encode["VEHICLE_YEAR_before 2015"] = 1
    
    data_encode = pd.DataFrame(data_encode)
    print(data_encode.columns)

    data_encode_scaled = scale_data(data_encode, config)

    # Predict data
    y_pred = model_data.predict(data_encode_scaled)

    if y_pred[0] == 0:
        y_pred = "Tidak melakukan klaim"
    else:
        y_pred = "Melakukan klaim"
    return {"res" : y_pred, "error_msg": ""}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080, reload=True)