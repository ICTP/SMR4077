# Based on:
# https://github.com/rapidsai-community/notebooks-contrib/blob/main/community_tutorials_and_guides/taxi/NYCTaxi-E2E.ipynb
# https://docs.rapids.ai/deployment/stable/examples/rapids-ec2-mnmg/notebook/

# Let's import our dependencies.
import os
import numpy as np
import cuspatial
import dask
import dask_cudf
from dask_ml.model_selection import train_test_split
from dask.distributed import Client, wait
from cuml.dask.common import utils as dask_utils
from cuml.dask.ensemble import RandomForestRegressor
from cuml.metrics import mean_squared_error

# Machine Learning Workflow

# 1. read and clean the data
# 2. add features
# 3. split into training and validation sets
# 4. fit a Random Forest model
# 5. predict on the validation set
# 6. compute RMSE


# For data cleanup, we define the following function
def clean(ddf, must_haves):
    # replace the extraneous spaces in column names and lower the font type
    tmp = {col: col.strip().lower() for col in list(ddf.columns)}
    ddf = ddf.rename(columns=tmp)
    ddf = ddf.rename(columns={
                                "tpep_pickup_datetime":  "pickup_datetime",
                                "tpep_dropoff_datetime": "dropoff_datetime",
                                "ratecodeid": "rate_code",
                             }
                    )

    ddf["pickup_datetime"]  = ddf["pickup_datetime"].astype("datetime64[ms]")
    ddf["dropoff_datetime"] = ddf["dropoff_datetime"].astype("datetime64[ms]")
    for col in ddf.columns:
        if col not in must_haves:
            ddf = ddf.drop(columns=col)
            continue
        if ddf[col].dtype == "object":
            # Fixing error: could not convert arg to str
            ddf = ddf.drop(columns=col)
        else:
            # downcast from 64bit to 32bit types for better performances
            if "int" in str(ddf[col].dtype):
                ddf[col] = ddf[col].astype("int32")
            if "float" in str(ddf[col].dtype):
                ddf[col] = ddf[col].astype("float32")
            ddf[col] = ddf[col].fillna(-1)

    return ddf

# When adding interesting features, we'll use a Haversine Distance calculation to find total trip distance
def haversine_dist(df):
    pickup  = cuspatial.GeoSeries.from_points_xy(df[["pickup_longitude",   "pickup_latitude"]].interleave_columns())
    dropoff = cuspatial.GeoSeries.from_points_xy(df[["dropoff_longitude", "dropoff_latitude"]].interleave_columns())
    df["h_distance"] = cuspatial.haversine_distance(pickup, dropoff)
    df["h_distance"] = df["h_distance"].astype("float32")
    return df


def main():
    # Dask-CUDA configuration
    os.environ["DASK_RMM__POOL_SIZE"] = "50GB"
    os.environ["DASK_UCX__CUDA_COPY"] = "True"
    os.environ["DASK_UCX__NVLINK"] = "True"
    os.environ["DASK_UCX__INFINIBAND"] = "True"
    os.environ["DASK_UCX__NET_DEVICES"] = "ib0"
    # Connect to a cluster through a Dask client
    client = Client(scheduler_file='out/scheduler.json')
    n_workers = 8
    # n_workers is the number of GPUs your cluster will have
    client.wait_for_workers(n_workers)
    print("\nclient:\n", client, flush=True)
    print("Workers are ready!",  flush=True)
    # 1. Read and Clean Data
    # On Leonardo we need to pre-download the data, and we assume that all the files are in the following directory
    base_path = 'data/nyctaxi/'
    # The data needs to be cleaned up before it can be used in a meaningful way.
    # We verify the columns have appropriate datatypes to make it ready for computation using cuML.
    # We create a list of all columns & dtypes the df must have for reading
    col_dtype = {
        "VendorID": "int32",
        "tpep_pickup_datetime": "datetime64[ms]",
        "tpep_dropoff_datetime": "datetime64[ms]",
        "passenger_count": "int32",
        "trip_distance": "float32",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "RatecodeID": "int32",
        "store_and_fwd_flag": "int32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "payment_type": "int32",
        "fare_amount": "float32",
        "extra": "float32",
        "mta_tax": "float32",
        "tip_amount": "float32",
        "total_amount": "float32",
        "tolls_amount": "float32",
        "improvement_surcharge": "float32",
    }
    # We define the folowing dictionary of required columns and their datatypes
    must_haves = {
        "pickup_datetime": "datetime64[ms]",
        "dropoff_datetime": "datetime64[ms]",
        "passenger_count": "int32",
        "trip_distance": "float32",
        "pickup_longitude": "float32",
        "pickup_latitude": "float32",
        "rate_code": "int32",
        "dropoff_longitude": "float32",
        "dropoff_latitude": "float32",
        "fare_amount": "float32",
    }
    # We read the csv files into a dask_cudf for 2014
    df_2014 = dask_cudf.read_csv(base_path+'2014/yellow_*.csv', dtype=col_dtype,)
    # and we clean the data using the map_partitions
    df_2014 = df_2014.map_partitions(clean, must_haves, meta=must_haves)
    # Similarly, we follow the same procedure also for 2015 data
    df_2015 = dask_cudf.read_csv(base_path+'2015/yellow_*.csv')
    df_2015 = df_2015.map_partitions(clean, must_haves, meta=must_haves)
    # In 2016, only January - June CSVs have the columns we need.
    # If we try to read base_path+2016/yellow_*.csv, Dask will not appreciate having differing schemas in the same DataFrame.
    # Instead, we'll need to create a list of the valid months and read them independently.
    months = [str(x).rjust(2, '0') for x in range(1, 7)]
    valid_files = [base_path+'2016/yellow_tripdata_2016-'+month+'.csv' for month in months]
    # Read and clean 2016 data
    df_2016 = dask_cudf.read_csv(valid_files).map_partitions(clean, must_haves, meta=must_haves)
    # Finally, we concatenate multiple DataFrames into one bigger one
    taxi_df = dask_cudf.concat([df_2014, df_2015, df_2016], axis=0)
    taxi_df = taxi_df.persist()
    # Now, we need to filter out any non-sensical records and outliers.
    # Specifically, we will only select records where tripdistance < 500 miles.
    # Similarly, we need to check abnormal fare_amount values for some records.
    # We will only select records where fare_amount < 500$.
    # Since we are interested in NYC, we also have to take coordinates into consideration
    # EDA yield the filter logic below.
    # apply a list of filter conditions to throw out records with missing or outlier values.
    query_frags = [
        'fare_amount > 1 and fare_amount < 500',
        'passenger_count > 0 and passenger_count < 6',
        'pickup_longitude > -75 and pickup_longitude < -73',
        'dropoff_longitude > -75 and dropoff_longitude < -73',
        'pickup_latitude > 40 and pickup_latitude < 42',
        'dropoff_latitude > 40 and dropoff_latitude < 42',
        'trip_distance > 0 and trip_distance < 500',
        'not (trip_distance > 50 and fare_amount < 50)',
        'not (trip_distance < 10 and fare_amount > 300)',
        'not dropoff_datetime <= pickup_datetime'
    ]
    taxi_df = taxi_df.query(' and '.join(query_frags))
    # reset_index and drop index column
    taxi_df = taxi_df.reset_index(drop=True)
    # 2. Add features
    # We'll add new features to the dataframe:
    # We can split the datetime column to retrieve year, month, day, hour, day_of_week columns.
    # Find the difference between pickup time and drop off time.
    # Haversine Distance between the pick-up and drop-off coordinates.
    taxi_df["hour"]  = taxi_df["pickup_datetime"].dt.hour.astype("int32")
    taxi_df["year"]  = taxi_df["pickup_datetime"].dt.year.astype("int32")
    taxi_df["month"] = taxi_df["pickup_datetime"].dt.month.astype("int32")
    taxi_df["day"]   = taxi_df["pickup_datetime"].dt.day.astype("int32")
    taxi_df["day_of_week"] = taxi_df["pickup_datetime"].dt.weekday.astype("int32")
    taxi_df["is_weekend"]  = (taxi_df["day_of_week"] >= 5).astype("int32")
    # calculate the time difference between dropoff and pickup.
    taxi_df["diff"] = taxi_df["dropoff_datetime"].astype("int32") - taxi_df["pickup_datetime"].astype("int32")
    taxi_df["diff"] = (taxi_df["diff"] / 1000).astype("int32")
    taxi_df["pickup_latitude_r"]   = taxi_df["pickup_latitude"]   // 0.01 * 0.01
    taxi_df["pickup_longitude_r"]  = taxi_df["pickup_longitude"]  // 0.01 * 0.01
    taxi_df["dropoff_latitude_r"]  = taxi_df["dropoff_latitude"]  // 0.01 * 0.01
    taxi_df["dropoff_longitude_r"] = taxi_df["dropoff_longitude"] // 0.01 * 0.01
    taxi_df = taxi_df.drop("pickup_datetime",  axis=1)
    taxi_df = taxi_df.drop("dropoff_datetime", axis=1)
    taxi_df = taxi_df.map_partitions(haversine_dist)
    taxi_df = taxi_df.persist()
    # 3. Split Data
    # Now, we split into training and validation sets
    X = taxi_df.drop(["fare_amount"], axis=1).astype("float32") 
    y = taxi_df["fare_amount"].astype("float32")
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True)
    workers = client.has_what().keys()
    X_train, X_test, y_train, y_test = dask_utils.persist_across_workers(client, [X_train, X_test, y_train, y_test], workers=workers)
    # 4. Create and fit a Random Forest Model
    # Now, we create cuml.dask RandomForest Regressor
    cu_dask_rf = RandomForestRegressor(ignore_empty_partitions=True)
    # fit RF model over the train dataset
    cu_dask_rf = cu_dask_rf.fit(X_train, y_train)
    # 5. Predict on validation set
    y_pred = cu_dask_rf.predict(X_test) # predict on validation set
    # 6. Compute RMSE
    score = mean_squared_error(y_pred.compute().to_numpy(), y_test.compute().to_numpy())
    print("Workflow Complete - After Training - RMSE on test set: ", np.sqrt(score))
    # Clean up resources
    client.close()


if __name__ == "__main__":
    main()



