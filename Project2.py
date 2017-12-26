import pandas as pd
import numpy as np
import matplotlib as plt
from geopy.distance import vincenty
from math import *
from datetime import datetime

# read the data from CSV file
all_data = pd.DataFrame(data=pd.read_csv('D:\Class\CSCI6515MLBigData\A2\geolife_raw.csv'))

# Put data in different virable to controll the size of it
data_frame = all_data
print("Original Data Rows count =", len(data_frame))

# Limatation o
limitation = 10


# Delete extra class from the data frame
def delete_class(df, class_todelete):
    for c in class_todelete:
        df = df[df.transportation_mode != c]
    return df


# Delete the Class run and motorcycle from the data frame
data_frame = delete_class(data_frame, ["motorcycle", "run"])
print("Delete two [\"motorcycle\", \"run\"] class  Data Rows count =", len(data_frame))

# Split the collected_time to col new_date and new_time
#  Delete the col collected_time
data_date = pd.to_datetime(data_frame.loc[:, "collected_time"])
data_frame = data_frame.drop("collected_time", axis=1)
date_val = [d.date() for d in data_date]
time_val = [d.time() for d in data_date]
data_frame['new_date'] = date_val  # The day
data_frame['new_time'] = time_val  # the time

# sort the data
data_frame = data_frame.sort_values(["t_user_id", "new_date", "new_time"])
# Change orgnizing of the columns
data_frame = data_frame[["t_user_id", "new_date", "transportation_mode", "new_time", "latitude", "longitude"]]
# Group the data by the
user = data_frame.groupby([data_frame.t_user_id, data_frame.new_date])
df = pd.DataFrame(data=np.array(user))


# Horizontal Bearing return value in degrees
def calcBearing(p1, p2):
    lat1 = p1[0]
    lon1 = p1[1]
    lat2 = p2[0]
    lon2 = p2[1]
    dLon = lon2 - lon1
    y = sin(dLon) * cos(lat2)
    x = cos(lat1) * sin(lat2) - sin(lat1) * cos(lat2) * cos(dLon)
    bearing = degrees(atan2(y, x))
    return bearing


# Get time by second
def get_second(time):
    h, m, s = str(time).split(':')
    result = int(h) * 3600 + (int(m) * 60) + (int(s))
    if result == 0:
        result = 1
    return result


# cal_subtraj_features
# Calculate the Distance, Speed, Acceleration, Bearing for the sub_traj
def cal_subtraj_features(subtraj):
    last_acceleration = 0
    result = list()

    for i in range(1, len(subtraj)):
        # Start point
        p1_lat = subtraj[i - 1, 4]
        p1_log = subtraj[i - 1, 5]
        p1 = (p1_lat, p1_log)
        # End point
        p2_lat = subtraj[i, 4]
        p2_log = subtraj[i, 5]
        p2 = (p2_lat, p2_log)

        # Calculate the distance in KM
        distance = (vincenty(p1, p2).meters)

        # Calculate the diffrence between the times of to point
        t1 = str(subtraj[i - 1, 3])
        t2 = str(subtraj[i, 3])
        FMT = '%H:%M:%S'
        if datetime.strptime(t2, FMT) > datetime.strptime(t1, FMT):
            tdelta = datetime.strptime(t2, FMT) - datetime.strptime(t1, FMT)
        else:
            tdelta = datetime.strptime(t2, FMT) - datetime.strptime(t1, FMT)

        tdelta = datetime.strptime(str(tdelta), FMT).time()

        # Cal Speed by Km/h
        speed = distance / get_second(tdelta)

        # Cal Acceleration Km/h
        acceleration = abs(speed - last_acceleration) / pow(get_second(tdelta), 2)
        last_acceleration = acceleration

        # Calculate the Bearing Degrees
        bearing = calcBearing(p1, p2)
        # Shift the bearing
        TH = 180
        # if (bearing > TH): bearing -= 360;
        if (bearing < TH): bearing += 360;

        # convert to degree
        bearing *= pi / 180

        result.append(np.array([distance, speed, acceleration, bearing]))

    return pd.DataFrame(result)


# Feature data
# input sub_traj_data and calculate the feature
# distance, speed, acceleration, bearing

def get_feature_data(_class, sub_traj_data):
    row_list = list()
    describe_data = sub_traj_data.describe()
    describe_data.loc[len(describe_data)] = sub_traj_data.median(axis=0)

    row_list.append(_class)
    for i in range(len(describe_data.iloc[0])):
        col = describe_data.iloc[:, i]
        row_list.append(col[1])
        row_list.append(col[2])
        row_list.append(col[8])
        row_list.append(col[3])
        row_list.append(col[7])
    return (row_list)


# user_day_traj
# get user day traj and split to by transportation class
# return feature description for each sub_traj
def user_day_traj(user_day):
    _class = user_day.iloc[0, 2]  # first class
    sub_traj = list()  # list to save data for one class
    result = list()  # list to save final result
    for i in range(0, len(user_day)):
        # add data of same class to the sub_traj
        if (_class == user_day.iloc[i, 2]):
            sub_traj.append(user_day.iloc[i])
        else:
            # append the feature data for the class
            feature_class = cal_subtraj_features(np.array(sub_traj))
            if len(feature_class) >= limitation:
                feature_class = get_feature_data(_class, feature_class)
                result.append(feature_class)

            # change the class
            _class = user_day.iloc[i, 2]
            # empty the sub_traj
            sub_traj = list()
            sub_traj.append(user_day.iloc[i])
    feature_class = cal_subtraj_features(np.array(sub_traj))
    if len(feature_class) >= limitation:
        feature_class = get_feature_data(_class, feature_class)
        result.append(feature_class)
    return result


# Creat the DataFram for all the UsersTraj
# get Grouped data by (User_ID, Date)
# return Datafrom for the Data
def preprocess_data(all_groped_data):
    # convert grouped data to data frame
    df = pd.DataFrame(data=np.array(all_groped_data))
    result = list()
    # for each user and each day
    for i in range(0, len(df)):
        if i % 100 == 0:  # just to show the script still work
            print('-', sep=' ', end='', flush=True)
        # day data
        user_day = df.iloc[i, 1]
        if len(user_day) >= limitation:
            user_day_feature = user_day_traj(user_day)
            if len(user_day_feature) > 0:
                for udf in user_day_feature:
                    result.append(udf)
    print("Done")
    return pd.DataFrame(data=result, columns=[
        "trans_mode", "dis_mean", "dis_std", "dis_median", "dis_min", "dis_max"
        , "spe_mean", "spe_std", "spe_median", "spe_min", "spe_max"
        , "acc_mean", "acc_std", "acc_median", "acc_min", "acc_max"
        , "bea_mean", "bea_std", "bea_median", "bea_min", "bea_max"])


result = preprocess_data(df)

print("Shape of Data is =", result.shape)
result.to_csv(r'result.csv', index=False)
