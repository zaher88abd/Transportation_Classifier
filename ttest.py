import pandas as pd
import scipy.stats as sp


#
# "trans_mode", "dis_mean", "dis_std", "dis_median", "dis_min", "dis_max"
#         , "spe_mean", "spe_std", "spe_median", "spe_min", "spe_max"
#         , "acc_mean", "acc_std", "acc_median", "acc_min", "acc_max"
#         , "bea_mean", "bea_std", "bea_median", "bea_min", "bea_max"])


def describe_class(_all_data):
    bus = pd.DataFrame(_all_data[_all_data["trans_mode"] == 'bus'])
    car = pd.DataFrame(_all_data[_all_data["trans_mode"] == 'car'])
    walk = pd.DataFrame(_all_data[_all_data["trans_mode"] == 'walk'])
    taxi = pd.DataFrame(_all_data[_all_data["trans_mode"] == 'taxi'])
    subway = pd.DataFrame(_all_data[_all_data["trans_mode"] == 'subway'])
    train = pd.DataFrame(_all_data[_all_data["trans_mode"] == 'train'])

    bus_mean_spe_mean = bus["spe_mean"]
    bus_mean_acc_mean = bus["acc_mean"]
    bus_mean_dis_mean = bus["dis_mean"]

    car_mean_spe_mean = car["spe_mean"]
    car_mean_acc_mean = car["acc_mean"]
    car_mean_dis_mean = car["dis_mean"]

    walk_mean_spe_mean = walk["spe_mean"]
    walk_mean_acc_mean = walk["acc_mean"]
    walk_mean_dis_mean = walk["dis_mean"]

    taxi_mean_spe_mean = taxi["spe_mean"]
    taxi_mean_acc_mean = taxi["acc_mean"]
    taxi_mean_dis_mean = taxi["dis_mean"]

    subway_mean_spe_mean = subway["spe_mean"]
    subway_mean_acc_mean = subway["acc_mean"]
    subway_mean_dis_mean = subway["dis_mean"]

    train_mean_spe_mean = train["spe_mean"]
    train_mean_acc_mean = train["acc_mean"]
    train_mean_dis_mean = train["dis_mean"]

    result = list()
    result.append(["bus", bus_mean_spe_mean, bus_mean_acc_mean, bus_mean_dis_mean])
    result.append(["car", car_mean_spe_mean, car_mean_acc_mean, car_mean_dis_mean])
    result.append(["walk", walk_mean_spe_mean, walk_mean_acc_mean, walk_mean_dis_mean])
    result.append(["taxi", taxi_mean_spe_mean, taxi_mean_acc_mean, taxi_mean_dis_mean])
    result.append(["subway", subway_mean_spe_mean, subway_mean_acc_mean, subway_mean_dis_mean])
    result.append(["train", train_mean_spe_mean, train_mean_acc_mean, train_mean_dis_mean])
    return pd.DataFrame(data=result, columns=["class", "spe_mean", "acc_mean", "dis_mean"])



data_frame = pd.DataFrame(data=pd.read_csv(r'result.csv'))

pares = [["bus", "car"], ["bus", "walk"], ["bus", "taxi"], ["bus", "subway"], ["bus", "train"]
    , ["car", "walk"], ["car", "taxi"], ["car", "subway"], ["car", "train"]
    , ["walk", "taxi"], ["walk", "subway"], ["walk", "train"]
    , ["subway", "train"]]

data_measures = ["spe_mean", "acc_mean", "dis_mean"]

data_class_means = describe_class(data_frame)
data_class_means.set_index("class", inplace=True)
print(len(data_frame))
for j in data_measures:
    for i in range(len(pares)):
        p = pares[i]
        x = p[0]
        y = p[1]
        t_test_58_fd = 0.005
        data_class_x = data_class_means.loc[x][j]
        data_class_y = data_class_means.loc[y][j]
        statistic, p_val = sp.ttest_ind(data_class_x, data_class_y, equal_var=True)

        print("Measure ", j, ":")
        print('p_val', p_val, 'statistic', statistic)
        if p_val >= t_test_58_fd:
            print('the mean of Speed = ', x,
                  'is identical to mean of Speed', y)
        else:
            print('the mean of  = ', x,
                  'is not identical to mean of model', y)
    print("______________________________________________")
