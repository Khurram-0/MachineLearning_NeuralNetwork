import numpy as np
import pandas as pd
# import matplotlib
import matplotlib.pyplot as plt
import sklearn
from sklearn import preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# Function for Q1: Initial Function
def given_function_lab3():
    x_1 = np.linspace(-1, 1, num=30000, endpoint=True)
    # implementation of given function
    y_1 = (0.2 * x_1 ** 4) + (2 * x_1 ** 3) + (0.1 * x_1 ** 2) + 10

    data_1 = pd.DataFrame(x_1, columns=["x_1"])  # create a dataframe (better for shuffling)
    data_1["y_1"] = y_1  # add y-values to dataframe

    return data_1


# Function for Q2: Shuffle
def shuffle_data(argument_1):
    # generate x and y data, given initial values
    data_1 = given_function_lab3()

    print("x-mean = ", np.mean(data_1["x_1"]))
    print("y-mean = ", np.mean(data_1["y_1"]))

    # is shuffle required?
    if argument_1 == "shuffle":
        print("argument is shuffle")
        # randomize data (frac=1) by rows (axis=0), "drop=true" prevents creating of column with old index entries
        data_1 = data_1.sample(frac=1, axis=0, random_state=1).reset_index(drop=True)
        # random_state=1 for reproducible results
    return data_1


# Function for Q3: split
def split_train_test_val(data_1):
    train_ratio = 0.3  # training set 30%
    val_ratio = 0.2  # validation set 20%
    test_ratio = 0.5  # test set 50%

    # split data, 30%=train data, 50%=test data, 20%=validation data
    train_1, test_1, val_1 = np.split(data_1, [int(train_ratio*len(data_1)), int((1-val_ratio)*len(data_1))])
    # data_1 is a dataframe so each set in our split data is also a dataframe

    print("train_1 is\n", train_1)
    print("test_1 is\n", test_1)
    print("val_1 is\n", val_1)

    return train_1, test_1, val_1


# Function for Q4: scale btn 0 and 1
def scale_data(data_1):
    data_1 = data_1.values  # make a numpy array for scaling
    # define scalar, in this case we scale  btn 0 & 1
    min_max_scalar = preprocessing.MinMaxScaler(feature_range=(0, 1))
    # fit the data to the new scale
    data_scaled_1 = min_max_scalar.fit_transform(data_1)
    # make array into dataframe and rename columns
    data_scaled_1 = pd.DataFrame(data_scaled_1, columns=["x_1", "y_1"])

    return data_scaled_1


# Function for Q5: MAE, MSE, RMSE, and r2 score
def mae_mse_rmse_r2score(test_1, y_pred_1):
    print("Length of y_test_1 is", len(test_1))
    print("Length of y_pred is", len(y_pred_1))
    mae_1 = sklearn.metrics.mean_absolute_error(test_1["y_1"], y_pred_1)  # return Mean Absolute Error
    print("mae =", mae_1)
    # return Mean Squared Error
    mse_1 = sklearn.metrics.mean_squared_error(test_1["y_1"], y_pred_1, squared=True)  # squared=True means return MSE
    print("mse =", mse_1)
    # return Root Mean Square Error, squared=False mean return RMSE
    rmse_1 = sklearn.metrics.mean_squared_error(test_1["y_1"], y_pred_1, squared=False)
    print("rmse =", rmse_1)
    # return r2 score
    r2score_1 = sklearn.metrics.r2_score(test_1["y_1"], y_pred_1)
    print("r2 score =", r2score_1)

    return mae_1, mse_1, rmse_1, r2score_1


# Structure 1 model: 3 hidden layers - 12, 8 and 4 units respectively
def structure1_model(train_1, test_1, val_1, activation_1):
    model_1 = Sequential()  # call sequential model
    # first hidden layer, 12 units (with input shape)
    model_1.add(Dense(12, input_dim=1, activation=activation_1))
    # second hidden layer, 8 units
    model_1.add(Dense(8, activation=activation_1))
    # third hidden layer 4 units
    model_1.add(Dense(4, activation=activation_1))
    # output layer, 1 unit
    model_1.add(Dense(1, activation=activation_1))
    # compile model; using loss=mse, optimizer=adam, metrics=mse (to confirm model is working)
    model_1.compile(loss="mse", optimizer="adam", metrics=["mse"])
    # fit data to model; using 20 epochs, and 12 batch size
    model_1.fit(train_1["x_1"], train_1["y_1"], epochs=20, batch_size=12, validation_data=(val_1["x_1"], val_1["y_1"]))
    model_1.summary()
    # use model to predict y-values
    y_pred_1 = model_1.predict(test_1["x_1"])

    return y_pred_1


# Structure 2 model: 1 hidden layer with 24 units (12 + 8 + 4 = 24 units)
def structure2_model(train_1, test_1, val_1, activation_1):
    model_1 = Sequential()  # call sequential model
    # first hidden layer, 12 units (with input shape)
    model_1.add(Dense(24, input_dim=1, activation=activation_1))
    # output layer, 1 unit
    model_1.add(Dense(1, activation=activation_1))
    # compile model; using loss=mse, optimizer=adam, metrics=mse (to confirm model is working)
    model_1.compile(loss="mse", optimizer="adam", metrics=["mse"])
    # fit data to model; using 20 epochs, and 12 batch size
    model_1.fit(train_1["x_1"], train_1["y_1"], epochs=20, batch_size=12, validation_data=(val_1["x_1"], val_1["y_1"]))
    # use model to predict y-values
    y_pred_1 = model_1.predict(test_1["x_1"])

    return y_pred_1


# --------------------------------------------Question #1: Initial Function TEST----------------------------------------
'''
data = given_function_lab3()  # call function (un-shuffled data)

# plotting commands
plt.figure()  # new figure
plt.scatter(data["x_1"], data["y_1"])  # build scatter plot
plt.title("INITIAL PLOT OF y=0.2x^4+2x^3+0.1x^2+10")
plt.xlabel("X VALUES (30'000 samples with -1<=x<=1)")
plt.ylabel("Y VALUES")
plt.grid()  # enable grid
'''
# ------------------------------------------------Question #2: Shuffle TEST---------------------------------------------
'''
# print un-shuffled x and y values
print("\nUn-shuffled data")
print(data)

argument = "shuffle"  # tell shuffle what to do (i.e. whether to shuffle or not)
data_shuffled = shuffle_data(argument)  # call shuffle function with argument

print("x-mean-shuffled = ", np.mean(data_shuffled["x_1"]))
print("y-mean-shuffled = ", np.mean(data_shuffled["y_1"]))

# print x & y values after calling shuffle function
print("\nshuffled data")
print(data_shuffled)

# plotting commands
plt.figure()  # new figure
plt.scatter(data_shuffled["x_1"], data_shuffled["y_1"])  # build scatter plot
plt.title("Plot of Shuffled Data")
plt.xlabel("X VALUES")
plt.ylabel("Y VALUES")
plt.grid()  # enable grid
'''
# ----------------------------------------------Question #3: split data TEST--------------------------------------------
'''
argument = "shuffle"  # tell shuffle what to do (i.e. whether to shuffle or not)
data_shuffle = shuffle_data(argument)  # call shuffle function with argument

# INITIALIZE
train = 0
test = 0
val = 0

train, test, val = split_train_test_val(data_shuffle)

if len(train) + len(test) + len(val) == len(data_shuffle):
    print(f"train ({len(train)}) percentage = {100 * len(train) / len(data_shuffle)}%")
    print(f"test ({len(test)}) percentage = {100 * len(test) / len(data_shuffle)}%")
    print(f"val ({len(val)}) percentage = {100 * len(val) / len(data_shuffle)}%")
else:
    print(f"\nThe split data doesn't sum up to {len(data_shuffle)}?? how tho")
    exit()
    
print("Train is")
print(train)
print("Test is")
print(test)
print("Val is")
print(val)
'''
# -------------------------------------------Question #4: scale btn 0 & 1 TEST------------------------------------------
'''
argument = "shuffle"  # tell shuffle what to do (i.e. whether to shuffle or not)
data = shuffle_data(argument)  # call shuffle function with argument

data_scaled = scale_data(data)

print("Scaled Data is this\n", data_scaled)

# plotting commands
plt.figure()  # new figure
plt.scatter(data_scaled["x_1"], data_scaled["y_1"])  # build scatter plot
plt.title("Scaled data")
plt.xlabel("X VALUES")
plt.ylabel("Y VALUES")
plt.grid()  # enable grid
'''
# --------------------------------------Question #5: MAE, MSE, RMSE, & r2 score-----------------------------------------
# See function above (tested in code below)
# -------------------------------Case#1: Shuffled & Unscaled, Structure 1, Relu activation------------------------------

data_case = shuffle_data("shuffle")  # shuffled data, unscaled
# split data into train, test and validation
train, test, val = split_train_test_val(data_case)

activation_function = "relu"  # type of activation function to use
# use structure 1 to create a model
y_pred = structure1_model(train, test, val, activation_function)

print("Length of test_y_1 is", len(test["y_1"]))
print("Length of y_pred is", len(y_pred))

print("\nFor this case we have:")
mae_mse_rmse_r2score(test, y_pred)

# plotting commands for actual data
plt.figure()  # new figure
plt.scatter(test["x_1"], test["y_1"])  # build scatter plot
plt.title("Actual data plot: x_test vs y_test")
plt.xlabel("x-test values")
plt.ylabel("y-test values")
plt.grid()  # enable grid

# plotting commands for predicted data
plt.figure()  # new figure
plt.scatter(test["x_1"], y_pred)  # build scatter plot
plt.title("Predicted data plot: x_test vs y_pred")
plt.xlabel("x-test values")
plt.ylabel("y-pred values")
plt.grid()  # enable grid

# -------------------------------Case#2: Shuffled & Unscaled, Structure 2, Relu activation------------------------------
'''
data_case = shuffle_data("shuffle")  # shuffled data, unscaled
# split data into train, test and validation
train, test, val = split_train_test_val(data_case)

activation_function = "relu"  # type of activation function to use
# use structure 1 to create a model
y_pred = structure2_model(train, test, val, activation_function)

print("Length of test_y_1 is", len(test["y_1"]))
print("Length of y_pred is", len(y_pred))

print("\nFor this case we have:")
mae_mse_rmse_r2score(test, y_pred)

# plotting commands for actual data
plt.figure()  # new figure
plt.scatter(test["x_1"], test["y_1"])  # build scatter plot
plt.title("Actual data plot: x_test vs y_test")
plt.xlabel("x-test values")
plt.ylabel("y-test values")
plt.grid()  # enable grid

# plotting commands for predicted data
plt.figure()  # new figure
plt.scatter(test["x_1"], y_pred)  # build scatter plot
plt.title("Predicted data plot: x_test vs y_pred")
plt.xlabel("x-test values")
plt.ylabel("y-pred values")
plt.grid()  # enable grid
'''
# -------------------------------Case#3: Shuffled & Unscaled, Structure 1, tanh activation------------------------------
'''
data_case = shuffle_data("shuffle")  # shuffled data, unscaled
# split data into train, test and validation
train, test, val = split_train_test_val(data_case)

activation_function = "tanh"  # type of activation function to use
# use structure 1 to create a model
y_pred = structure1_model(train, test, val, activation_function)

print("Length of test_y_1 is", len(test["y_1"]))
print("Length of y_pred is", len(y_pred))

print("\nFor this case we have:")
mae_mse_rmse_r2score(test, y_pred)

# plotting commands for actual data
plt.figure()  # new figure
plt.scatter(test["x_1"], test["y_1"])  # build scatter plot
plt.title("Actual data plot: x_test vs y_test")
plt.xlabel("x-test values")
plt.ylabel("y-test values")
plt.grid()  # enable grid

# plotting commands for predicted data
plt.figure()  # new figure
plt.scatter(test["x_1"], y_pred)  # build scatter plot
plt.title("Predicted data plot: x_test vs y_pred")
plt.xlabel("x-test values")
plt.ylabel("y-pred values")
plt.grid()  # enable grid
'''
# ---------------------------------Case#4: Shuffled & Scaled, Structure 1, Relu activation------------------------------
'''
data_case = shuffle_data("shuffle")  # shuffled data, unscaled

# split data into train, test and validation
train, test, val = split_train_test_val(data_case)

# scale the data (each is dataframe: i.e. includes both x and y)
train = scale_data(train)
test = scale_data(test)
val = scale_data(val)

activation_function = "relu"  # type of activation function to use
# use structure 1 to create a model
y_pred = structure1_model(train, test, val, activation_function)

print("Length of test_y_1 is", len(test["y_1"]))
print("Length of y_pred is", len(y_pred))

print("\nFor this case we have:")
mae_mse_rmse_r2score(test, y_pred)

# plotting commands for actual data
plt.figure()  # new figure
plt.scatter(test["x_1"], test["y_1"])  # build scatter plot
plt.title("Actual data plot: x_test vs y_test")
plt.xlabel("x-test values")
plt.ylabel("y-test values")
plt.grid()  # enable grid

# plotting commands for predicted data
plt.figure()  # new figure
plt.scatter(test["x_1"], y_pred)  # build scatter plot
plt.title("Predicted data plot: x_test vs y_pred")
plt.xlabel("x-test values")
plt.ylabel("y-pred values")
plt.grid()  # enable grid
'''
# ---------------------------------Case#5: Shuffled & Scaled, Structure 1, tanh activation------------------------------
'''
data_case = shuffle_data("shuffle")  # shuffled data, unscaled

# split data into train, test and validation
train, test, val = split_train_test_val(data_case)

# scale the data (each is dataframe: i.e. includes both x and y)
train = scale_data(train)
test = scale_data(test)
val = scale_data(val)

activation_function = "tanh"  # type of activation function to use
# use structure 1 to create a model
y_pred = structure1_model(train, test, val, activation_function)

print("Length of test_y_1 is", len(test["y_1"]))
print("Length of y_pred is", len(y_pred))

print("\nFor this case we have:")
mae_mse_rmse_r2score(test, y_pred)

# plotting commands for actual data
plt.figure()  # new figure
plt.scatter(test["x_1"], test["y_1"])  # build scatter plot
plt.title("Actual data plot: x_test vs y_test")
plt.xlabel("x-test values")
plt.ylabel("y-test values")
plt.grid()  # enable grid

# plotting commands for predicted data
plt.figure()  # new figure
plt.scatter(test["x_1"], y_pred)  # build scatter plot
plt.title("Predicted data plot: x_test vs y_pred")
plt.xlabel("x-test values")
plt.ylabel("y-pred values")
plt.grid()  # enable grid
'''
# -----------------------------Case#1.1: Un-shuffled & Unscaled, Structure 1, Relu activation---------------------------
'''
data_case = shuffle_data("un-shuffled")  # un-shuffled data, unscaled
# split data into train, test and validation
train, test, val = split_train_test_val(data_case)

activation_function = "relu"  # type of activation function to use
# use structure 1 to create a model
y_pred = structure1_model(train, test, val, activation_function)

print("Length of test_y_1 is", len(test["y_1"]))
print("Length of y_pred is", len(y_pred))

print("\nFor this case we have:")
mae_mse_rmse_r2score(test, y_pred)

# plotting commands for actual data
plt.figure()  # new figure
plt.scatter(test["x_1"], test["y_1"])  # build scatter plot
plt.title("Actual data plot: x_test vs y_test")
plt.xlabel("x-test values")
plt.ylabel("y-test values")
plt.grid()  # enable grid

# plotting commands for predicted data
plt.figure()  # new figure
plt.scatter(test["x_1"], y_pred)  # build scatter plot
plt.title("Predicted data plot: x_test vs y_pred")
plt.xlabel("x-test values")
plt.ylabel("y-pred values")
plt.grid()  # enable grid
'''
# -----------------------------Case#1.2: Un-Shuffled & Unscaled, Structure 2, Relu activation---------------------------
'''
data_case = shuffle_data("un-shuffled")  # un-shuffled data, unscaled
# split data into train, test and validation
train, test, val = split_train_test_val(data_case)

activation_function = "relu"  # type of activation function to use
# use structure 1 to create a model
y_pred = structure2_model(train, test, val, activation_function)

print("Length of test_y_1 is", len(test["y_1"]))
print("Length of y_pred is", len(y_pred))

print("\nFor this case we have:")
mae_mse_rmse_r2score(test, y_pred)

# plotting commands for actual data
plt.figure()  # new figure
plt.scatter(test["x_1"], test["y_1"])  # build scatter plot
plt.title("Actual data plot: x_test vs y_test")
plt.xlabel("x-test values")
plt.ylabel("y-test values")
plt.grid()  # enable grid

# plotting commands for predicted data
plt.figure()  # new figure
plt.scatter(test["x_1"], y_pred)  # build scatter plot
plt.title("Predicted data plot: x_test vs y_pred")
plt.xlabel("x-test values")
plt.ylabel("y-pred values")
plt.grid()  # enable grid
'''
# -----------------------------Case#1.3: Un-Shuffled & Unscaled, Structure 1, tanh activation---------------------------
'''
data_case = shuffle_data("un-shuffled")  # un-shuffled data, unscaled
# split data into train, test and validation
train, test, val = split_train_test_val(data_case)

activation_function = "tanh"  # type of activation function to use
# use structure 1 to create a model
y_pred = structure1_model(train, test, val, activation_function)

print("Length of test_y_1 is", len(test["y_1"]))
print("Length of y_pred is", len(y_pred))

print("\nFor this case we have:")
mae_mse_rmse_r2score(test, y_pred)

# plotting commands for actual data
plt.figure()  # new figure
plt.scatter(test["x_1"], test["y_1"])  # build scatter plot
plt.title("Actual data plot: x_test vs y_test")
plt.xlabel("x-test values")
plt.ylabel("y-test values")
plt.grid()  # enable grid

# plotting commands for predicted data
plt.figure()  # new figure
plt.scatter(test["x_1"], y_pred)  # build scatter plot
plt.title("Predicted data plot: x_test vs y_pred")
plt.xlabel("x-test values")
plt.ylabel("y-pred values")
plt.grid()  # enable grid
'''
# -----------------------------Case#1.4: Un-Shuffled & Scaled, Structure 1, Relu activation-----------------------------
'''
data_case = shuffle_data("un-shuffled")  # un-shuffled data, unscaled

# split data into train, test and validation
train, test, val = split_train_test_val(data_case)

# scale the data (each is dataframe: i.e. includes both x and y)
train = scale_data(train)
test = scale_data(test)
val = scale_data(val)

activation_function = "relu"  # type of activation function to use
# use structure 1 to create a model
y_pred = structure1_model(train, test, val, activation_function)

print("Length of test_y_1 is", len(test["y_1"]))
print("Length of y_pred is", len(y_pred))

print("\nFor this case we have:")
mae_mse_rmse_r2score(test, y_pred)

# plotting commands for actual data
plt.figure()  # new figure
plt.scatter(test["x_1"], test["y_1"])  # build scatter plot
plt.title("Actual data plot: x_test vs y_test")
plt.xlabel("x-test values")
plt.ylabel("y-test values")
plt.grid()  # enable grid

# plotting commands for predicted data
plt.figure()  # new figure
plt.scatter(test["x_1"], y_pred)  # build scatter plot
plt.title("Predicted data plot: x_test vs y_pred")
plt.xlabel("x-test values")
plt.ylabel("y-pred values")
plt.grid()  # enable grid
'''
# -----------------------------Case#1.5: Un-Shuffled & Scaled, Structure 1, tanh activation-----------------------------
'''
data_case = shuffle_data("un-shuffled")  # un-shuffled data, unscaled

# split data into train, test and validation
train, test, val = split_train_test_val(data_case)

# scale the data (each is dataframe: i.e. includes both x and y)
train = scale_data(train)
test = scale_data(test)
val = scale_data(val)

activation_function = "tanh"  # type of activation function to use
# use structure 1 to create a model
y_pred = structure1_model(train, test, val, activation_function)

print("Length of test_y_1 is", len(test["y_1"]))
print("Length of y_pred is", len(y_pred))

print("\nFor this case we have:")
mae_mse_rmse_r2score(test, y_pred)

# plotting commands for actual data
plt.figure()  # new figure
plt.scatter(test["x_1"], test["y_1"])  # build scatter plot
plt.title("Actual data plot: x_test vs y_test")
plt.xlabel("x-test values")
plt.ylabel("y-test values")
plt.grid()  # enable grid

# plotting commands for predicted data
plt.figure()  # new figure
plt.scatter(test["x_1"], y_pred)  # build scatter plot
plt.title("Predicted data plot: x_test vs y_pred")
plt.xlabel("x-test values")
plt.ylabel("y-pred values")
plt.grid()  # enable grid
'''
# ------------------------------------------------------TEST SECTION----------------------------------------------------
'''
train_scaled = scale_data(train)
print("\nTrain Scaled is\n", train_scaled)
print("Train Mean is", np.mean(train))
print("Train_scaled Mean is", np.mean(train_scaled))

plt.figure()  # new figure
plt.scatter(train_scaled["x_1"], train_scaled["y_1"])  # build scatter plot
plt.title("Scaled data")
plt.xlabel("X VALUES")
plt.ylabel("Y VALUES")
plt.grid()  # enable grid

test_scaled = scale_data(test)
print("Test Scaled is\n", test_scaled)
print("Test Mean is", np.mean(test))
print("Test_scaled Mean is", np.mean(test_scaled))

plt.figure()  # new figure
plt.scatter(test_scaled["x_1"], test_scaled["y_1"])  # build scatter plot
plt.title("Scaled data")
plt.xlabel("X VALUES")
plt.ylabel("Y VALUES")
plt.grid()  # enable grid

val_scaled = scale_data(val)
print("Val Scaled is\n", val_scaled)
print("Val Mean is", np.mean(val))
print("Val_scaled Mean is", np.mean(val_scaled))
plt.figure()  # new figure
plt.scatter(val_scaled["x_1"], val_scaled["y_1"])  # build scatter plot
plt.title("Scaled data")
plt.xlabel("X VALUES")
plt.ylabel("Y VALUES")
plt.grid()  # enable grid
'''

plt.show()
