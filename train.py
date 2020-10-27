#!/usr/bin/python2.7
# -*-coding:Utf-8-*-

from sys import exit, argv as av
import csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#
##
##  First : try to open the data's file and set Globals
##
#

if __name__ == "__main__":
    if len(av) < 2:
        print("Usage :\n./train.py [-v][-c int int float][file.txt] data.csv\n" +\
                "-v\t\t: visual\n-c\t\t: custom values for Cycles (number of iterations ofthe algorithm), " +\
                "Traces (number of traces to keep for the second visual graph), and Learning rate (alpha). " +\
                "Default values are 100 50 0.05\nfile.txt\t: output file where you want to save thetas\n" + \
                "data.csv\t: file where data are taken by the program")
        exit()
    output_set = False
    data_set = False
    Output = "thetas.txt"
    Visual = False
    Cycles = 100
    Traces = 50
    Learning_rate = 0.05
    Init_theta0, Init_theta1 = float(0), float(0)
    Custom = False
    i = 1
    while i < len(av):
        if ".csv" in av[i]:
            if data_set:
                exit("Can't load multiple .csv files")
            try:
                data = pd.read_csv(av[i])
                data_set = True
            except:
                exit(av[i] + " is not a valid .csv file")
        elif ".txt" in av[i]:
            if output_set:
                exit("Only one output .txt file allowed")
            Output = av[i]
            output_set = True
        elif av[i] == "-v":
            Visual = True
        elif av[i] == "-c":
            try:
                Custom = True
                Cycles = int(av[i + 1])
                Traces = int(av[i + 2])
                Learning_rate = float(av[i + 3])
                i += 3
            except:
                exit("Wrong use of '-c', please refer to the manual")
        else:
            exit(av[i] + " is not a valid argument")
        i += 1
    if not data_set:
        exit("No .csv file given")
    try:
        X, Y = data.iloc[0:len(data), 0], data.iloc[0:len(data), 1]
        Data_size = len(X)
    except:
        exit(".csv file not valid")
    if Data_size < 2:
        exit("Not enough data for the program to work")
    if Cycles > 10000:
        exit("Cycles have to be int between 0 and 10000")
    if Traces > Cycles or Traces < 1:
        exit("Traces have to be int between 0 and the number of Cycles")
    if Learning_rate > 5 or Learning_rate <= 0:
        exit("Learning rate has to be between 0 and 5")
    if Custom:
        print("Using custom values\nCycles\t\t: {}\nTraces\t\t: {}\nLearning rate\t: {}".\
                format(Cycles, Traces, Learning_rate))

#
##
##   Functions
##
#

#
#   Displaying all data
#

def display(x, y, cost, theta, theta_history):
    font_size = 12
    line_points = np.linspace(0, max(x), 100)
    figure, axs = plt.subplots(1, 3, figsize=(20, 6))
    axs[0].set_title("Price of cars depending on their metrage", fontsize=font_size, y=1.01,\
            color="green", fontweight="bold")
    axs[0].scatter(x, y, color="green", label="data")
    axs[0].set_xlabel("Km", fontsize=font_size, color="green", fontweight="bold")
    axs[0].set_ylabel("Price", fontsize=font_size, color="green", fontweight="bold")
    axs[0].plot(line_points, predicted_y(theta, line_points), '-r', label="prediction")
    axs[0].legend(fancybox=False, framealpha=1, shadow=True, borderpad=1)
    axs[1].scatter(x, y, color="green", label="data")
    axs[1].set_title("With the prediction history", fontsize=font_size, y=1.01,\
            color="green", fontweight="bold")
    axs[1].set_xlabel("Km", fontsize=font_size, color="green", fontweight="bold")
    axs[1].set_ylabel("Price", fontsize=font_size, color="green", fontweight="bold")
    axs[1].plot(line_points, predicted_y(theta_history[0], line_points), 'lightcoral', label='prediction history')
    for theta_tmp in theta_history[1:-1]:
        axs[1].plot(line_points, predicted_y(theta_tmp, line_points), 'lightcoral')
    axs[1].plot(line_points, predicted_y(theta, line_points), '-r', label="prediction")
    axs[1].legend(fancybox=False, framealpha=1, shadow=True, borderpad=1)
    axs[2].set_title("Cost function", fontsize=font_size, y=1.01,\
            color="green", fontweight="bold")
    axs[2].plot(np.arange(0, Cycles), cost,'-r', label="cost")
    axs[2].legend(fancybox=False, framealpha=1, shadow=True, borderpad=1)
    axs[2].set_xlabel("Iterations", fontsize=font_size, color="green", fontweight="bold")
    axs[2].set_ylabel("Cost", fontsize=font_size, color="green", fontweight="bold")
    plt.rcParams["font.family"] = "sans-serif"
    plt.show()

#
#   Standardizes inputs to make the gradient descent algorithm more efficient.
#   It calculates how many standard deviations a value is far from the mean of
#   the entire data set. Unstandardize gets th values back to the normal form.
#

def standardize(data):
    return ((data - np.mean(data)) / np.std(data))

def unstandardize(stand, ref):
    return (stand * np.std(ref) + np.mean(ref))


#
#   Calculate new thetas depending on the new predicted values
#   after standardization and unstandardization of x and y data.
#

def calc_thetas(theta, x, y):
    theta[1] = (y[0] - y[1]) / (x[0] - x[1])
    theta[0] = theta[1] * x[0] * -1 + y[0]
    return (theta)

#
#   mse stands for mean squared error.
#   It is the cost function.
#

def mse(size, pred_y, real_y):
    mse = (1 / size) * sum(val**2 for val in (real_y - pred_y))
    return (mse)

#
#   Uses our thetas to predict a y value from a x input.
#

def predicted_y(theta, x):
    return (theta[0] + theta[1] * x)

#
#   Calculates partial derivative for thetas in the cost function.
#

def calc_partial_derivative(theta, x, y, n):
    pd_theta = [0, 0]

    pd_theta[0] = -(2/n)*sum(y - predicted_y(theta, x))
    pd_theta[1] = -(2/n)*sum(x * (y - predicted_y(theta, x)))
    return (pd_theta)

#
#   Calculates new thetas from the old ones.
#

def reevaluate_thetas(theta, x, y, n, alpha):
    pd_theta = calc_partial_derivative(theta, x, y, n)
    theta[0] -= Learning_rate * pd_theta[0]
    theta[1] -= Learning_rate * pd_theta[1]
    return (theta)

#
#   Gradient descent algorithme try to catch the thetas values for which
#   the cost function is the lower.
#   Cost function is defined : mse = (1/n) * sum(yi - (theta0 + theta1 * xi))
#   Where xi are the input x values and yi the input y ones.
#

def gradient_descent(theta, x, y, n, alpha, cycles, traces):
    cost = []
    theta_history = [[0.0,0.0]]
    for i in range(cycles):
        theta = reevaluate_thetas(theta, x, y, n, alpha)
        cost.append(mse(n, predicted_y(theta, x), y))
        if i % (cycles / traces) == 0:
            theta_history.append([theta[0], theta[1]])
    return (theta, theta_history, cost)

#
##
##  Script
##
#

if __name__ == "__main__":
    print("Calculating thetas...")
    theta = [Init_theta0, Init_theta1]
    try:
        x, y = standardize(X), standardize(Y)
        (theta, theta_history, cost) = gradient_descent(theta, x, y, float(Data_size),\
                Learning_rate, Cycles, Traces)
        y = predicted_y(theta, x)
        y = unstandardize(y, Y)
        theta = calc_thetas(theta, X, y)
        for tmp_theta in theta_history:
            y = predicted_y(tmp_theta, x)
            y = unstandardize(y, Y)
            tmp_theta = calc_thetas(tmp_theta, X, y)
    except:
        exit("Error : program exited, data may be not valid")
    print("theta0 = {0:.4f}\ntheta1 = {1:.4f}".format(theta[0], theta[1]))
    try:
        np.savetxt(Output, theta)
        print("Complete values saved in \"" + Output + "\"")
    except:
        print("Could not save " + Output)
    if Visual == True:
        display(X, Y, cost, theta, theta_history)
