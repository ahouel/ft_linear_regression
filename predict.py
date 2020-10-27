#!/usr/bin/python2.7
# -*-coding:Utf-8-*-

from sys import exit, argv as av
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#
##
##  Functions
##
#

def predict(theta):
    mode = 'm'
    while True:
        try:
            if mode == 'm':
                val = raw_input("Enter a mileage / \'p\' to switch to to price / \'q\' to quit:\n")
            if mode == 'p':
                val = raw_input("Enter a price / \'m\' to switch to to mileage / \'q\' to quit:\n")
        except:
            exit("Quiting the program...")
        if val == 'p' or val == 'm':
            mode = val
            continue
        if val == 'q':
            print("Quiting the program...")
            return
        try:
            num = float(val)
            if num < 0:
                print("Positive values only")
                continue
            if num > 1000000000000:
                print("Number too high")
                continue
            if mode == 'm':
                result = theta[0] + theta[1] * num
                if result < 0:
                    result = 0
                print("Estimated price for {} Km is {:.2f}".format(num, result))
            if mode == 'p':
                if theta[1] == '0':
                    print("theta1 is egal 0, any mileage makes a price of {}".format(num))
                else:
                    result = (num - theta[0]) / theta[1]
                    if result < 0:
                        result = 0
                    print("Estimated mileage for the price of {} is {:.4f}".format(num, result))
        except ValueError as error:
            print("Invalid input : " + str(error))

#
##
##  Script
##
#

if __name__ == "__main__":
    theta = [0.0, 0.0]
    if len(av) != 2 or not av[1].endswith(".txt"):
        exit("Usage :\n./predict.py file.txt")
    try:
        theta = np.loadtxt(av[1], dtype=np.longdouble)
        print("theta0 = {0:.4f}\ntheta1 = {1:.4f}\n".format(theta[0], theta[1]))
    except:
        exit("Could not load " + av[1])
    predict(theta)
