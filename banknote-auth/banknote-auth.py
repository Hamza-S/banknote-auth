import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv
import math as math
import random
import sys
np.set_printoptions(threshold=sys.maxsize)

def getDesignMatrixBankNote(iFile):  # Construct the design matrix for bank note dataset
    numInputs = sum(1 for line in open(iFile))
    with open(iFile) as f:
        data = [(line.rstrip()) for line in f]

    inputs = [None] * numInputs
    columnNum = 0
    for i in range(0, numInputs):
        size = len(data[i].split(','))
        columnNum = size
        input = [None] * (size)
        for j in range(0, size):
            temp = data[i].split(',')
            # <----- Modify this if/else block to label each class -1 or +1
            if (j == size - 1 and (i < 762)):
                input[j] = -1
            else:
                input[j] = (float(temp[j]))
        inputs[i] = input
    columnNum = columnNum + 1
    X = np.zeros((numInputs, columnNum))

    for i in range(0, numInputs):
        for j in range(0, len(inputs[0])):
            X[i][0] = 1
            X[i][j+1] = inputs[i][j]
    return X


def empBinaryLoss(X, w, rows, cols):  # Binary loss method
    # Sum 1..|X|: [Check predictor = sign(t)]
    predictorOutput = [None] * rows
    for i in range(0, rows):
        y_x = 0

        for j in range(1, cols):
            # Compute predictor output for one input
            y_x = y_x + (X[i][j] * w[j])
        predictorOutput[i] = y_x + w[0, 0]
    loss_sum = 0
    for i in range(0, rows):
        t = X[i][cols]
        if (np.sign(predictorOutput[i][0]) != np.sign(t)):
            loss_sum += 1
    return (loss_sum/(X.shape[0]))


def gethingeLoss(X, w, rows, cols):  # Hinge loss method
    # Sum 1..|X|: [Check predictor = sign(t)]
    predictorOutput = [None] * rows
    for i in range(0, rows):
        y_x = 0

        for j in range(1, cols):

            y_x = y_x + (X[i][j] * w[j])
        # predictor output for one input
        predictorOutput[i] = y_x + w[0, 0]
    loss_sum = 0
    for i in range(0, rows):
        t = (X[i][cols])
        left = 0
        right = 1 - (t*predictorOutput[i][0])
        loss_sum += max(left, right)
    return (loss_sum/(X.shape[0]))


def SGD(X, T, l):  # SGD for SoftSVM method
    f2 = open("binloss.txt", "w")
    f3 = open("hingeloss.txt", "w")
    cols = X.shape[1] - 1
    theta_j = np.zeros((cols, 1))  # Initialize theta^0
    rows = X.shape[0]
    for j in range(1, T):  # loop 1 ... T
        w_j = (1 / (l*j))
        w_j = theta_j * w_j  # Set w_j
        rand_index = random.randint(0, rows-1)  # Get a random index
        # Find the t_i that corresponds to the random index
        t_i = X[rand_index][cols]
        x_i = np.zeros((cols, 1))
        for i in range(0, cols):
            x_i[i][0] = X[rand_index][i]  # Get x_i vector from that random row
        product = np.vdot(w_j, x_i)  # Compute <w_j,x_i>
        product = product * t_i  # Compute t_i<w_j,x_i>
        if (product < 1):  # Compare
            theta_j = np.add(theta_j, (t_i * x_i))  # Change theta
        else:
            theta_j = theta_j
        bin_loss = empBinaryLoss(X, w_j, rows, cols)  # Calculate binary loss
        hinge_loss = gethingeLoss(X, w_j, rows, cols)  # Calculate hinge loss
        f2.write(str(bin_loss) + "\n")
        f3.write(str(hinge_loss) + "\n")
    return w_j  # Return final iterate


def normalize(w): #normalize the w vector 
    norm = np.linalg.norm(w)
    if norm == 0:
        return w
    return w / norm


def lossMultiClass(X, w1, w2, w3): #multiclass predictor (experimental)
    #Given three different predictors, test all 3 and find a winner
    cols = X.shape[1] - 1
    rows = X.shape[0]
    x_i = np.zeros((cols, 1))
    w1 = normalize(w1)
    w2 = normalize(w2)
    w3 = normalize(w3)
    f2 = open("binloss.txt", "a")
    loss_sum = 0
    for i in range(0, rows):
        dot1 = 0
        dot2 = 0
        dot3 = 0
        t = X[i][cols]
        for j in range(0, cols):
            x_i[j][0] = X[i][j]
        dot1 = np.vdot(x_i, w1)
        dot2 = np.vdot(x_i, w2)
        dot3 = np.vdot(x_i, w3)
        ws = [dot1, dot2, dot3]
        winner = ws.index(max(ws)) + 1
        if (winner != t):
            loss_sum += 1
    return (loss_sum/(X.shape[0]))


# =============MAIN CONTROLLER/OUTPUT================
lamda = 0.01 #hyperparameter tweak
X = getDesignMatrixBankNote('data_banknote_authentication.txt')
w = SGD(X, 500, lamda) #returns the w vector used for prediction
loss1 = [0] * 6
loss2 = [0] * 499
xAxis = [0] * 499
with open('loss_multi.txt') as f:
    loss1 = [float((line.rstrip())) for line in f]
with open('hingeloss.txt') as f:
    loss2 = [float((line.rstrip())) for line in f]
for i in range(0, 499):
    xAxis[i] = i

# ==========Plotting preformance====================
plt.plot(xAxis, loss1, label='Binary Loss')
plt.plot(xAxis, loss2, label='Hinge loss')
plt.legend()
plt.ylim(0, 1)
plt.title("Hinge Loss, lambda = " + str(lamda))
plt.title("Binary Loss, lambda = " + str(lamda))
plt.ylabel("Loss")
plt.xlabel("Run")
plt.show()