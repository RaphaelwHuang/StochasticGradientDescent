#python 3.

import random
import numpy as np
import xlsxwriter
import pylab as pl

# The number of test examples
N = 400



# check X, which the Euclidean projection on two Scenarios
def euclideanProjection(scenario, gaussianComponents):
    # The number of dimension
    dimension = 5

    if scenario == 1:
        for i in range(dimension):
            if gaussianComponents[i] > 1:
                gaussianComponents[i] = 1
            if gaussianComponents[i] < -1:
                gaussianComponents[i] = -1
    else:
        sum = 0
        for i in range(dimension):
            sum += np.sqrt(gaussianComponents[i])
        # u1^2 + u2^2 + u3^2 + u4^2 + u5^2 <= 1^2
        if sum > 1:
            for i in range(dimension):
                gaussianComponents[i] /= sum

    return gaussianComponents

# Data Distribution D from Gaussian
def dataDistribution(scenario, sigma):
    # y random equal 0 or -1
    y = random.randint(0, 1)

    # The number of dimension
    dimension = 5

    # 5 i.i.d. Gaussian Components
    gaussianComponents = []

    if y == 0:
        y -= 1
        for i in range(dimension):
            u = random.gauss(-1 / 5, sigma)
            gaussianComponents.append(u)

    else:
        for i in range(dimension):
            u = random.gauss(1 / 5, sigma)
            gaussianComponents.append(u)

    # Check the Euclidean projection
    data = euclideanProjection(scenario, gaussianComponents)

    # Return training data: [X, y]
    data.append(y)
    return data

# Generate the test example
def testExample(scenario, sigma):
    # Test examples
    examples = []

    # 400 of test examples
    for i in range(N):
        examples.append(dataDistribution(scenario, sigma))

    return examples


# Compute Gradient lost func = ▽ι(wt, zt)
def gradient(w, z):
    # the value of X * y
    for i in range(6):
        z[i] *= z[5]

    # X * y array
    Xy = np.array(z)

    # Logistic Regression
    regression = -1 / (1 + np.exp(np.dot(w, z))) * Xy

    return regression

# Stochastic Gradient Descent
def sgd(scenario, sigma, n):
    # learningRate is fixed, which is 0.15
    learningRate = 0.15

    # The parameter value W (from 0 to T)
    W = []

    # Initialization W1 = 0
    w = np.zeros(6)
    W.append(w)

    # Training
    for i in range(n):
        # Draw the example Zt
        z = dataDistribution(scenario, sigma)

        # Compute Gt and Update Wt+1
        w = w - learningRate * gradient(w, z)
        euclideanProjection(scenario, w)
        W.append(w)

    # Return w = 1/T * sum of each w (from 0 to T)
    avgW = np.average(W)
    return avgW


# The binary classification error (the risk under ’0-1’ loss)
# sign(<w, (x,1)>) != y
def classificationError(w, z):
    y = z[5]

    # change x to (x, 1)
    if y == -1:
        z[5] = 1

    tem = np.sign(np.dot(w, z))
    error = np.array(tem) != y
    return error * 1

# The logistic loss function
def logisticLoss(w, z):
    y = z[5]
    return np.log(1 + np.exp(-y * np.dot(w, z)))


# Calculate logistic loss and binary classification error
def evaluateTestExample(W, examples):

    lsitOfLogisticLoss = []
    lsitOfclassificationError = []
    for w in W:
        lossess = []
        errors = []
        for example in examples:
            loss = logisticLoss(w, example)
            lossess.append(loss)
            error = classificationError(w, example)
            errors.append(error)
        lsitOfLogisticLoss.append(np.average(lossess))
        lsitOfclassificationError.append(np.average(errors))

    return lsitOfLogisticLoss, lsitOfclassificationError

def main(scenario, sigma, workbook):
    row = 0
    col = 0

    # For output
    worksheet = workbook.add_worksheet()

    # the number of iterations of the SGD for n = 50; 100; 500; 1000.
    listOfn = [50, 100, 500, 1000]


    examples = testExample(scenario, sigma)
    liskRisk = []
    listError = []
    for n in listOfn:
        # Run the SGD 30 times
        W = []
        for i in range(30):
            W.append(sgd(scenario, sigma, 50))

        #Get the result
        lsitOfLogisticLoss, lsitOfclassificationError =  evaluateTestExample(W, examples)
        print("*******************")
        print("Scenario: ", scenario)
        worksheet.write(row, col, scenario)
        print("Sigma: ", sigma)
        worksheet.write(row, col + 1, sigma)
        print("n: ", n)
        worksheet.write(row, col + 2, n)

        print("Logistic Loss")
        # Mean of logistic loss
        meanll = np.average(lsitOfLogisticLoss)
        print("Mean: ", meanll)
        worksheet.write(row, col + 3, meanll)

        # Std Dev of logistic loss
        stdll = np.std(lsitOfLogisticLoss)
        print("Std Dev: ", stdll)
        worksheet.write(row, col + 4, stdll)

        # Min of logistic loss
        min = np.min(lsitOfLogisticLoss)
        print("Min: ", min)
        worksheet.write(row, col + 5, min)

        # Excess Risk of logistic loss
        risk = np.average(lsitOfLogisticLoss) - np.min(lsitOfLogisticLoss)
        liskRisk.append(risk)
        print("Excess Risk: ", risk)
        worksheet.write(row, col + 6, risk)

        print("Classification Error")
        # Mean of Classification Error
        meance = np.average(lsitOfclassificationError)
        print("Mean: ", meance)
        worksheet.write(row, col + 7, meance)
        listError.append(meance)

        # Std Dev of Classification Error
        stdce = np.std(lsitOfclassificationError)
        print("Std Dev: ", stdce)
        worksheet.write(row, col + 8, stdce)

        row += 1

    return liskRisk, listError


# Scenario = 1, sigma = 0.05
workbook = xlsxwriter.Workbook('result1.xlsx')
listRisk1, listError1 = main(1, 0.05, workbook)

# Scenario = 1, sigma = 0.3
workbook = xlsxwriter.Workbook('result2.xlsx')
listRisk12, listError2 = main(1, 0.3, workbook)

# Scenario = 2, sigma = 0.05
workbook = xlsxwriter.Workbook('result3.xlsx')
listRisk3, listError3 = main(2, 0.05, workbook)
# # Scenario = 2, sigma = 0.3
workbook = xlsxwriter.Workbook('result4.xlsx')
listRisk4, listError4 = main(2, 0.3, workbook)
workbook.close()


# Make an array of x values
x = [50, 100, 500, 1000]
# use pylab to plot figures
# pl.plot(x, listRisk1, 'r')
# pl.plot(x, listRisk2, 'b')
# pl.plot(x, listRisk3, 'r')
# pl.plot(x, listRisk4, 'b')


# pl.plot(x, listError1, 'r')
# pl.plot(x, listError2, 'b')
pl.plot(x, listError3, 'r')
pl.plot(x, listError4, 'b')
# show the plot on the screen
pl.show()