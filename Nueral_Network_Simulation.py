import math
import copy
import sys
import numpy as np
import random
import decimal
learningRate = 0.75
function = str(sys.argv[1])
maxError = 10**(-8)
inputList = []
sumInputList = [(0,0), (1,0), (0,1), (1,1)]

for line in open("10000_pairs.txt"):
    temp = line.split()
    temp[0] = float(temp[0])
    temp[1] = float(temp[1])
    inputList.append(tuple(temp))

def toBaseX(num, X):
    num = int(num)
    result = ""
    if(num == 0):
        return "0"
    while(num > 0):
        r = num % X
        num = int(num/X)
        result = str(int(r))+result
    return result

def truth_table(bits, n):
    dictionary = {}
    digits = 2**bits
    results = toBaseX(n, 2)
    while(len(results) < digits):
        results = "0" + results
    for x in range(0, digits):
        y = digits-x-1
        z = toBaseX(y, 2)
        while(len(z) < bits):
            z = "0" + z
        dictionary[tuple(z)] = results[x]
    return dictionary

def pretty_pink_tt(table):
    numbers = int(math.log(int(len(table.keys())), 2))
    toPrint = ""
    for x in range(0, numbers):
        toPrint = toPrint + str(x) + "\t"
    toPrint = toPrint + "Out"
    print(toPrint)
    print("")
    for x in table.keys():
        toPrint = ""
        for y in x:
            toPrint = toPrint + y + "\t"
        toPrint = toPrint + table[x]
        print(toPrint)
    return ""

def step(num):
    if(num > 0):
        return 1
    else:
        return 0

new_step = np.vectorize(step)

def sigmoid(num):
    return 1/(1 + math.exp((-1)*num))
    
new_sigmoid = np.vectorize(sigmoid)

def dSigmoid(num):
    return math.exp((-1)*num)/((1+math.exp((-1)*num))**2)

new_dSigmoid = np.vectorize(dSigmoid)

def perceptron(A, w, b, x):
    build = 0
    if(type(w[0]) != int or type(w[0]) != float):
        if(type(w) == list):
            return A((x@np.array(w))+np.array(b))
        else:
            return A((x@w)+b)
    for z in range(0, len(w)):
        build = build + int(w[z])*float(x[z])
    build = build+b
    return A(build)

def check(n, w, b):
    bits = len(w)
    table = truth_table(bits, n)
    total = 0
    correct = 0
    for x in table.keys():
        total = total+1
        if(perceptron(step, w, b, x) == int(table[x])):
            correct = correct+1
    return correct/total

def runEpoch(table, bits, vector, error, learningRate):
    l = []
    for x in table.keys():
        l.append(copy.copy(x))
    for x in l:
        missValue = int(table[x]) - perceptron(step, vector, error, x)
        error = error + learningRate*missValue
        newVector = []
        for y in range(0, bits):
            newVector.append(learningRate*missValue*int(x[y]) + float(vector[y]))
        vector = tuple(newVector)
    return (vector, error)

def perceptronTraining(reps, table, bits, learningRate):
    baseVector = []
    for x in range(0, bits):
        baseVector.append(0)
    baseVector = tuple(baseVector)
    baseError = 0
    for w in range(0, 100):
        temp = runEpoch(table, bits, copy.copy(baseVector), copy.copy(baseError), learningRate)
        if(temp[0] == baseVector and temp[1] == baseError):
            return (temp[0], temp[1], check(reps, baseVector, baseError))
        baseVector = temp[0]
        baseError = temp[1]
    return (baseVector, baseError, check(reps, baseVector, baseError))

def totalPossible(bits):
    correct = 0
    total = 0
    for x in range(0, 2**(2**bits)):
        total = total+1
        if(perceptronTraining(x, truth_table(bits, x), bits)[2] == 1):
            correct = correct+1
    return (correct, total)

def XOR221(vector):
    p3 = perceptron(step, (-1, 1), 0.5, vector)
    p4 = perceptron(step, (1, -1), 0.5, vector)
    return perceptron(step, (1, 1), -1, (p3, p4))

def XORMatrix(vector): #XOR HAPPENS HERE
    w1 = [[1, 1], [-1, -1]]
    b1 = [-0.5, 1.5]
    w2 = [1, 1]
    b2 = -1
    layer1 = perceptron(new_step, w1, b1, np.array(vector))
    return perceptron(new_step, w2, b2, np.array(layer1))

def diamondStep(x, y):
    vector = np.array([x, y])
    w1 = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    b1 = [1, 1, 1, 1]
    w2 = [1, 1, 1, 1]
    b2 = -3.5
    layer1 = perceptron(new_step, w1, b1, vector)
    return perceptron(new_step, w2, b2, tuple(layer1))

def circleStep(x, y):
    p = 0.5
    j = -2.396
    vector = np.array([x, y])
    w1 = [[-1, -1], [1, -1], [1, 1], [-1, 1]]
    b1 = [p, p, p, p]
    w2 = [1, 1, 1, 1]
    b2 = j
    layer1 = perceptron(new_sigmoid, w1, b1, vector)
    x = (perceptron(new_sigmoid, w2, b2, tuple(layer1)))
    if(x <= 0.5):
        return 0
    else:
        return 1

def minimizeA():
    points = np.array([0, 0])
    error = 1
    while(error > maxError):
        gradient = 0
        gradient = np.array([(8*points[0]-3*points[1]+24), (4*points[1]-3*points[0]-20)])
        print("Gradient:", gradient)
        gradient = (-1)*learningRate*gradient
        error = np.linalg.norm(gradient)
        print("point:", points)
        points = points+gradient
    return points

def minimizeB():
    points = np.array([0, 0])
    error = 1
    while(error > maxError):
        gradient = np.array([(2*(points[0]-points[1]**2)), (2*(-2*points[0]*points[1]+2*(points[1]**3)+points[1]-1))])
        print("Gradient:", gradient)
        gradient = (-1)*learningRate*gradient
        error = np.linalg.norm(gradient)
        print("point:", points)
        points = points+gradient
    return points

def trainRound(w, b, layers, inputs, ideal):
    vector = np.array([inputs])
    layerValues = [vector]
    dots = []
    for z in range(0, layers):
        dots.append(layerValues[z]@w[z]+b[z])
        layerValues.append(perceptron(new_sigmoid, w[z], b[z], layerValues[z]))
    deltaN = new_dSigmoid(dots[layers-1])*(ideal-layerValues[layers])
    deltaL = [deltaN]
    for temp in range(1, layers):
        z = layers-temp
        deltaL.append(new_dSigmoid(dots[z-1])*(deltaL[len(deltaL)-1]@w[z].T))
    for z in range(0, layers):
        b[z] = b[z] + learningRate*deltaL[layers-z-1]
        w[z] = w[z] + learningRate*(layerValues[z].T@deltaL[layers-z-1])
    return (w, b)

def trainRoundSum(w, b, layers, inputs, ideal):
    vector = np.array([inputs])
    layerValues = [vector]
    dots = []
    for z in range(0, layers):
        dots.append(layerValues[z]@w[z]+b[z])
        layerValues.append(perceptron(new_sigmoid, w[z], b[z], layerValues[z]))
    print(inputs, "-", layerValues[layers])
    deltaN = new_dSigmoid(dots[layers-1])*(ideal-layerValues[layers])
    deltaL = [deltaN]
    for temp in range(1, layers):
        z = layers-temp
        deltaL.append(new_dSigmoid(dots[z-1])*(deltaL[len(deltaL)-1]@w[z].T))
    for z in range(0, layers):
        b[z] = b[z] + learningRate*deltaL[layers-z-1]
        w[z] = w[z] + learningRate*(layerValues[z].T@deltaL[layers-z-1])
    return (w, b)

def trainSum():
    w1 = np.array([[random.uniform(-1, 1), random.uniform(-1, 1)], [random.uniform(-1, 1), random.uniform(-1, 1)]])
    b1 = np.array([[random.uniform(-1, 1), random.uniform(-1, 1)]])
    w2 = np.array([[random.uniform(-1, 1), random.uniform(-1, 1)], [random.uniform(-1, 1), random.uniform(-1, 1)]])
    b2 = np.array([[random.uniform(-1, 1), random.uniform(-1, 1)]])
    w = []
    b = []
    w.append(w1)
    w.append(w2)
    b.append(b1)
    b.append(b2)
    rounds = 0
    while(rounds < 10000):
        for (x,y) in sumInputList:
            if(x == 1 and y == 1):
                ideal = np.array([1,0])
            elif((x == 1 and y == 0) or (x == 0 and y == 1)):
                ideal = np.array([0,1])
            else:
                ideal = np.array([0,0])
            temp = trainRoundSum(w, b, 2, [x,y], ideal)
            w = temp[0]
            b = temp[1]
        rounds = rounds+1
        print("")
        print("epoch:", rounds)
        
def trainCircle():
    j = random.uniform(0, 1)
    p = random.uniform(-5, 5)
    w1 = np.array([[random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)], [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]])
    b1 = np.array([[j, j, j, j]])
    w2 = np.array([[random.uniform(-1, 1)], [random.uniform(-1, 1)], [random.uniform(-1, 1)], [random.uniform(-1, 1)]])
    b2 = np.array([[p]])
    w = []
    b = []
    w.append(w1)
    w.append(w2)
    b.append(b1)
    b.append(b2)
    rounds = 0
    while(rounds == rounds):
        for (x,y) in inputList:
            temp = (x**2 + y**2)**0.5
            if(temp > 1):
                ideal = 0
            else:
                ideal = 1
            temp = trainRound(w, b, 2, [x,y], ideal)
            w = temp[0]
            b = temp[1]
        incorrect = 0
        correct = 0
        for (x,y) in inputList:
            vector = np.array([x, y])
            layer1 = perceptron(new_sigmoid, w[0], b[0], vector)
            layer2 = perceptron(new_sigmoid, w[1], b[1], layer1)
            temp = (x**2 + y**2)**0.5
            if(temp > 1):
                ideal = 0
            else:
                ideal = 1
            if(np.linalg.norm(layer2) > 0.5):
                layer2 = 1
            else:
                layer2 = 0
            if(layer2 != ideal):
                incorrect = incorrect+1
            else:
                correct = correct+1
        rounds = rounds+1
        print("epoch:", rounds, "- incorrect:", incorrect)

if(function == 'S'):
    trainSum()
elif(function == 'C'):
    learningRate = 0.035
    trainCircle()
else:
    print("invalid input")