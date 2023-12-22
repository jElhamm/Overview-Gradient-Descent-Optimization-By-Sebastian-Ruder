# This code implements the Nesterov_Accelarated_Gradient optimization algorithm based on the cost function provided by the user.


import sys
import numpy as np
import scipy as sc
from sympy import Symbol, diff, lambdify, sympify, N


def nesterovAcceleratedGradient(cost_function, function):
    x = Symbol('x')
    print("f(x) =", cost_function)
    function_derivative = diff(cost_function, x)
    print("df(x)/dx =", function_derivative)
    initialApproximation = float(input("\n---> Enter initial approximation: "))
    learningRate = float(input("---> Enter learning rate: "))
    momentumConstant = float(input("---> Enter momentum constant: "))
    errorTolerance = float(input("---> Enter error tolerance: "))
    print("\n---------------------------------------------------------------")
    print(" *** Starting Nesterov Accelerated Gradient")
    print("        --->  x0 =", initialApproximation)
    print("        --->  f(x0) =", function(initialApproximation))
 
    #----------------------------------------------------------------------------------------------------------------------------------------------------
    iterationCount = 0
    current_x = initialApproximation
    currentVelocity = 0
    while True:
        iterationCount += 1
        functionDerivativeValue = lambdify(x, function_derivative, "numpy")(current_x -momentumConstant * currentVelocity)
        updatedVelocity = momentumConstant * currentVelocity + learningRate * functionDerivativeValue
        current_x = current_x - updatedVelocity

        if abs(N(current_x - initialApproximation)) < errorTolerance:
            break

        initialApproximation = current_x
        currentVelocity = updatedVelocity
    #----------------------------------------------------------------------------------------------------------------------------------------------------

    print(" *** Number of Iterations =", iterationCount)
    print("        --->  Minima is at =", current_x)
    print("        --->  Minimum value of Cost Function =", function(current_x))
    print("---------------------------------------------------------------\n")
    