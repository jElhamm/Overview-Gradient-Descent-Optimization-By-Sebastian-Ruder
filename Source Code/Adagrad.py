# This code implements the Adagrad optimization algorithm based on the cost function provided by the user.


import sys
import numpy as np 
import scipy as sc 
from sympy import symbols, lambdify, diff, N

def adagrad(cost_function, f):
    x = symbols('x')
    print("cost_function(x) = ", cost_function)
    f_derivative = diff(cost_function, x)
    print("df(x)/dx = ", f_derivative)
    initialApproximation = float(input("\n---> Enter initial approximation: "))
    current_x = initialApproximation
    learningRate = float(input("---> Enter learning rate: "))
    errorTolerance = float(input("---> Enter error tolerance: "))
    print("\n---------------------------------------------------------------")
    print(" *** Starting Adagrad")
    print("        --->  initial x = ", initialApproximation)
    print("        ---> cost_function(initial x) = ", f(initialApproximation))
 
    #----------------------------------------------------------------------------------------------------------------------------------------------------------
    iterationCount = 0
    previous_x = 0.0
    sumSquared_gradients = 0
    epsilon = 10.0 ** -8
    while True:
        iterationCount += 1
        previous_x = current_x
        fk_derivative = (lambdify(x, f_derivative, "numpy"))(current_x)                           # Compute the derivative for the current_x
        sumSquared_gradients += fk_derivative ** 2                                                # Update the sum of squared gradients
        adaptive_learning_rate = learningRate / ((sumSquared_gradients + epsilon) ** 0.5)         # Compute the adaptive learning rate
        current_x -= adaptive_learning_rate * fk_derivative                                       # Update current_x using the adaptive learning rate
        if abs(N(current_x - previous_x)) < errorTolerance:
            break
    #----------------------------------------------------------------------------------------------------------------------------------------------------------

    print(" ***Number of Iterations = ", iterationCount)
    print("        --->  Minima is at = ", current_x)
    print("        --->  Minimum value of Cost Function = ", f(current_x))
    print("---------------------------------------------------------------\n")
  


# Code execution section

def main():
    x = Symbol('x')
    cost_function=input("---> Enter cost function f(x): ").strip()
    c_f=sympify(cost_function)
    f = lambdify(x, c_f, "numpy")
    adagrad(c_f, f)


if __name__ == "__main__":
    main()