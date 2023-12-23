# This code implements the rms_prop optimization algorithm based on the cost function provided by the user.


import sys
import numpy as np
from sympy import Symbol, diff, lambdify, sympify, N


def rms_prop(cost_function, function):
    x = Symbol('x')
    print("f(x) =", cost_function)
    function_derivative = diff(cost_function, x)
    print("df(x)/dx =", function_derivative)
    initialApproximation = float(input("\n---> Enter initial approximation: "))
    errorTolerance = float(input("---> Enter error tolerance: "))
    learningRate = float(input("---> Enter learning rate: "))
    print("\n---------------------------------------------------------------")
    print("Starting RMS Prop")
    print("        --->  x0 =", initialApproximation)
    print("        --->  f(x0) =", function(initialApproximation))
 
    #----------------------------------------------------------------------------------------------------------------------------------------------------
    iterationCount = 0
    current_x = initialApproximation
    previous_x = 0.0
    eg2t_0 = 0.0
    epsilon = 10.0**-8
    while True:
        iterationCount =+1
        previous_x = initialApproximation
        initialApproximation = current_x
        eg2t_k = eg2t_0

        function_derivative_value = (lambdify(x, function_derivative, "numpy"))(current_x)
        gt = function_derivative_value
        eg2t_k = 0.9 * eg2t_0 + (0.1) * (gt**2)
        current_x = current_x - learningRate * gt / ((eg2t_k + epsilon)**0.5)

        if abs(N(current_x - initialApproximation)) < errorTolerance or abs(N(current_x - previous_x)) < 0.1 * errorTolerance:
            break
    #----------------------------------------------------------------------------------------------------------------------------------------------------

    print("Number of Iterations =", iterationCount)
    print("        --->  Minima is at =", current_x)
    print("        --->  Minimum value of Cost Function =", function(current_x))
    print("---------------------------------------------------------------\n")



# Code execution section

def main():
    x = Symbol('x')
    costFunction = input("---> Enter cost function f(x): ").strip()
    costFunctionSympy = sympify(costFunction)
    costFunctionNumpy = lambdify(x, costFunctionSympy, "numpy")
    rms_prop(costFunctionSympy, costFunctionNumpy)

if __name__ == "__main__":
    main()