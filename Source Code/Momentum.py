import numpy as np 
import scipy as sc 
from sympy import diff, lambdify, symbols, N


def momentum(cost_function, func):
    print("f(x) = ", cost_function)
    funcDerivative = diff(cost_function, x)
    print("df(x)/dx = ", funcDerivative)
    initialApproximation = float(input("Enter initial approximation: "))
    x0 = initialApproximation
    learningRate = float(input("Enter learning rate: "))
    momentumConstant = float(input("Enter momentum constant: "))
    errorTolerance = float(input("Enter error tolerance: "))
    print("\n---------------------------------------------------------------")
    print("Starting Momentum")
    print("    Initial Approximation (x0) = ", initialApproximation)
    print("                         f(x0) = ", func(initialApproximation))
    
    #---------------------------------------------------------------------------------------------------------------------------------
    iterationCount = 0
    xk = x0
    vk_prev = 0
    while True:
        iterationCount += 1
        func_derivative_value = (lambdify(x, funcDerivative, "numpy"))(xk)            # Calculate the derivative value at current xk
        vk = momentumConstant * vk_prev + learningRate * func_derivative_value        # Update the velocity using momentum formula
        xk -= vk                                                                      # Update xk using the velocity
        # Check if the difference between current xk and previous xk is within the error tolerance
        if abs(N(xk - x0)) < errorTolerance:
            break
        x0 = xk
        vk_prev = vk
    #---------------------------------------------------------------------------------------------------------------------------------
    
    print("Number of Iterations = ", iterationCount)
    print("    Minima is at = ", xk)
    print("    Minimum value of Cost Function = ", func(xk))
 


# Code execution section
    
def main():
    x = Symbol('x')
    cost_function_input = input("Enter cost function f(x): ").strip()
    cost_function = sympify(cost_function_input)
    # lambdify cost_function for fast parallel multipoint computation
    func = lambdify(x, cost_function, "numpy")
    momentum(cost_function, func)


if __name__ == "__main__":
    main()