# This code implements the Adamax optimization algorithm based on the cost function provided by the user.

import sys
import numpy as np 
import scipy as sc 
from sympy import Symbol, diff, lambdify, sympify

def adamax(cost_function, function):
    x = Symbol('x')
    print("f(x) = ", cost_function)
    derivative = diff(cost_function, x)
    print("df(x)/dx = ", derivative)
    initialApproximation = float(input("\n---> Enter initial approximation: "))
    x0 = initialApproximation
    errorTolerance = float(input("---> Enter error tolerance: "))
    learningRate = 0.002
    print("\n---------------------------------------------------------------")
    print(" *** Starting AdaMax")
    print("        ---> x0 = ", initialApproximation)
    print("        ---> f(x0) = ", function(initialApproximation))

    #----------------------------------------------------------------------------------------------------------------------------------------------------------
    iterationCount = 0                                                        # Keeps track of the number of iterations
    xk = x0                                                                   # Current value of x in each iteration
    x_prev = 0.0                                                              # Previous value of x
    m0 = 0.0                                                                  # Initial value of the first moment vector
    mk = 0.0                                                                  # Current value of the first moment vector
    v0 = 0.0                                                                  # Initial value of the second moment vector
    vk = 0.0                                                                  # Current value of the second moment vector
    beta1 = 0.9                                                               # Decay rate for the first moment vector
    beta2 = 0.999                                                             # Decay rate for the second moment vector
    epsilon = 10**-8                                                          # Small constant for numerical stability
    while True:
      iterationCount += 1 
      x_prev = x0
      x0 = xk
      m0 = mk
      v0 = vk
      derivative_value = (lambdify(x, derivative, "numpy"))(xk)               # Compute the derivative value at xk
      gt = derivative_value                                                   # Set gt as the computed derivative value
      mk = beta1 * m0 + (1 - beta1) * gt                                      # Compute the updated first moment vector
      vk = max(beta2 * v0, gt)                                                # Compute the updated second moment vector
      mc_k = mk / (1 - beta1**iterationCount)                                 # Compute the bias-corrected first moment estimate
      xk = xk - learningRate * mc_k / vk                                      # Update the value of xk using the Adam optimization formula

      if abs(N(xk - x0)) < float(errorTolerance) or abs(N(xk - x_prev)) < 0.1 * float(errorTolerance):
          break
    #----------------------------------------------------------------------------------------------------------------------------------------------------------
    
    print(" *** Number of Iterations = ", iterationCount)
    print("        ---> Minima is at = ", xk)
    print("        ---> Minimum value of Cost Function = ", function(xk))
    print("---------------------------------------------------------------\n")



# Code execution section

def main():
    x = Symbol('x')
    cost_function = input("---> Enter cost function f(x): ").strip()
    c_f = sympify(cost_function)                                              # will lambdify c_f for fast parallel multipoint computation
    f = lambdify(x, c_f, "numpy")
    adamax(c_f, f)

if __name__ == "__main__":
    main()