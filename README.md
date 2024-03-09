# A Review Article On Gradient Descent Optimization Algorithms

   This repository contains the complete implementation of the article titled *"A Review Article On Gradient Descent Optimization Algorithms" by Sebastian Roeder*. 
   It includes the implementation of various existing optimization algorithms for gradient descent.


## Table of Contents

   - [Introduction](#introduction)
   - [Algorithms](#algorithms)
   - [Usage](#usage)
   - [License](#license)

## Introduction

   This repository serves as a comprehensive resource for understanding and implementing gradient descent optimization algorithms discussed 
   in the article "A Review Article On Gradient Descent Optimization Algorithms" by Sebastian Roeder. 
   The implementation covers a range of algorithms that can be utilized in the field of machine learning and optimization.

## Algorithms

   * Each algorithm is implemented as a separate module in this repository, accompanied by comprehensive documentation and code examples. 
   The following optimization algorithms have been implemented:

   1. [*Adam*](Source%20Code/Adam.py): Combines the benefits of momentum and RMSprop, using adaptive learning rates and momentum to converge faster.
      - Usage: Widely used and effective for a wide range of optimization problems.


   2. [*Nadam*](Source%20Code/Nadam.py): Combines Nesterov accelerated gradient and Adam, benefiting from both lookahead updates and adaptive learning rates.
       - Usage: A more advanced variant of Adam that offers improved convergence properties.


   3. [*Adamax*](Source%20Code/Adamax.py): A variant of Adam that incorporates the maximum norm of the past gradients for adaptive learning rates.
      - Usage: Effective for models with different ranges of parameter magnitudes.


   4. [*Amsgrad*](Source%20Code/Amsgrad.py): A modification to Adam that addresses the problem of the adaptive learning rate not achieving convexity for some objective functions.
      - Usage: Helps avoid overshooting in non-convex optimization problems.


   5. [*AdaGrad*](Source%20Code/Adagrad.py): Adapts the learning rate of each parameter based on the historical gradients, giving more weight to infrequent features.
      - Usage: Suitable for sparse datasets, where some features occur infrequently.


   6. [*RMSprop*](Source%20Code/RmsProp.py): A variation of AdaGrad that addresses its aggressive and monotonically decreasing learning rate.
      - Usage: Effective for non-stationary (changing) optimization problems.


   7. [*Momentum*](Source%20Code/Momentum.py): Adds momentum to the gradient descent update by accumulating a moving average of past gradients.
      - Usage: Accelerates convergence, especially in the presence of sparse gradients or noisy data.

   
   10. [*AdaDelta*](Source%20Code/Adadelta.py): An extension of AdaGrad that further improves the learning rate adaptation by eliminating the need for an initial learning rate.
      - Usage: Overcomes the learning rate decay problem of AdaGrad.


   8. [*Batch Gradient Descent*](Source%20Code/BatchGradientDescent.py): A basic optimization algorithm that updates the model parameters using the gradients of the entire training dataset.
      - Usage: Suitable for small to medium-sized datasets.


   9. [*Nesterov Accelerated Gradient*](Source%20Code/NesterovAccelarated.py): A modification of momentum that improves convergence by using a lookahead update.
      - Usage: Helps achieve faster convergence by reducing oscillations.


## Usage

   To use the implemented algorithms, follow these steps:

   1. Clone this repository to your local machine.
   2. Navigate to the respective algorithm module of interest.
   3. Read the provided documentation to understand the algorithm's theory, parameters, and usage.
   4. Refer to the code examples to see how the algorithm is applied in practical scenarios.
   5. Integrate the algorithms into your own machine learning or optimization projects by importing the necessary modules.


## References

   [An overview of gradient descent optimization algorithms](https://www.ruder.io/optimizing-gradient-descent/)


## License

   This repository is licensed under the MIT License.
   See the [LICENSE](./LICENSE) file for more details.