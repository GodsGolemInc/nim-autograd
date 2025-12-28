## nim-autograd: Automatic differentiation for ML
##
## Provides reverse-mode automatic differentiation for training
## neural networks with gradient descent optimization.
##
## v0.0.1: Tape and gradient tracking
## v0.0.2: Backward pass implementation
## v0.0.3: Common gradient functions

import nimml_autograd/tape
import nimml_autograd/backward
import nimml_autograd/gradients

export tape
export backward
export gradients
