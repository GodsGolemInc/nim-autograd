## nim-autograd: Automatic differentiation for ML
##
## Provides reverse-mode automatic differentiation for training
## neural networks with gradient descent optimization.
##
## v0.0.1: Tape and gradient tracking
## v0.0.2: Backward pass implementation

import nimml_autograd/tape
import nimml_autograd/backward

export tape
export backward
