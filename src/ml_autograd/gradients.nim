## Common Gradient Functions
##
## Provides gradient implementations for common ML operations.
## These are registered with the global gradient registry.

import std/[tables]
import ml_core
import ./tape
import ./backward

# Note: These are placeholder implementations.
# Real implementations would need actual tensor arithmetic.

# Unary operation gradients

proc gradNeg*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of negation: d/dx(-x) = -1
  ## Result: -grad
  @[grad]  # Placeholder: would negate grad

proc gradAbs*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of abs: d/dx(|x|) = sign(x)
  @[grad]  # Placeholder: would multiply by sign(input)

proc gradExp*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of exp: d/dx(e^x) = e^x
  @[output]  # Placeholder: would multiply grad by output

proc gradLog*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of log: d/dx(ln(x)) = 1/x
  @[grad]  # Placeholder: would divide by input

proc gradSqrt*(grad: TensorRef, inputs: seq[TensorRef],
               output: TensorRef): seq[TensorRef] =
  ## Gradient of sqrt: d/dx(sqrt(x)) = 1/(2*sqrt(x))
  @[grad]  # Placeholder

proc gradSquare*(grad: TensorRef, inputs: seq[TensorRef],
                 output: TensorRef): seq[TensorRef] =
  ## Gradient of square: d/dx(x^2) = 2x
  @[grad]  # Placeholder: would multiply by 2*input

proc gradSin*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of sin: d/dx(sin(x)) = cos(x)
  @[grad]  # Placeholder

proc gradCos*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of cos: d/dx(cos(x)) = -sin(x)
  @[grad]  # Placeholder

proc gradTanh*(grad: TensorRef, inputs: seq[TensorRef],
               output: TensorRef): seq[TensorRef] =
  ## Gradient of tanh: d/dx(tanh(x)) = 1 - tanh(x)^2
  @[grad]  # Placeholder

# Binary operation gradients

proc gradAdd*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of addition: d/dx(a+b) = (1, 1)
  @[grad, grad]

proc gradSub*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of subtraction: d/dx(a-b) = (1, -1)
  @[grad, grad]  # Placeholder: second should be negated

proc gradMul*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of multiplication: d/dx(a*b) = (b, a)
  if inputs.len >= 2:
    @[inputs[1], inputs[0]]  # Placeholder: would multiply by grad
  else:
    @[grad]

proc gradDiv*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of division: d/dx(a/b) = (1/b, -a/b^2)
  @[grad, grad]  # Placeholder

proc gradPow*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of power: d/dx(a^b) = (b*a^(b-1), a^b*ln(a))
  @[grad, grad]  # Placeholder

# Matrix operation gradients

proc gradMatMul*(grad: TensorRef, inputs: seq[TensorRef],
                 output: TensorRef): seq[TensorRef] =
  ## Gradient of matrix multiplication: C = A @ B
  ## dL/dA = dL/dC @ B^T
  ## dL/dB = A^T @ dL/dC
  @[grad, grad]  # Placeholder

proc gradTranspose*(grad: TensorRef, inputs: seq[TensorRef],
                    output: TensorRef): seq[TensorRef] =
  ## Gradient of transpose: just transpose the gradient
  @[grad]  # Placeholder

# Activation function gradients

proc gradRelu*(grad: TensorRef, inputs: seq[TensorRef],
               output: TensorRef): seq[TensorRef] =
  ## Gradient of ReLU: d/dx(max(0,x)) = 1 if x > 0 else 0
  @[grad]  # Placeholder: would mask by (input > 0)

proc gradSigmoid*(grad: TensorRef, inputs: seq[TensorRef],
                  output: TensorRef): seq[TensorRef] =
  ## Gradient of sigmoid: d/dx(σ(x)) = σ(x)(1-σ(x))
  @[output]  # Placeholder: would compute output * (1 - output) * grad

proc gradSoftmax*(grad: TensorRef, inputs: seq[TensorRef],
                  output: TensorRef): seq[TensorRef] =
  ## Gradient of softmax (Jacobian)
  @[grad]  # Placeholder: complex Jacobian computation

proc gradLeakyRelu*(grad: TensorRef, inputs: seq[TensorRef],
                    output: TensorRef): seq[TensorRef] =
  ## Gradient of Leaky ReLU: 1 if x > 0 else alpha
  @[grad]  # Placeholder

proc gradElu*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of ELU
  @[grad]  # Placeholder

proc gradGelu*(grad: TensorRef, inputs: seq[TensorRef],
               output: TensorRef): seq[TensorRef] =
  ## Gradient of GELU (approximate)
  @[grad]  # Placeholder

# Reduction operation gradients

proc gradSum*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of sum: broadcast gradient to input shape
  @[grad]  # Placeholder: would broadcast

proc gradMean*(grad: TensorRef, inputs: seq[TensorRef],
               output: TensorRef): seq[TensorRef] =
  ## Gradient of mean: broadcast gradient / n
  @[grad]  # Placeholder

# Loss function gradients

proc gradMseLoss*(grad: TensorRef, inputs: seq[TensorRef],
                  output: TensorRef): seq[TensorRef] =
  ## Gradient of MSE loss: d/dx(mean((y-x)^2)) = 2(x-y)/n
  @[grad, grad]  # Placeholder

proc gradCrossEntropyLoss*(grad: TensorRef, inputs: seq[TensorRef],
                           output: TensorRef): seq[TensorRef] =
  ## Gradient of cross-entropy loss
  @[grad, grad]  # Placeholder

# Register all standard gradients

proc registerStandardGradients*() =
  ## Register all standard gradient functions with the global registry
  # Unary ops
  registerGradient(opNeg, gradNeg)
  registerGradient(opAbs, gradAbs)
  registerGradient(opExp, gradExp)
  registerGradient(opLog, gradLog)
  registerGradient(opSqrt, gradSqrt)
  registerGradient(opSquare, gradSquare)
  registerGradient(opSin, gradSin)
  registerGradient(opCos, gradCos)
  registerGradient(opTanh, gradTanh)

  # Binary ops
  registerGradient(opAdd, gradAdd)
  registerGradient(opSub, gradSub)
  registerGradient(opMul, gradMul)
  registerGradient(opDiv, gradDiv)
  registerGradient(opPow, gradPow)

  # Matrix ops
  registerGradient(opMatMul, gradMatMul)
  registerGradient(opTranspose, gradTranspose)

  # Activations
  registerGradient(opRelu, gradRelu)
  registerGradient(opSigmoid, gradSigmoid)
  registerGradient(opSoftmax, gradSoftmax)

  # Reductions
  registerGradient(opSum, gradSum)
  registerGradient(opMean, gradMean)

# Auto-register on import
registerStandardGradients()
