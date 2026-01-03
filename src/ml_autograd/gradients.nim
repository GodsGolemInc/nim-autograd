## Common Gradient Functions
##
## Provides gradient implementations for common ML operations.
## These are registered with the global gradient registry.

import ml_core
import ./backward
import ./tensor_ops

# Unary operation gradients

proc gradNeg*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of negation: d/dx(-x) = -1
  ## Result: -grad
  @[neg(grad)]

proc gradAbs*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of abs: d/dx(|x|) = sign(x)
  ## Result: grad * sign(input)
  if inputs.len == 0:
    return @[grad]
  @[mul(grad, sign(inputs[0]))]

proc gradExp*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of exp: d/dx(e^x) = e^x
  ## Result: grad * output (output = exp(input))
  @[mul(grad, output)]

proc gradLog*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of log: d/dx(ln(x)) = 1/x
  ## Result: grad / input
  if inputs.len == 0:
    return @[grad]
  @[`div`(grad, inputs[0])]

proc gradSqrt*(grad: TensorRef, inputs: seq[TensorRef],
               output: TensorRef): seq[TensorRef] =
  ## Gradient of sqrt: d/dx(sqrt(x)) = 1/(2*sqrt(x)) = 1/(2*output)
  ## Result: grad / (2 * output)
  let two = scale(output, 2.0)
  @[`div`(grad, two)]

proc gradSquare*(grad: TensorRef, inputs: seq[TensorRef],
                 output: TensorRef): seq[TensorRef] =
  ## Gradient of square: d/dx(x^2) = 2x
  ## Result: grad * 2 * input
  if inputs.len == 0:
    return @[grad]
  let twoX = scale(inputs[0], 2.0)
  @[mul(grad, twoX)]

proc gradSin*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of sin: d/dx(sin(x)) = cos(x)
  ## Result: grad * cos(input)
  if inputs.len == 0:
    return @[grad]
  @[mul(grad, cosRef(inputs[0]))]

proc gradCos*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of cos: d/dx(cos(x)) = -sin(x)
  ## Result: grad * (-sin(input))
  if inputs.len == 0:
    return @[grad]
  @[mul(grad, neg(sinRef(inputs[0])))]

proc gradTanh*(grad: TensorRef, inputs: seq[TensorRef],
               output: TensorRef): seq[TensorRef] =
  ## Gradient of tanh: d/dx(tanh(x)) = 1 - tanh(x)^2 = 1 - output^2
  ## Result: grad * (1 - output^2)
  let outputSquared = square(output)
  let onesT = ones(output.shape, output.dtype)
  let oneMinusOutputSquared = sub(onesT, outputSquared)
  @[mul(grad, oneMinusOutputSquared)]

# Binary operation gradients

proc gradAdd*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of addition: d/dx(a+b) = (1, 1)
  ## Both inputs receive the full gradient
  @[grad, grad]

proc gradSub*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of subtraction: d/dx(a-b) = (1, -1)
  ## First input receives grad, second receives -grad
  @[grad, neg(grad)]

proc gradMul*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of multiplication: d/dx(a*b) = (b*grad, a*grad)
  ## Chain rule: d/da = b, d/db = a
  if inputs.len >= 2:
    @[mul(grad, inputs[1]), mul(grad, inputs[0])]
  else:
    @[grad]

proc gradDiv*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of division: d/dx(a/b) = (1/b, -a/b^2)
  ## d/da = grad / b
  ## d/db = -grad * a / b^2 = -grad * output / b
  if inputs.len >= 2:
    let a = inputs[0]
    let b = inputs[1]
    # d/da = grad / b
    let gradA = `div`(grad, b)
    # d/db = -grad * a / b^2
    let bSquared = square(b)
    let gradB = neg(`div`(mul(grad, a), bSquared))
    @[gradA, gradB]
  else:
    @[grad]

proc gradPow*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of power: d/dx(a^b) = (b*a^(b-1), a^b*ln(a))
  ## d/da = grad * b * a^(b-1) = grad * b * output / a
  ## d/db = grad * output * ln(a)
  if inputs.len >= 2:
    let a = inputs[0]
    let b = inputs[1]
    # d/da = grad * b * output / a
    let gradA = `div`(mul(mul(grad, b), output), a)
    # d/db = grad * output * ln(a)
    let gradB = mul(mul(grad, output), logRef(a))
    @[gradA, gradB]
  else:
    @[grad]

# Matrix operation gradients

proc gradMatMul*(grad: TensorRef, inputs: seq[TensorRef],
                 output: TensorRef): seq[TensorRef] =
  ## Gradient of matrix multiplication: C = A @ B
  ## dL/dA = dL/dC @ B^T
  ## dL/dB = A^T @ dL/dC
  if inputs.len >= 2:
    let a = inputs[0]
    let b = inputs[1]
    let gradA = matmul(grad, transpose(b))
    let gradB = matmul(transpose(a), grad)
    @[gradA, gradB]
  else:
    @[grad]

proc gradTranspose*(grad: TensorRef, inputs: seq[TensorRef],
                    output: TensorRef): seq[TensorRef] =
  ## Gradient of transpose: transpose the gradient back
  @[transpose(grad)]

# Activation function gradients

proc gradRelu*(grad: TensorRef, inputs: seq[TensorRef],
               output: TensorRef): seq[TensorRef] =
  ## Gradient of ReLU: d/dx(max(0,x)) = 1 if x > 0 else 0
  ## Result: grad * (input > 0)
  if inputs.len == 0:
    return @[grad]
  @[mul(grad, reluMask(inputs[0]))]

proc gradSigmoid*(grad: TensorRef, inputs: seq[TensorRef],
                  output: TensorRef): seq[TensorRef] =
  ## Gradient of sigmoid: d/dx(σ(x)) = σ(x)(1-σ(x)) = output * (1-output)
  ## Result: grad * output * (1 - output)
  let onesT = ones(output.shape, output.dtype)
  let oneMinusOutput = sub(onesT, output)
  @[mul(grad, mul(output, oneMinusOutput))]

proc gradSoftmax*(grad: TensorRef, inputs: seq[TensorRef],
                  output: TensorRef): seq[TensorRef] =
  ## Gradient of softmax (simplified: element-wise for now)
  ## Full Jacobian: dS_i/dx_j = S_i * (δ_ij - S_j)
  ## For cross-entropy loss, gradient simplifies to (output - target)
  ## Here we use simplified form: grad * output * (1 - output)
  let onesT = ones(output.shape, output.dtype)
  let oneMinusOutput = sub(onesT, output)
  @[mul(grad, mul(output, oneMinusOutput))]

proc gradLeakyRelu*(grad: TensorRef, inputs: seq[TensorRef],
                    output: TensorRef): seq[TensorRef] =
  ## Gradient of Leaky ReLU: 1 if x > 0 else alpha
  ## Result: grad * leakyReluMask(input, alpha)
  if inputs.len == 0:
    return @[grad]
  @[mul(grad, leakyReluMask(inputs[0], 0.01))]

proc gradElu*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of ELU: d/dx(ELU(x)) = 1 if x > 0 else alpha * exp(x)
  ## For x <= 0: ELU(x) = alpha * (exp(x) - 1), so derivative = alpha * exp(x) = output + alpha
  ## Simplified: use output + 1 for negative region (alpha=1)
  if inputs.len == 0:
    return @[grad]
  # For simplicity, use ReLU-like mask for positive, (output + 1) for negative
  let mask = reluMask(inputs[0])
  let onesT = ones(inputs[0].shape, inputs[0].dtype)
  let negMask = sub(onesT, mask)  # 1 where x <= 0
  let outputPlusOne = add(output, onesT)
  let eluGrad = add(mask, mul(negMask, outputPlusOne))
  @[mul(grad, eluGrad)]

proc gradGelu*(grad: TensorRef, inputs: seq[TensorRef],
               output: TensorRef): seq[TensorRef] =
  ## Gradient of GELU (approximate)
  ## GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
  ## For simplicity, use numerical approximation: d/dx GELU(x) ≈ sigmoid(1.702 * x)
  ## + x * 1.702 * sigmoid(1.702*x) * (1 - sigmoid(1.702*x))
  if inputs.len == 0:
    return @[grad]
  let x = inputs[0]
  let scaled = scale(x, 1.702)
  let sig = sigmoidRef(scaled)
  let onesT = ones(x.shape, x.dtype)
  let oneMinusSig = sub(onesT, sig)
  let term1 = sig
  let term2 = mul(scale(mul(mul(x, sig), oneMinusSig), 1.702), onesT)
  let geluGrad = add(term1, term2)
  @[mul(grad, geluGrad)]

# Reduction operation gradients

proc gradSum*(grad: TensorRef, inputs: seq[TensorRef],
              output: TensorRef): seq[TensorRef] =
  ## Gradient of sum: broadcast gradient to input shape
  ## d/dx(sum(x)) = ones(input.shape)
  if inputs.len == 0:
    return @[grad]
  @[broadcast(grad, inputs[0].shape)]

proc gradMean*(grad: TensorRef, inputs: seq[TensorRef],
               output: TensorRef): seq[TensorRef] =
  ## Gradient of mean: broadcast gradient / n to input shape
  ## d/dx(mean(x)) = 1/n * ones(input.shape)
  if inputs.len == 0:
    return @[grad]
  let n = inputs[0].shape.size.float
  let scaledGrad = scale(grad, 1.0 / n)
  @[broadcast(scaledGrad, inputs[0].shape)]

# Loss function gradients

proc gradMseLoss*(grad: TensorRef, inputs: seq[TensorRef],
                  output: TensorRef): seq[TensorRef] =
  ## Gradient of MSE loss: L = mean((pred - target)^2)
  ## dL/d_pred = 2 * (pred - target) / n
  ## dL/d_target = -2 * (pred - target) / n
  if inputs.len >= 2:
    let pred = inputs[0]
    let target = inputs[1]
    let diff = sub(pred, target)
    let n = pred.shape.size.float
    let scaleFactor = 2.0 / n
    let gradPred = mul(grad, scale(diff, scaleFactor))
    let gradTarget = neg(gradPred)
    @[gradPred, gradTarget]
  else:
    @[grad]

proc gradCrossEntropyLoss*(grad: TensorRef, inputs: seq[TensorRef],
                           output: TensorRef): seq[TensorRef] =
  ## Gradient of cross-entropy loss (with softmax)
  ## For softmax-cross-entropy: dL/d_logits = softmax(logits) - one_hot(target)
  ## Simplified: dL/d_pred = -target / pred, dL/d_target = -log(pred)
  if inputs.len >= 2:
    let pred = inputs[0]
    let target = inputs[1]
    # dL/d_pred = grad * (-target / pred)
    let gradPred = neg(mul(grad, `div`(target, pred)))
    # dL/d_target = grad * (-log(pred))
    let gradTarget = neg(mul(grad, logRef(pred)))
    @[gradPred, gradTarget]
  else:
    @[grad]

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
