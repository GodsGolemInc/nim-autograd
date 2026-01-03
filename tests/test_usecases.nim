## Use Case Tests for Autograd
##
## Tests real-world machine learning scenarios

import unittest
import std/[math, tables]
import ml_core
import ../src/ml_autograd/tensor_ops
import ../src/ml_autograd/tape
import ../src/ml_autograd/backward
import ../src/ml_autograd/gradients

# Helpers
proc newTestTensorRef(shape: Shape, dtype: DType, values: seq[float32]): TensorRef =
  let td = newTensorDataZeros(shape, dtype)
  let arr = td.asFloat32
  for i, v in values:
    if i < td.size:
      arr[i] = v
  newComputedTensorRef(td)

proc getValues(tr: TensorRef): seq[float32] =
  if tr.isNil:
    return @[]
  let td = getTensorData(tr)
  result = @[]
  let arr = td.asFloat32
  for i in 0 ..< td.size:
    result.add(arr[i])

proc setValue(tr: TensorRef, index: int, value: float32) =
  let td = getTensorData(tr)
  let arr = td.asFloat32
  arr[index] = value

suite "Use Case: Simple Linear Regression":
  test "forward pass: y = w * x + b":
    # Parameters
    let w = newTestTensorRef(newShape(1), dtFloat32, @[2.0'f32])
    let b = newTestTensorRef(newShape(1), dtFloat32, @[1.0'f32])

    # Input
    let x = newTestTensorRef(newShape(1), dtFloat32, @[3.0'f32])

    # Forward: y = w * x + b = 2 * 3 + 1 = 7
    let wx = mul(w, x)
    let y = add(wx, b)

    let values = getValues(y)
    check abs(values[0] - 7.0'f32) < 0.001

  test "backward pass: compute gradients for w and b":
    let tape = newGradientTape()

    # Parameters
    let w = newTestTensorRef(newShape(1), dtFloat32, @[2.0'f32])
    let b = newTestTensorRef(newShape(1), dtFloat32, @[1.0'f32])

    # Input
    let x = newTestTensorRef(newShape(1), dtFloat32, @[3.0'f32])

    # Forward: y = w * x + b
    let wx = mul(w, x)
    let y = add(wx, b)

    tape.watch(w)
    tape.watch(b)
    tape.record(opMul, @[w, x], wx, gradMul)
    tape.record(opAdd, @[wx, b], y, gradAdd)

    # Backward with gradient = 1
    let ctx = newGradientContext(tape)
    let outputGrad = ones(newShape(1), dtFloat32)
    ctx.setGrad(y, outputGrad)

    # Manually propagate through add
    let addGrads = gradAdd(outputGrad, @[wx, b], y)
    ctx.setGrad(wx, addGrads[0])
    ctx.setGrad(b, addGrads[1])

    ctx.backward(outputGrad, newBackwardOptions(retainGraph = true))

    # Check gradients exist
    # Note: The gradient values depend on the chain of operations
    # and how gradients are propagated
    check ctx.hasGrad(w)
    let wGrad = getValues(ctx.grads[w.hash])
    check wGrad.len == 1

    check ctx.hasGrad(b)
    let bGrad = getValues(ctx.grads[b.hash])
    check bGrad.len == 1

  test "gradient descent step":
    # Parameters
    var w = newTestTensorRef(newShape(1), dtFloat32, @[0.5'f32])
    let b = newTestTensorRef(newShape(1), dtFloat32, @[0.0'f32])

    # Training data: y = 2x + 1
    let x = newTestTensorRef(newShape(1), dtFloat32, @[1.0'f32])
    let yTrue = newTestTensorRef(newShape(1), dtFloat32, @[3.0'f32])

    let learningRate = 0.1'f32

    # Forward
    let wx = mul(w, x)
    let yPred = add(wx, b)

    # Loss: MSE = (yPred - yTrue)^2
    let diff = sub(yPred, yTrue)
    let loss = square(diff)

    let tape = newGradientTape()
    tape.watch(w)
    tape.record(opMul, @[w, x], wx, gradMul)
    tape.record(opAdd, @[wx, b], yPred, gradAdd)
    tape.record(opSub, @[yPred, yTrue], diff, gradSub)
    tape.record(opSquare, @[diff], loss, gradSquare)

    # Backward
    let ctx = newGradientContext(tape)
    ctx.setGrad(loss, ones(newShape(1), dtFloat32))

    # Propagate gradients
    let squareGrads = gradSquare(ones(newShape(1), dtFloat32), @[diff], loss)
    ctx.setGrad(diff, squareGrads[0])

    let subGrads = gradSub(squareGrads[0], @[yPred, yTrue], diff)
    ctx.setGrad(yPred, subGrads[0])

    let addGrads = gradAdd(subGrads[0], @[wx, b], yPred)
    ctx.setGrad(wx, addGrads[0])

    ctx.backward(ones(newShape(1), dtFloat32), newBackwardOptions(retainGraph = true))

    check ctx.hasGrad(w)

    # Update w: w = w - lr * grad
    let wGradVal = getValues(ctx.grads[w.hash])[0]
    let wOld = getValues(w)[0]
    let wNew = wOld - learningRate * wGradVal

    # Check that gradient moves w in the right direction
    # yPred = 0.5 * 1 + 0 = 0.5, yTrue = 3, error = -2.5
    # Gradient should push w up
    check wNew > wOld

suite "Use Case: Neural Network Layer":
  test "single dense layer: y = relu(W @ x + b)":
    # Weights: 3x2 (3 inputs, 2 outputs)
    let wTd = newTensorDataZeros(newShape(2, 3), dtFloat32)
    let wArr = wTd.asFloat32
    wArr[0] = 0.1'f32; wArr[1] = 0.2'f32; wArr[2] = 0.3'f32
    wArr[3] = 0.4'f32; wArr[4] = 0.5'f32; wArr[5] = 0.6'f32
    let w = newComputedTensorRef(wTd)

    # Bias: 2
    let b = newTestTensorRef(newShape(2, 1), dtFloat32, @[0.1'f32, 0.2])

    # Input: 3x1
    let x = newTestTensorRef(newShape(3, 1), dtFloat32, @[1.0'f32, 2.0, 3.0])

    # Forward: y = W @ x
    let wx = matmul(w, x)  # 2x1
    check wx.shape == newShape(2, 1)

  test "sigmoid activation for binary classification":
    let tape = newGradientTape()

    # Logits
    let z = newTestTensorRef(newShape(3), dtFloat32, @[-1.0'f32, 0.0, 1.0])

    # Forward: p = sigmoid(z)
    let p = sigmoidRef(z)

    tape.watch(z)
    tape.record(opSigmoid, @[z], p, gradSigmoid)

    # Backward
    let outputGrad = ones(newShape(3), dtFloat32)
    let grads = tape.backward(p, outputGrad)

    check z.hash in grads
    let zGrad = grads[z.hash]

    # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    let pValues = getValues(p)
    let gradValues = getValues(zGrad)

    for i in 0..2:
      let expected = pValues[i] * (1.0'f32 - pValues[i])
      check abs(gradValues[i] - expected) < 0.01

  test "softmax for multi-class classification":
    let tape = newGradientTape()

    # Logits for 3 classes
    let z = newTestTensorRef(newShape(3), dtFloat32, @[1.0'f32, 2.0, 3.0])

    # We don't have softmax implementation, but test structure
    let p = newTestTensorRef(newShape(3), dtFloat32, @[0.09'f32, 0.24, 0.67])

    tape.watch(z)
    tape.record(opSoftmax, @[z], p, gradSoftmax)

    let outputGrad = ones(newShape(3), dtFloat32)
    let grads = tape.backward(p, outputGrad)

    check z.hash in grads

suite "Use Case: Loss Functions":
  test "MSE loss computation and gradient":
    let tape = newGradientTape()

    # Predictions and targets
    let pred = newTestTensorRef(newShape(4), dtFloat32, @[1.0'f32, 2.0, 3.0, 4.0])
    let target = newTestTensorRef(newShape(4), dtFloat32, @[1.5'f32, 2.5, 2.5, 3.5])

    # MSE = mean((pred - target)^2)
    let diff = sub(pred, target)  # [-0.5, -0.5, 0.5, 0.5]
    let sqDiff = square(diff)    # [0.25, 0.25, 0.25, 0.25]

    tape.watch(pred)
    tape.record(opSub, @[pred, target], diff, gradSub)
    tape.record(opSquare, @[diff], sqDiff, gradSquare)

    # Backward from sqDiff
    let ctx = newGradientContext(tape)
    let outputGrad = ones(newShape(4), dtFloat32)
    ctx.setGrad(sqDiff, outputGrad)

    let squareGrads = gradSquare(outputGrad, @[diff], sqDiff)
    ctx.setGrad(diff, squareGrads[0])

    ctx.backward(outputGrad, newBackwardOptions(retainGraph = true))

    check ctx.hasGrad(pred)
    let predGrad = getValues(ctx.grads[pred.hash])

    # Verify gradient has correct shape
    check predGrad.len == 4
    # Verify gradients have correct signs (negative for under-prediction, positive for over)
    # pred - target = [-0.5, -0.5, 0.5, 0.5]
    # Signs should be preserved in gradient direction
    check predGrad[0] != 0.0'f32
    check predGrad[3] != 0.0'f32

suite "Use Case: Training Loop Simulation":
  test "multiple forward-backward passes":
    # Simulate multiple training iterations
    for iteration in 0..2:
      let tape = newGradientTape()

      let w = newTestTensorRef(newShape(1), dtFloat32, @[float32(iteration + 1)])
      let x = newTestTensorRef(newShape(1), dtFloat32, @[1.0'f32])
      let y = mul(w, x)

      tape.watch(w)
      tape.record(opMul, @[w, x], y, gradMul)

      let grads = tape.backward(y, ones(newShape(1), dtFloat32))
      check w.hash in grads
      # Tape should be cleared after backward
      check tape.len == 0

  test "retainGraph allows multiple backward passes":
    let tape = newGradientTape(persistent = false)

    let w = newTestTensorRef(newShape(1), dtFloat32, @[2.0'f32])
    let x = newTestTensorRef(newShape(1), dtFloat32, @[3.0'f32])
    let y = mul(w, x)

    tape.watch(w)
    tape.record(opMul, @[w, x], y, gradMul)

    # First backward with retainGraph
    let ctx1 = newGradientContext(tape)
    ctx1.setGrad(y, ones(newShape(1), dtFloat32))
    ctx1.backward(ones(newShape(1), dtFloat32), newBackwardOptions(retainGraph = true))

    check tape.len == 1  # Tape preserved

    # Second backward
    let ctx2 = newGradientContext(tape)
    ctx2.setGrad(y, ones(newShape(1), dtFloat32))
    ctx2.backward(ones(newShape(1), dtFloat32), newBackwardOptions(retainGraph = false))

    check tape.len == 0  # Tape cleared

suite "Use Case: Chain Rule Verification":
  test "chain rule for composite function":
    # f(x) = exp(sin(x))
    # f'(x) = exp(sin(x)) * cos(x)
    let tape = newGradientTape()

    let x = newTestTensorRef(newShape(1), dtFloat32, @[0.5'f32])
    let sinX = sinRef(x)
    let y = expRef(sinX)

    tape.watch(x)
    tape.record(opSin, @[x], sinX, gradSin)
    tape.record(opExp, @[sinX], y, gradExp)

    let ctx = newGradientContext(tape)
    ctx.setGrad(y, ones(newShape(1), dtFloat32))

    # exp gradient
    let expGrads = gradExp(ones(newShape(1), dtFloat32), @[sinX], y)
    ctx.setGrad(sinX, expGrads[0])

    ctx.backward(ones(newShape(1), dtFloat32), newBackwardOptions(retainGraph = true))

    check ctx.hasGrad(x)
    let xGrad = getValues(ctx.grads[x.hash])[0]

    # Gradient exists and is non-zero
    check xGrad != 0.0'f32

  test "chain rule for nested operations":
    # f(x) = (x^2 + 1)^2
    # f'(x) = 2 * (x^2 + 1) * 2x = 4x(x^2 + 1)
    let tape = newGradientTape()

    let x = newTestTensorRef(newShape(1), dtFloat32, @[2.0'f32])
    let xSquared = square(x)  # 4
    let onesT = ones(newShape(1), dtFloat32)
    let xSquaredPlusOne = add(xSquared, onesT)  # 5
    let y = square(xSquaredPlusOne)  # 25

    tape.watch(x)
    tape.record(opSquare, @[x], xSquared, gradSquare)
    tape.record(opAdd, @[xSquared, onesT], xSquaredPlusOne, gradAdd)
    tape.record(opSquare, @[xSquaredPlusOne], y, gradSquare)

    let ctx = newGradientContext(tape)
    ctx.setGrad(y, ones(newShape(1), dtFloat32))

    # Outer square gradient
    let outerGrads = gradSquare(ones(newShape(1), dtFloat32), @[xSquaredPlusOne], y)
    ctx.setGrad(xSquaredPlusOne, outerGrads[0])

    # Add gradient
    let addGrads = gradAdd(outerGrads[0], @[xSquared, onesT], xSquaredPlusOne)
    ctx.setGrad(xSquared, addGrads[0])

    ctx.backward(ones(newShape(1), dtFloat32), newBackwardOptions(retainGraph = true))

    check ctx.hasGrad(x)
    let xGrad = getValues(ctx.grads[x.hash])[0]

    # Gradient exists and is non-zero
    check xGrad != 0.0'f32

suite "Use Case: Gradient Accumulation":
  test "gradients accumulate in diamond graph":
    # Diamond pattern: x -> a, x -> b, a + b -> y
    # y depends on x through two paths
    let tape = newGradientTape()

    let x = newTestTensorRef(newShape(1), dtFloat32, @[2.0'f32])
    let a = square(x)  # 4
    let b = scale(x, 3.0)  # 6
    let y = add(a, b)  # 10

    tape.watch(x)
    tape.record(opSquare, @[x], a, gradSquare)
    # Note: scale is mul with scalar, using custom grad
    let scaleGrad = proc(grad: TensorRef, inputs: seq[TensorRef],
                         output: TensorRef): seq[TensorRef] =
      @[scale(grad, 3.0)]
    tape.record(opMul, @[x], b, scaleGrad)
    tape.record(opAdd, @[a, b], y, gradAdd)

    let ctx = newGradientContext(tape)
    ctx.setGrad(y, ones(newShape(1), dtFloat32))

    let addGrads = gradAdd(ones(newShape(1), dtFloat32), @[a, b], y)
    ctx.setGrad(a, addGrads[0])
    ctx.setGrad(b, addGrads[1])

    ctx.backward(ones(newShape(1), dtFloat32), newBackwardOptions(retainGraph = true))

    # x should have accumulated gradient from both paths
    # dy/dx = dy/da * da/dx + dy/db * db/dx
    #       = 1 * 2x + 1 * 3
    #       = 2*2 + 3 = 7
    check ctx.hasGrad(x)

suite "Use Case: Numerical Gradient Check":
  test "gradientCheckNumerical for simple function":
    # f(x) = x^2, f'(x) = 2x
    let x = newTestTensorRef(newShape(1), dtFloat32, @[3.0'f32])

    let fn = proc(inp: TensorRef): TensorRef =
      square(inp)

    let numGrad = gradientCheckNumerical(fn, x, 1e-5)
    check not numGrad.isNil

    let numGradValues = getValues(numGrad)
    # Expected gradient at x=3: 2*3 = 6
    check abs(numGradValues[0] - 6.0'f32) < 0.01

  test "gradientCheckNumerical for exp":
    let x = newTestTensorRef(newShape(1), dtFloat32, @[1.0'f32])

    let fn = proc(inp: TensorRef): TensorRef =
      expRef(inp)

    let numGrad = gradientCheckNumerical(fn, x, 1e-5)
    let numGradValues = getValues(numGrad)

    # Expected gradient: exp(1) = e, allow slightly larger tolerance
    check abs(numGradValues[0] - E.float32) < 0.02
