## Integration Tests for Autograd
##
## Tests the full flow: tape recording -> backward pass -> gradient computation

import unittest
import std/[math, tables]
import ml_core
import ../src/ml_autograd/tensor_ops
import ../src/ml_autograd/tape
import ../src/ml_autograd/backward
import ../src/ml_autograd/gradients

# Helper to create TensorRef with data
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

suite "Integration: Tape to Backward":
  test "full forward-backward flow with negation":
    let tape = newGradientTape()
    let x = newTestTensorRef(newShape(3), dtFloat32, @[1.0'f32, 2.0, 3.0])
    let y = newTestTensorRef(newShape(3), dtFloat32, @[-1.0'f32, -2.0, -3.0])

    tape.watch(x)
    tape.record(opNeg, @[x], y, gradNeg)

    check tape.len == 1

    # Run backward
    let outputGrad = ones(newShape(3), dtFloat32)
    let grads = tape.backward(y, outputGrad)

    # Check gradient exists for x
    check x.hash in grads
    let xGrad = grads[x.hash]
    check not xGrad.isNil
    # grad_x should be -1 (negation gradient)
    let values = getValues(xGrad)
    check abs(values[0] - (-1.0'f32)) < 0.001
    check abs(values[1] - (-1.0'f32)) < 0.001
    check abs(values[2] - (-1.0'f32)) < 0.001

  test "full forward-backward flow with addition":
    let tape = newGradientTape()
    let a = newTestTensorRef(newShape(3), dtFloat32, @[1.0'f32, 2.0, 3.0])
    let b = newTestTensorRef(newShape(3), dtFloat32, @[4.0'f32, 5.0, 6.0])
    let c = newTestTensorRef(newShape(3), dtFloat32, @[5.0'f32, 7.0, 9.0])

    tape.watch(a)
    tape.watch(b)
    tape.record(opAdd, @[a, b], c, gradAdd)

    let outputGrad = ones(newShape(3), dtFloat32)
    let grads = tape.backward(c, outputGrad)

    # Both a and b should have gradient = 1 (addition rule)
    check a.hash in grads
    check b.hash in grads

  test "chained operations: y = -(-x)":
    let tape = newGradientTape()
    let x = newTestTensorRef(newShape(2), dtFloat32, @[1.0'f32, 2.0])
    let y1 = neg(x)  # y1 = -x
    let y2 = neg(y1) # y2 = -y1 = x

    tape.watch(x)
    tape.record(opNeg, @[x], y1, gradNeg)
    tape.record(opNeg, @[y1], y2, gradNeg)

    check tape.len == 2

    let outputGrad = ones(newShape(2), dtFloat32)
    let ctx = newGradientContext(tape)
    ctx.setGrad(y2, outputGrad)
    ctx.setGrad(y1, outputGrad)  # Set intermediate gradient

    ctx.backward(outputGrad, newBackwardOptions(retainGraph = true))

    # gradient of y2 w.r.t x should be 1 (double negation cancels)
    check ctx.hasGrad(x)

  test "multiplication gradient flow":
    let tape = newGradientTape()
    let a = newTestTensorRef(newShape(2), dtFloat32, @[2.0'f32, 3.0])
    let b = newTestTensorRef(newShape(2), dtFloat32, @[4.0'f32, 5.0])
    let c = mul(a, b)

    tape.watch(a)
    tape.watch(b)
    tape.record(opMul, @[a, b], c, gradMul)

    let outputGrad = ones(newShape(2), dtFloat32)
    let grads = tape.backward(c, outputGrad)

    # grad_a = grad * b = [4, 5]
    # grad_b = grad * a = [2, 3]
    check a.hash in grads
    check b.hash in grads

suite "Integration: Multiple Operations":
  test "y = (a + b) * c":
    let tape = newGradientTape()
    let a = newTestTensorRef(newShape(2), dtFloat32, @[1.0'f32, 2.0])
    let b = newTestTensorRef(newShape(2), dtFloat32, @[3.0'f32, 4.0])
    let c = newTestTensorRef(newShape(2), dtFloat32, @[2.0'f32, 2.0])

    let sum_ab = add(a, b)  # [4, 6]
    let y = mul(sum_ab, c)  # [8, 12]

    tape.watch(a)
    tape.watch(b)
    tape.watch(c)
    tape.record(opAdd, @[a, b], sum_ab, gradAdd)
    tape.record(opMul, @[sum_ab, c], y, gradMul)

    check tape.len == 2

    let outputGrad = ones(newShape(2), dtFloat32)
    let ctx = newGradientContext(tape)
    ctx.setGrad(y, outputGrad)

    # Need to set intermediate gradients for full propagation
    let mulGrads = gradMul(outputGrad, @[sum_ab, c], y)
    ctx.setGrad(sum_ab, mulGrads[0])

    ctx.backward(outputGrad, newBackwardOptions(retainGraph = true))

    check ctx.hasGrad(a)
    check ctx.hasGrad(b)
    check ctx.hasGrad(c)

  test "y = exp(x) gradient":
    let tape = newGradientTape()
    let x = newTestTensorRef(newShape(2), dtFloat32, @[0.0'f32, 1.0])
    let y = expRef(x)  # [1, e]

    tape.watch(x)
    tape.record(opExp, @[x], y, gradExp)

    let outputGrad = ones(newShape(2), dtFloat32)
    let grads = tape.backward(y, outputGrad)

    # grad_x = grad * exp(x) = exp(x)
    check x.hash in grads
    let xGrad = grads[x.hash]
    let values = getValues(xGrad)
    check abs(values[0] - 1.0'f32) < 0.01  # exp(0) = 1
    check abs(values[1] - E.float32) < 0.01  # exp(1) = e

  test "y = tanh(x) gradient":
    let tape = newGradientTape()
    let x = newTestTensorRef(newShape(2), dtFloat32, @[0.0'f32, 1.0])
    let y = tanhRef(x)

    tape.watch(x)
    tape.record(opTanh, @[x], y, gradTanh)

    let outputGrad = ones(newShape(2), dtFloat32)
    let grads = tape.backward(y, outputGrad)

    # grad_x = grad * (1 - tanh(x)^2)
    check x.hash in grads
    let xGrad = grads[x.hash]
    let values = getValues(xGrad)
    # At x=0: tanh(0)=0, grad = 1 - 0 = 1
    check abs(values[0] - 1.0'f32) < 0.01

suite "Integration: Matrix Operations":
  test "matmul backward":
    let tape = newGradientTape()

    # A: 2x3, B: 3x2 -> C: 2x2
    let aTd = newTensorDataZeros(newShape(2, 3), dtFloat32)
    aTd.fillFloat32(1.0)
    let a = newComputedTensorRef(aTd)

    let bTd = newTensorDataZeros(newShape(3, 2), dtFloat32)
    bTd.fillFloat32(1.0)
    let b = newComputedTensorRef(bTd)

    let c = matmul(a, b)

    tape.watch(a)
    tape.watch(b)
    tape.record(opMatMul, @[a, b], c, gradMatMul)

    let outputGrad = ones(newShape(2, 2), dtFloat32)
    let grads = tape.backward(c, outputGrad)

    check a.hash in grads
    check b.hash in grads

    # grad_a shape should be 2x3
    let gradA = grads[a.hash]
    check gradA.shape == newShape(2, 3)

    # grad_b shape should be 3x2
    let gradB = grads[b.hash]
    check gradB.shape == newShape(3, 2)

  test "transpose backward":
    let tape = newGradientTape()

    let xTd = newTensorDataZeros(newShape(2, 3), dtFloat32)
    xTd.fillFloat32(1.0)
    let x = newComputedTensorRef(xTd)
    let y = transpose(x)  # 3x2

    tape.watch(x)
    tape.record(opTranspose, @[x], y, gradTranspose)

    let outputGrad = ones(newShape(3, 2), dtFloat32)
    let grads = tape.backward(y, outputGrad)

    check x.hash in grads
    # grad_x should be transposed back to 2x3
    check grads[x.hash].shape == newShape(2, 3)

suite "Integration: Activation Functions":
  test "relu backward":
    let tape = newGradientTape()
    let x = newTestTensorRef(newShape(4), dtFloat32, @[-1.0'f32, 0.0, 1.0, 2.0])
    let y = newTestTensorRef(newShape(4), dtFloat32, @[0.0'f32, 0.0, 1.0, 2.0])

    tape.watch(x)
    tape.record(opRelu, @[x], y, gradRelu)

    let outputGrad = ones(newShape(4), dtFloat32)
    let grads = tape.backward(y, outputGrad)

    check x.hash in grads
    let xGrad = grads[x.hash]
    let values = getValues(xGrad)
    # ReLU gradient: 0 for x <= 0, 1 for x > 0
    check values[0] == 0.0'f32
    check values[1] == 0.0'f32
    check values[2] == 1.0'f32
    check values[3] == 1.0'f32

  test "sigmoid backward":
    let tape = newGradientTape()
    let x = newTestTensorRef(newShape(2), dtFloat32, @[0.0'f32, 1.0])
    let y = sigmoidRef(x)

    tape.watch(x)
    tape.record(opSigmoid, @[x], y, gradSigmoid)

    let outputGrad = ones(newShape(2), dtFloat32)
    let grads = tape.backward(y, outputGrad)

    check x.hash in grads
    let xGrad = grads[x.hash]
    let values = getValues(xGrad)
    # sigmoid'(0) = 0.5 * 0.5 = 0.25
    check abs(values[0] - 0.25'f32) < 0.01

suite "Integration: Reduction Operations":
  test "sum backward":
    let tape = newGradientTape()
    let x = newTestTensorRef(newShape(4), dtFloat32, @[1.0'f32, 2.0, 3.0, 4.0])
    let y = newTestTensorRef(newShape(1), dtFloat32, @[10.0'f32])

    tape.watch(x)
    tape.record(opSum, @[x], y, gradSum)

    let outputGrad = ones(newShape(1), dtFloat32)
    let grads = tape.backward(y, outputGrad)

    check x.hash in grads
    let xGrad = grads[x.hash]
    check xGrad.shape == newShape(4)
    # Sum gradient broadcasts: all elements get the gradient
    let values = getValues(xGrad)
    check values[0] == 1.0'f32
    check values[1] == 1.0'f32
    check values[2] == 1.0'f32
    check values[3] == 1.0'f32

  test "mean backward":
    let tape = newGradientTape()
    let x = newTestTensorRef(newShape(4), dtFloat32, @[1.0'f32, 2.0, 3.0, 4.0])
    let y = newTestTensorRef(newShape(1), dtFloat32, @[2.5'f32])

    tape.watch(x)
    tape.record(opMean, @[x], y, gradMean)

    let outputGrad = ones(newShape(1), dtFloat32)
    let grads = tape.backward(y, outputGrad)

    check x.hash in grads
    let xGrad = grads[x.hash]
    check xGrad.shape == newShape(4)
    # Mean gradient = 1/n for each element
    let values = getValues(xGrad)
    check abs(values[0] - 0.25'f32) < 0.01
    check abs(values[1] - 0.25'f32) < 0.01

suite "Integration: computeGradients API":
  test "computeGradients returns ordered gradients":
    let tape = newGradientTape()
    let w = newTestTensorRef(newShape(2), dtFloat32, @[1.0'f32, 2.0])
    let b = newTestTensorRef(newShape(2), dtFloat32, @[0.1'f32, 0.2])
    let x = newTestTensorRef(newShape(2), dtFloat32, @[1.0'f32, 1.0])

    # Simple linear: y = w * x + b
    let wx = mul(w, x)
    let y = add(wx, b)

    tape.watch(w)
    tape.watch(b)
    tape.record(opMul, @[w, x], wx, gradMul)
    tape.record(opAdd, @[wx, b], y, gradAdd)

    # Set up gradient context
    let ctx = newGradientContext(tape)
    let outputGrad = ones(newShape(2), dtFloat32)
    ctx.setGrad(y, outputGrad)

    # Backward propagation
    let addGrads = gradAdd(outputGrad, @[wx, b], y)
    ctx.setGrad(wx, addGrads[0])
    ctx.setGrad(b, addGrads[1])

    ctx.backward(outputGrad, newBackwardOptions(retainGraph = true))

    check ctx.hasGrad(w)
    check ctx.hasGrad(b)

suite "Integration: autogradOp":
  test "autogradOp automatically gets gradient function":
    let tape = newGradientTape()
    let x = newTestTensorRef(newShape(3), dtFloat32, @[1.0'f32, 2.0, 3.0])
    let y = neg(x)

    tape.watch(x)
    tape.autogradOp(opNeg, @[x], y)

    check tape.len == 1
    check tape.entries[0].gradFn != nil

  test "autogradOp skips unregistered operations":
    let tape = newGradientTape()
    let x = newTestTensorRef(newShape(3), dtFloat32, @[1.0'f32, 2.0, 3.0])
    let y = newTestTensorRef(newShape(3), dtFloat32, @[1.0'f32, 2.0, 3.0])

    tape.watch(x)
    # opConv2d is not registered
    tape.autogradOp(opConv2d, @[x], y)

    check tape.len == 1
    # gradFn should be nil for unregistered op
    check tape.entries[0].gradFn.isNil

suite "Integration: Error Handling":
  test "backward with shape mismatch raises":
    let tape = newGradientTape()
    let x = newTestTensorRef(newShape(3), dtFloat32, @[1.0'f32, 2.0, 3.0])
    let y = newTestTensorRef(newShape(3), dtFloat32, @[1.0'f32, 2.0, 3.0])

    # Create a bad gradient function that returns wrong shape
    let badGradFn = proc(grad: TensorRef, inputs: seq[TensorRef],
                         output: TensorRef): seq[TensorRef] =
      @[newTestTensorRef(newShape(4), dtFloat32, @[1.0'f32, 2.0, 3.0, 4.0])]

    tape.watch(x)
    tape.record(opNeg, @[x], y, badGradFn)

    let ctx = newGradientContext(tape)
    ctx.setGrad(y, ones(newShape(3), dtFloat32))

    expect BackwardError:
      ctx.backward(ones(newShape(3), dtFloat32), newBackwardOptions(checkShapes = true))

  test "backward without checkShapes succeeds":
    let tape = newGradientTape()
    let x = newTestTensorRef(newShape(3), dtFloat32, @[1.0'f32, 2.0, 3.0])
    let y = newTestTensorRef(newShape(3), dtFloat32, @[1.0'f32, 2.0, 3.0])

    # Create a gradient function that returns wrong shape
    let badGradFn = proc(grad: TensorRef, inputs: seq[TensorRef],
                         output: TensorRef): seq[TensorRef] =
      @[newTestTensorRef(newShape(4), dtFloat32, @[1.0'f32, 2.0, 3.0, 4.0])]

    tape.watch(x)
    tape.record(opNeg, @[x], y, badGradFn)

    let ctx = newGradientContext(tape)
    ctx.setGrad(y, ones(newShape(3), dtFloat32))

    # With checkShapes = false, should not raise
    ctx.backward(ones(newShape(3), dtFloat32), newBackwardOptions(checkShapes = false))

suite "Integration: Gradient Registry":
  test "all standard gradients are registered":
    check hasGradientFn(opNeg)
    check hasGradientFn(opAbs)
    check hasGradientFn(opExp)
    check hasGradientFn(opLog)
    check hasGradientFn(opSqrt)
    check hasGradientFn(opSquare)
    check hasGradientFn(opSin)
    check hasGradientFn(opCos)
    check hasGradientFn(opTanh)
    check hasGradientFn(opAdd)
    check hasGradientFn(opSub)
    check hasGradientFn(opMul)
    check hasGradientFn(opDiv)
    check hasGradientFn(opPow)
    check hasGradientFn(opMatMul)
    check hasGradientFn(opTranspose)
    check hasGradientFn(opRelu)
    check hasGradientFn(opSigmoid)
    check hasGradientFn(opSoftmax)
    check hasGradientFn(opSum)
    check hasGradientFn(opMean)

  test "getGradientFn returns correct function":
    let fn = getGradientFn(opAdd)
    check not fn.isNil

    let grad = ones(newShape(3), dtFloat32)
    let a = newTestTensorRef(newShape(3), dtFloat32, @[1.0'f32, 2.0, 3.0])
    let b = newTestTensorRef(newShape(3), dtFloat32, @[4.0'f32, 5.0, 6.0])
    let output = add(a, b)

    let grads = fn(grad, @[a, b], output)
    check grads.len == 2
    check grads[0] == grad
    check grads[1] == grad

  test "registerGradient overwrites existing":
    let customGradFn = proc(grad: TensorRef, inputs: seq[TensorRef],
                            output: TensorRef): seq[TensorRef] =
      @[grad]

    # Save original
    let originalFn = getGradientFn(opNeg)

    # Register custom
    registerGradient(opNeg, customGradFn)
    check getGradientFn(opNeg) == customGradFn

    # Restore original
    registerGradient(opNeg, originalFn)
