## Tests for gradient functions module

import unittest
import std/[tables]
import ml_core
import ../src/ml_autograd/tape
import ../src/ml_autograd/backward
import ../src/ml_autograd/gradients

# Helper to create TensorRef with unique hash
var tensorCounter {.global.} = 0

proc newUniqueTensorRef(shape: Shape, dtype: DType): TensorRef =
  inc tensorCounter
  var uniqueHash: Hash256
  let counter = tensorCounter
  uniqueHash[0] = byte(counter and 0xFF)
  uniqueHash[1] = byte((counter shr 8) and 0xFF)
  uniqueHash[2] = byte((counter shr 16) and 0xFF)
  uniqueHash[3] = byte((counter shr 24) and 0xFF)
  newTensorRef(uniqueHash, shape, dtype)

suite "Unary Gradient Functions":
  test "gradNeg":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = gradNeg(grad, @[input], output)
    check grads.len == 1

  test "gradAbs":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = gradAbs(grad, @[input], output)
    check grads.len == 1

  test "gradExp":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = gradExp(grad, @[input], output)
    check grads.len == 1

  test "gradLog":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = gradLog(grad, @[input], output)
    check grads.len == 1

  test "gradSqrt":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = gradSqrt(grad, @[input], output)
    check grads.len == 1

  test "gradTanh":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = gradTanh(grad, @[input], output)
    check grads.len == 1

suite "Binary Gradient Functions":
  test "gradAdd":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let a = newUniqueTensorRef(newShape(5), dtFloat32)
    let b = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = gradAdd(grad, @[a, b], output)
    check grads.len == 2
    check grads[0] == grad  # Both get same gradient
    check grads[1] == grad

  test "gradSub":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let a = newUniqueTensorRef(newShape(5), dtFloat32)
    let b = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = gradSub(grad, @[a, b], output)
    check grads.len == 2

  test "gradMul":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let a = newUniqueTensorRef(newShape(5), dtFloat32)
    let b = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = gradMul(grad, @[a, b], output)
    check grads.len == 2
    # In multiplication, gradient for a is b and vice versa
    check grads[0] == b
    check grads[1] == a

  test "gradDiv":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let a = newUniqueTensorRef(newShape(5), dtFloat32)
    let b = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = gradDiv(grad, @[a, b], output)
    check grads.len == 2

  test "gradPow":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let a = newUniqueTensorRef(newShape(5), dtFloat32)
    let b = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = gradPow(grad, @[a, b], output)
    check grads.len == 2

suite "Matrix Gradient Functions":
  test "gradMatMul":
    let grad = newUniqueTensorRef(newShape(5, 5), dtFloat32)
    let a = newUniqueTensorRef(newShape(5, 5), dtFloat32)
    let b = newUniqueTensorRef(newShape(5, 5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5, 5), dtFloat32)

    let grads = gradMatMul(grad, @[a, b], output)
    check grads.len == 2

  test "gradTranspose":
    let grad = newUniqueTensorRef(newShape(5, 3), dtFloat32)
    let input = newUniqueTensorRef(newShape(3, 5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5, 3), dtFloat32)

    let grads = gradTranspose(grad, @[input], output)
    check grads.len == 1

suite "Activation Gradient Functions":
  test "gradRelu":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = gradRelu(grad, @[input], output)
    check grads.len == 1

  test "gradSigmoid":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = gradSigmoid(grad, @[input], output)
    check grads.len == 1

  test "gradSoftmax":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = gradSoftmax(grad, @[input], output)
    check grads.len == 1

  test "gradLeakyRelu":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = gradLeakyRelu(grad, @[input], output)
    check grads.len == 1

  test "gradElu":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = gradElu(grad, @[input], output)
    check grads.len == 1

  test "gradGelu":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = gradGelu(grad, @[input], output)
    check grads.len == 1

suite "Reduction Gradient Functions":
  test "gradSum":
    let grad = newUniqueTensorRef(newShape(1), dtFloat32)
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(1), dtFloat32)

    let grads = gradSum(grad, @[input], output)
    check grads.len == 1

  test "gradMean":
    let grad = newUniqueTensorRef(newShape(1), dtFloat32)
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(1), dtFloat32)

    let grads = gradMean(grad, @[input], output)
    check grads.len == 1

suite "Loss Gradient Functions":
  test "gradMseLoss":
    let grad = newUniqueTensorRef(newShape(1), dtFloat32)
    let pred = newUniqueTensorRef(newShape(5), dtFloat32)
    let target = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(1), dtFloat32)

    let grads = gradMseLoss(grad, @[pred, target], output)
    check grads.len == 2

  test "gradCrossEntropyLoss":
    let grad = newUniqueTensorRef(newShape(1), dtFloat32)
    let pred = newUniqueTensorRef(newShape(5), dtFloat32)
    let target = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(1), dtFloat32)

    let grads = gradCrossEntropyLoss(grad, @[pred, target], output)
    check grads.len == 2

suite "Standard Gradient Registration":
  test "standard gradients are registered":
    # After importing gradients module, standard grads should be registered
    check hasGradientFn(opNeg)
    check hasGradientFn(opAdd)
    check hasGradientFn(opMul)
    check hasGradientFn(opMatMul)
    check hasGradientFn(opRelu)
    check hasGradientFn(opSigmoid)

  test "can retrieve registered gradient function":
    let fn = getGradientFn(opAdd)
    check not fn.isNil

    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let a = newUniqueTensorRef(newShape(5), dtFloat32)
    let b = newUniqueTensorRef(newShape(5), dtFloat32)
    let out1 = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = fn(grad, @[a, b], out1)
    check grads.len == 2
