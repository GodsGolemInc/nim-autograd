## Tests for backward pass module

import unittest
import std/[tables, strutils]
import ml_core
import ../src/ml_autograd/tape
import ../src/ml_autograd/backward

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

suite "BackwardOptions":
  test "newBackwardOptions defaults":
    let opts = newBackwardOptions()
    check not opts.retainGraph
    check not opts.createGraph
    check opts.checkShapes

  test "newBackwardOptions custom":
    let opts = newBackwardOptions(retainGraph = true, createGraph = true)
    check opts.retainGraph
    check opts.createGraph

suite "Backward Pass":
  test "backward with empty tape":
    let tape = newGradientTape()
    let ctx = newGradientContext(tape)
    let outputGrad = newUniqueTensorRef(newShape(1), dtFloat32)

    # Should not raise
    ctx.backward(outputGrad)

  test "backward with single operation":
    let tape = newGradientTape()
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    # Simple gradient function that passes gradient through
    let gradFn = proc(grad: TensorRef, inputs: seq[TensorRef],
                      output: TensorRef): seq[TensorRef] =
      @[grad]

    tape.watch(input)
    tape.record(opNeg, @[input], output, gradFn)

    let ctx = newGradientContext(tape)
    let outputGrad = newUniqueTensorRef(newShape(5), dtFloat32)
    ctx.setGrad(output, outputGrad)

    ctx.backward(outputGrad)

    check ctx.hasGrad(input)

  test "backward clears tape by default":
    let tape = newGradientTape()
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let gradFn = proc(grad: TensorRef, inputs: seq[TensorRef],
                      output: TensorRef): seq[TensorRef] =
      @[grad]

    tape.watch(input)
    tape.record(opNeg, @[input], output, gradFn)
    check tape.len == 1

    let ctx = newGradientContext(tape)
    let outputGrad = newUniqueTensorRef(newShape(5), dtFloat32)
    ctx.setGrad(output, outputGrad)
    ctx.backward(outputGrad)

    check tape.len == 0  # Cleared

  test "backward with retainGraph":
    let tape = newGradientTape()
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let gradFn = proc(grad: TensorRef, inputs: seq[TensorRef],
                      output: TensorRef): seq[TensorRef] =
      @[grad]

    tape.watch(input)
    tape.record(opNeg, @[input], output, gradFn)

    let ctx = newGradientContext(tape)
    let outputGrad = newUniqueTensorRef(newShape(5), dtFloat32)
    ctx.setGrad(output, outputGrad)

    let opts = newBackwardOptions(retainGraph = true)
    ctx.backward(outputGrad, opts)

    check tape.len == 1  # Not cleared

  test "backward with multiple operations":
    let tape = newGradientTape()
    let x = newUniqueTensorRef(newShape(5), dtFloat32)
    let y = newUniqueTensorRef(newShape(5), dtFloat32)
    let z = newUniqueTensorRef(newShape(5), dtFloat32)

    # y = -x, z = -y
    let negGradFn = proc(grad: TensorRef, inputs: seq[TensorRef],
                         output: TensorRef): seq[TensorRef] =
      @[grad]

    tape.watch(x)
    tape.record(opNeg, @[x], y, negGradFn)
    tape.record(opNeg, @[y], z, negGradFn)

    let ctx = newGradientContext(tape)
    let outputGrad = newUniqueTensorRef(newShape(5), dtFloat32)
    ctx.setGrad(z, outputGrad)
    ctx.setGrad(y, outputGrad)  # Set intermediate gradient

    ctx.backward(outputGrad)

    check ctx.hasGrad(x)
    check ctx.hasGrad(y)

  test "backward without gradient function":
    let tape = newGradientTape()
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    tape.watch(input)
    tape.record(opNeg, @[input], output)  # No gradFn

    let ctx = newGradientContext(tape)
    let outputGrad = newUniqueTensorRef(newShape(5), dtFloat32)
    ctx.setGrad(output, outputGrad)

    # Should not raise, just skip operations without gradFn
    ctx.backward(outputGrad)

suite "Backward from Tape":
  test "backward returns gradient table":
    let tape = newGradientTape()
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    let gradFn = proc(grad: TensorRef, inputs: seq[TensorRef],
                      output: TensorRef): seq[TensorRef] =
      @[grad]

    tape.watch(input)
    tape.record(opNeg, @[input], output, gradFn)

    let outputGrad = newUniqueTensorRef(newShape(5), dtFloat32)
    let grads = tape.backward(output, outputGrad)

    check input.hash in grads

suite "Compute Gradients":
  test "computeGradients for parameters":
    let tape = newGradientTape()
    let w = newUniqueTensorRef(newShape(5), dtFloat32)
    let b = newUniqueTensorRef(newShape(5), dtFloat32)
    let out1 = newUniqueTensorRef(newShape(5), dtFloat32)
    let out2 = newUniqueTensorRef(newShape(5), dtFloat32)

    let passThrough = proc(grad: TensorRef, inputs: seq[TensorRef],
                           output: TensorRef): seq[TensorRef] =
      @[grad]

    tape.watch(w)
    tape.watch(b)
    tape.record(opNeg, @[w], out1, passThrough)
    tape.record(opNeg, @[b], out2, passThrough)

    # Compute gradients for specific parameters
    let grads = tape.computeGradients(out2, @[w, b])

    check grads.len == 2

suite "Chain Rule Helpers":
  test "chainAdd":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let left = newUniqueTensorRef(newShape(5), dtFloat32)
    let right = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = chainAdd(grad, left, right)
    check grads.len == 2
    check grads[0] == grad  # Left gets full gradient
    check grads[1] == grad  # Right gets full gradient

  test "chainMul":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let left = newUniqueTensorRef(newShape(5), dtFloat32)
    let right = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = chainMul(grad, left, right)
    check grads.len == 2

  test "chainNeg":
    let grad = newUniqueTensorRef(newShape(5), dtFloat32)
    let input = newUniqueTensorRef(newShape(5), dtFloat32)

    let grads = chainNeg(grad, input)
    check grads.len == 1

  test "chainMatMul":
    let grad = newUniqueTensorRef(newShape(5, 5), dtFloat32)
    let left = newUniqueTensorRef(newShape(5, 5), dtFloat32)
    let right = newUniqueTensorRef(newShape(5, 5), dtFloat32)

    let grads = chainMatMul(grad, left, right)
    check grads.len == 2

suite "Gradient Registry":
  test "register and get gradient":
    let registry = GradientRegistry(gradFns: initTable[OpKind, GradFn]())

    let myGradFn = proc(grad: TensorRef, inputs: seq[TensorRef],
                        output: TensorRef): seq[TensorRef] =
      @[grad]

    registry.registerGradient(opAbs, myGradFn)

    let gotFn = registry.getGradientFn(opAbs)
    check not gotFn.isNil

  test "global registry":
    let myGradFn = proc(grad: TensorRef, inputs: seq[TensorRef],
                        output: TensorRef): seq[TensorRef] =
      @[grad]

    registerGradient(opSqrt, myGradFn)
    check hasGradientFn(opSqrt)

  test "get nonexistent gradient":
    let registry = GradientRegistry(gradFns: initTable[OpKind, GradFn]())
    let fn = registry.getGradientFn(opConv2d)
    check fn.isNil

suite "Autograd Operation Wrapper":
  test "autogradOp records with gradient function":
    # Register a gradient function first
    let myGradFn = proc(grad: TensorRef, inputs: seq[TensorRef],
                        output: TensorRef): seq[TensorRef] =
      @[grad]
    registerGradient(opExp, myGradFn)

    let tape = newGradientTape()
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    tape.watch(input)
    tape.autogradOp(opExp, @[input], output)

    check tape.len == 1
    check tape.entries[0].gradFn != nil

  test "autogradOp skips when no input needs grad":
    let tape = newGradientTape()
    let input = newUniqueTensorRef(newShape(5), dtFloat32)
    let output = newUniqueTensorRef(newShape(5), dtFloat32)

    # Don't watch input
    tape.autogradOp(opNeg, @[input], output)

    check tape.len == 0  # Not recorded
