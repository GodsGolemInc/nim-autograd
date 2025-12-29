## Tests for gradient tape module

import unittest
import std/[options, strutils]
import ml_core
import ../src/ml_autograd/tape

# Helper to create TensorRef with unique hash
var tensorCounter {.global.} = 0

proc newUniqueTensorRef(shape: Shape, dtype: DType): TensorRef =
  ## Create a tensor ref with a unique hash
  inc tensorCounter
  var uniqueHash: Hash256  # array[32, byte]
  let counter = tensorCounter
  uniqueHash[0] = byte(counter and 0xFF)
  uniqueHash[1] = byte((counter shr 8) and 0xFF)
  uniqueHash[2] = byte((counter shr 16) and 0xFF)
  uniqueHash[3] = byte((counter shr 24) and 0xFF)
  newTensorRef(uniqueHash, shape, dtype)

suite "TapeEntry Creation":
  test "newTapeEntry":
    let t1 = newTensorRef(newShape(10), dtFloat32)
    let t2 = newTensorRef(newShape(10), dtFloat32)
    let out1 = newTensorRef(newShape(10), dtFloat32)

    let entry = newTapeEntry(opAdd, @[t1, t2], out1)
    check entry.op == opAdd
    check entry.inputs.len == 2
    check entry.output == out1
    check entry.gradFn.isNil
    check entry.savedTensors.len == 0

  test "newTapeEntry with gradFn":
    let t1 = newTensorRef(newShape(5), dtFloat32)
    let out1 = newTensorRef(newShape(5), dtFloat32)

    let gradFn = proc(grad: TensorRef, inputs: seq[TensorRef],
                      output: TensorRef): seq[TensorRef] =
      @[grad]

    let entry = newTapeEntry(opNeg, @[t1], out1, gradFn)
    check not entry.gradFn.isNil

  test "saveForBackward":
    let t1 = newTensorRef(newShape(5), dtFloat32)
    let t2 = newTensorRef(newShape(5), dtFloat32)
    let out1 = newTensorRef(newShape(5), dtFloat32)

    let entry = newTapeEntry(opAdd, @[t1, t2], out1)
    entry.saveForBackward(t1, t2)
    check entry.savedTensors.len == 2

  test "string representation":
    let t1 = newTensorRef(newShape(5), dtFloat32)
    let out1 = newTensorRef(newShape(5), dtFloat32)
    let entry = newTapeEntry(opNeg, @[t1], out1)
    let s = $entry
    check "TapeEntry" in s
    check "neg" in s

suite "GradientTape Creation":
  test "newGradientTape":
    let tape = newGradientTape()
    check tape.entries.len == 0
    check tape.len == 0
    check not tape.persistent
    check tape.enabled

  test "newGradientTape persistent":
    let tape = newGradientTape(persistent = true)
    check tape.persistent

  test "string representation":
    let tape = newGradientTape()
    let s = $tape
    check "GradientTape" in s
    check "0 entries" in s

suite "GradientTape Watch":
  test "watch tensor":
    let tape = newGradientTape()
    let t = newTensorRef(newShape(10), dtFloat32)

    check not tape.isWatching(t)
    tape.watch(t)
    check tape.isWatching(t)

  test "unwatch tensor":
    let tape = newGradientTape()
    let t = newTensorRef(newShape(10), dtFloat32)

    tape.watch(t)
    check tape.isWatching(t)
    tape.unwatch(t)
    check not tape.isWatching(t)

  test "multiple watches":
    let tape = newGradientTape()
    let t1 = newTensorRef(newShape(5), dtFloat32)
    let t2 = newTensorRef(newShape(10), dtFloat32)

    tape.watch(t1)
    tape.watch(t2)
    check tape.isWatching(t1)
    check tape.isWatching(t2)

  test "requiresGrad":
    let tape = newGradientTape()
    let t1 = newUniqueTensorRef(newShape(5), dtFloat32)
    let t2 = newUniqueTensorRef(newShape(5), dtFloat32)  # Same shape but unique hash

    tape.watch(t1)
    check tape.requiresGrad(t1)
    check not tape.requiresGrad(t2)

suite "GradientTape Recording":
  test "record entry":
    let tape = newGradientTape()
    let t1 = newTensorRef(newShape(5), dtFloat32)
    let out1 = newTensorRef(newShape(5), dtFloat32)

    let entry = newTapeEntry(opNeg, @[t1], out1)
    tape.record(entry)
    check tape.len == 1

  test "record convenience":
    let tape = newGradientTape()
    let t1 = newTensorRef(newShape(5), dtFloat32)
    let t2 = newTensorRef(newShape(5), dtFloat32)
    let out1 = newTensorRef(newShape(5), dtFloat32)

    tape.record(opAdd, @[t1, t2], out1)
    check tape.len == 1
    check tape.entries[0].op == opAdd

  test "record multiple":
    let tape = newGradientTape()
    let t1 = newTensorRef(newShape(5), dtFloat32)
    let t2 = newTensorRef(newShape(5), dtFloat32)
    let t3 = newTensorRef(newShape(5), dtFloat32)

    tape.record(opNeg, @[t1], t2)
    tape.record(opExp, @[t2], t3)
    check tape.len == 2

  test "recording disabled":
    let tape = newGradientTape()
    tape.disable()

    let t1 = newTensorRef(newShape(5), dtFloat32)
    let out1 = newTensorRef(newShape(5), dtFloat32)
    tape.record(opNeg, @[t1], out1)

    check tape.len == 0

  test "enable/disable":
    let tape = newGradientTape()
    let t1 = newTensorRef(newShape(5), dtFloat32)
    let out1 = newTensorRef(newShape(5), dtFloat32)

    tape.disable()
    check not tape.enabled
    tape.record(opNeg, @[t1], out1)
    check tape.len == 0

    tape.enable()
    check tape.enabled
    tape.record(opNeg, @[t1], out1)
    check tape.len == 1

suite "GradientTape Clear/Reset":
  test "clear entries":
    let tape = newGradientTape()
    let t1 = newTensorRef(newShape(5), dtFloat32)
    let out1 = newTensorRef(newShape(5), dtFloat32)

    tape.watch(t1)
    tape.record(opNeg, @[t1], out1)
    check tape.len == 1
    check tape.watchingCount == 1

    tape.clear()
    check tape.len == 0
    check tape.watchingCount == 1  # Watches preserved

  test "reset completely":
    let tape = newGradientTape()
    let t1 = newTensorRef(newShape(5), dtFloat32)
    let out1 = newTensorRef(newShape(5), dtFloat32)

    tape.watch(t1)
    tape.record(opNeg, @[t1], out1)

    tape.reset()
    check tape.len == 0
    check tape.watchingCount == 0

suite "GradientContext Creation":
  test "newGradientContext":
    let ctx = newGradientContext()
    check ctx.tape != nil
    check ctx.gradsCount == 0

  test "newGradientContext with tape":
    let tape = newGradientTape()
    let ctx = newGradientContext(tape)
    check ctx.tape == tape

  test "string representation":
    let ctx = newGradientContext()
    let s = $ctx
    check "GradientContext" in s

suite "GradientContext Gradient Management":
  test "setGrad and getGrad":
    let ctx = newGradientContext()
    let t = newTensorRef(newShape(5), dtFloat32)
    let grad = newTensorRef(newShape(5), dtFloat32)

    check ctx.getGrad(t).isNone
    ctx.setGrad(t, grad)
    check ctx.getGrad(t).isSome
    check ctx.getGrad(t).get == grad

  test "hasGrad":
    let ctx = newGradientContext()
    let t = newTensorRef(newShape(5), dtFloat32)
    let grad = newTensorRef(newShape(5), dtFloat32)

    check not ctx.hasGrad(t)
    ctx.setGrad(t, grad)
    check ctx.hasGrad(t)

  test "accumGrad":
    let ctx = newGradientContext()
    let t = newTensorRef(newShape(5), dtFloat32)
    let grad1 = newTensorRef(newShape(5), dtFloat32)
    let grad2 = newTensorRef(newShape(5), dtFloat32)

    ctx.accumGrad(t, grad1)
    check ctx.hasGrad(t)
    ctx.accumGrad(t, grad2)  # Would add in real implementation
    check ctx.hasGrad(t)

  test "clearGrads":
    let ctx = newGradientContext()
    let t1 = newUniqueTensorRef(newShape(5), dtFloat32)
    let t2 = newUniqueTensorRef(newShape(5), dtFloat32)  # Unique hash
    let grad1 = newUniqueTensorRef(newShape(5), dtFloat32)
    let grad2 = newUniqueTensorRef(newShape(5), dtFloat32)

    ctx.setGrad(t1, grad1)
    ctx.setGrad(t2, grad2)
    check ctx.gradsCount == 2

    ctx.clearGrads()
    check ctx.gradsCount == 0

suite "Gradient Input Tracking":
  test "needsInputGrad":
    let tape = newGradientTape()
    let t1 = newUniqueTensorRef(newShape(5), dtFloat32)
    let t2 = newUniqueTensorRef(newShape(5), dtFloat32)  # Unique hash
    let t3 = newUniqueTensorRef(newShape(5), dtFloat32)  # Unique hash

    tape.watch(t1)
    tape.watch(t3)

    let needs = tape.needsInputGrad(@[t1, t2, t3])
    check needs.len == 3
    check needs[0] == true   # t1 watched
    check needs[1] == false  # t2 not watched
    check needs[2] == true   # t3 watched

  test "anyInputNeedsGrad":
    let tape = newGradientTape()
    let t1 = newUniqueTensorRef(newShape(5), dtFloat32)
    let t2 = newUniqueTensorRef(newShape(5), dtFloat32)  # Unique hash
    let t3 = newUniqueTensorRef(newShape(5), dtFloat32)  # Unique hash

    check not tape.anyInputNeedsGrad(@[t1, t2])

    tape.watch(t3)
    check tape.anyInputNeedsGrad(@[t1, t2, t3])

suite "TapeScope":
  test "enterScope and exitScope":
    let tape = newGradientTape()
    let t = newTensorRef(newShape(5), dtFloat32)
    let out1 = newTensorRef(newShape(5), dtFloat32)

    let scope = tape.enterScope()
    tape.record(opNeg, @[t], out1)
    check tape.len == 1

    scope.exitScope()
    check tape.len == 0  # Cleared after exit

  test "persistent tape preserves entries":
    let tape = newGradientTape(persistent = true)
    let t = newTensorRef(newShape(5), dtFloat32)
    let out1 = newTensorRef(newShape(5), dtFloat32)

    let scope = tape.enterScope()
    tape.record(opNeg, @[t], out1)
    check tape.len == 1

    scope.exitScope()
    check tape.len == 1  # Preserved for persistent tape

suite "Gradient Checking Utilities":
  test "checkGradShape matching":
    let shape = newShape(10, 20)
    check checkGradShape(shape, shape)

  test "checkGradShape mismatching":
    let shape1 = newShape(10, 20)
    let shape2 = newShape(10, 30)
    check not checkGradShape(shape1, shape2)

  test "checkGradDtype float32":
    check checkGradDtype(dtFloat32, dtFloat32)
    check checkGradDtype(dtFloat32, dtFloat64)
    check not checkGradDtype(dtFloat32, dtFloat16)

  test "checkGradDtype float16":
    check checkGradDtype(dtFloat16, dtFloat16)
    check checkGradDtype(dtFloat16, dtFloat32)
    check checkGradDtype(dtFloat16, dtFloat64)

  test "checkGradDtype int (no gradient)":
    check not checkGradDtype(dtInt32, dtInt32)
    check not checkGradDtype(dtInt32, dtFloat32)
