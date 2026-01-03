## Backward Pass Implementation
##
## Provides the backward pass algorithm for automatic differentiation.
## Traverses the computation graph in reverse to compute gradients.

import std/[tables, hashes, options]
import ml_core
import ./tape
import ./tensor_ops

type
  BackwardError* = object of CatchableError
    ## Error during backward pass

  BackwardOptions* = object
    ## Options for backward pass
    retainGraph*: bool       ## Keep tape entries after backward
    createGraph*: bool       ## Allow gradients of gradients
    checkShapes*: bool       ## Verify gradient shapes match

proc newBackwardOptions*(retainGraph = false, createGraph = false,
                         checkShapes = true): BackwardOptions =
  BackwardOptions(
    retainGraph: retainGraph,
    createGraph: createGraph,
    checkShapes: checkShapes
  )

# Backward pass implementation

proc backward*(ctx: GradientContext, outputGrad: TensorRef,
               options: BackwardOptions = newBackwardOptions()) =
  ## Perform backward pass from output to all watched tensors
  ## outputGrad: Gradient of the loss w.r.t. the output
  ##
  ## This traverses the tape in reverse order and calls
  ## gradient functions to propagate gradients backward.

  let tape = ctx.tape
  if tape.len == 0:
    return  # Nothing to do

  # Traverse tape in reverse order (last operation first)
  for i in countdown(tape.len - 1, 0):
    let entry = tape.entries[i]

    # Get gradient for output of this operation
    let outputGradOpt = ctx.getGrad(entry.output)
    if outputGradOpt.isNone:
      # This operation's output wasn't needed
      continue

    let outGrad = outputGradOpt.get

    # Check if we have a gradient function
    if entry.gradFn.isNil:
      # No gradient function - skip
      continue

    # Compute gradients for inputs
    let inputGrads = entry.gradFn(outGrad, entry.inputs, entry.output)

    # Accumulate gradients for inputs
    for j, inputGrad in inputGrads:
      if j < entry.inputs.len and not inputGrad.isNil:
        let inp = entry.inputs[j]

        # Check shape if requested
        if options.checkShapes:
          if inputGrad.shape != inp.shape:
            raise newException(BackwardError,
              "Gradient shape mismatch: expected " & $inp.shape &
              " got " & $inputGrad.shape)

        # Accumulate gradient (add to existing if present)
        let existingOpt = ctx.getGrad(inp)
        if existingOpt.isSome:
          let existing = existingOpt.get
          let sumGrad = add(existing, inputGrad)
          ctx.setGrad(inp, sumGrad)
        else:
          ctx.setGrad(inp, inputGrad)

  # Clear tape if not retaining
  if not options.retainGraph:
    tape.clear()

proc backward*(tape: GradientTape, target: TensorRef,
               outputGrad: TensorRef = nil,
               options: BackwardOptions = newBackwardOptions()): Table[Hash256, TensorRef] =
  ## Perform backward pass and return gradient table
  ## target: The tensor to differentiate (usually loss)
  ## outputGrad: Gradient of loss (defaults to ones if nil)
  ## Returns: Table mapping tensor hashes to their gradients

  let ctx = newGradientContext(tape)

  # Set initial gradient (usually 1 for scalar loss)
  if outputGrad.isNil:
    # Create a ones tensor with same shape as target
    let onesRef = ones(target.shape, target.dtype)
    ctx.setGrad(target, onesRef)
  else:
    ctx.setGrad(target, outputGrad)

  # Run backward pass
  ctx.backward(outputGrad, options)

  # Return gradients
  result = ctx.grads

# Gradient computation utilities

proc computeGradients*(tape: GradientTape, loss: TensorRef,
                       parameters: seq[TensorRef]): seq[TensorRef] =
  ## Compute gradients for specific parameters
  ## loss: The loss tensor to differentiate
  ## parameters: Tensors to compute gradients for
  ## Returns: Gradients in same order as parameters

  let grads = tape.backward(loss)

  result = @[]
  for param in parameters:
    if param.hash in grads:
      result.add(grads[param.hash])
    else:
      # No gradient for this parameter
      result.add(nil)

proc gradientCheckNumerical*(fn: proc(x: TensorRef): TensorRef,
                             x: TensorRef, epsilon: float = 1e-5): TensorRef =
  ## Numerical gradient checking (finite differences)
  ## fn: Function to differentiate
  ## x: Point at which to compute gradient
  ## epsilon: Step size for finite differences
  ## Returns: Numerically estimated gradient
  ##
  ## Uses central difference: (f(x+eps) - f(x-eps)) / (2*eps)

  let xData = getTensorData(x)
  let (resultRef, resultData) = newComputedTensor(x.shape, x.dtype)

  case x.dtype
  of dtFloat32:
    let xArr = xData.asFloat32
    let outArr = resultData.asFloat32
    for i in 0 ..< x.size:
      # Save original value
      let origVal = xArr[i]

      # f(x + epsilon)
      xArr[i] = origVal + epsilon.float32
      let fPlus = fn(x)
      let fPlusData = getTensorData(fPlus)
      var fPlusSum: float32 = 0.0
      let fPlusArr = fPlusData.asFloat32
      for j in 0 ..< fPlusData.size:
        fPlusSum += fPlusArr[j]

      # f(x - epsilon)
      xArr[i] = origVal - epsilon.float32
      let fMinus = fn(x)
      let fMinusData = getTensorData(fMinus)
      var fMinusSum: float32 = 0.0
      let fMinusArr = fMinusData.asFloat32
      for j in 0 ..< fMinusData.size:
        fMinusSum += fMinusArr[j]

      # Central difference
      outArr[i] = (fPlusSum - fMinusSum) / (2.0'f32 * epsilon.float32)

      # Restore original value
      xArr[i] = origVal

  of dtFloat64:
    let xArr = xData.asFloat64
    let outArr = resultData.asFloat64
    for i in 0 ..< x.size:
      let origVal = xArr[i]

      xArr[i] = origVal + epsilon
      let fPlus = fn(x)
      let fPlusData = getTensorData(fPlus)
      var fPlusSum: float64 = 0.0
      let fPlusArr = fPlusData.asFloat64
      for j in 0 ..< fPlusData.size:
        fPlusSum += fPlusArr[j]

      xArr[i] = origVal - epsilon
      let fMinus = fn(x)
      let fMinusData = getTensorData(fMinus)
      var fMinusSum: float64 = 0.0
      let fMinusArr = fMinusData.asFloat64
      for j in 0 ..< fMinusData.size:
        fMinusSum += fMinusArr[j]

      outArr[i] = (fPlusSum - fMinusSum) / (2.0 * epsilon)
      xArr[i] = origVal
  else:
    discard

  resultRef

# Chain rule helpers

proc chainAdd*(grad: TensorRef, left, right: TensorRef): seq[TensorRef] =
  ## Gradient for addition: d/dx (a + b) = (1, 1)
  ## Both inputs get the full gradient
  @[grad, grad]

proc chainMul*(grad: TensorRef, left, right: TensorRef): seq[TensorRef] =
  ## Gradient for multiplication: d/dx (a * b) = (b, a)
  ## grad_left = grad * right, grad_right = grad * left
  @[mul(grad, right), mul(grad, left)]

proc chainNeg*(grad: TensorRef, input: TensorRef): seq[TensorRef] =
  ## Gradient for negation: d/dx (-x) = -1
  ## grad_input = -grad
  @[neg(grad)]

proc chainMatMul*(grad: TensorRef, left, right: TensorRef): seq[TensorRef] =
  ## Gradient for matrix multiplication
  ## d/dA (A @ B) = grad @ B.T
  ## d/dB (A @ B) = A.T @ grad
  @[matmul(grad, transpose(right)), matmul(transpose(left), grad)]

# Gradient function registry

type GradientRegistry* = ref object
  ## Registry of gradient functions by operation
  gradFns*: Table[OpKind, GradFn]

var globalGradientRegistry* {.global.} = GradientRegistry(
  gradFns: initTable[OpKind, GradFn]()
)

proc registerGradient*(registry: GradientRegistry, op: OpKind, gradFn: GradFn) =
  ## Register a gradient function for an operation
  registry.gradFns[op] = gradFn

proc registerGradient*(op: OpKind, gradFn: GradFn) =
  ## Register a gradient function in the global registry
  globalGradientRegistry.gradFns[op] = gradFn

proc getGradientFn*(registry: GradientRegistry, op: OpKind): GradFn =
  ## Get gradient function for an operation
  if op in registry.gradFns:
    registry.gradFns[op]
  else:
    nil

proc getGradientFn*(op: OpKind): GradFn =
  ## Get gradient function from global registry
  globalGradientRegistry.getGradientFn(op)

proc hasGradientFn*(op: OpKind): bool =
  ## Check if operation has a registered gradient function
  op in globalGradientRegistry.gradFns

# Autograd-aware operation wrapper

proc autogradOp*(tape: GradientTape, op: OpKind,
                 inputs: seq[TensorRef], output: TensorRef) =
  ## Wrapper to record operation with appropriate gradient function
  if tape.anyInputNeedsGrad(inputs):
    let gradFn = getGradientFn(op)
    tape.record(op, inputs, output, gradFn)
