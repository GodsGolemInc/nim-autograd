## Backward Pass Implementation
##
## Provides the backward pass algorithm for automatic differentiation.
## Traverses the computation graph in reverse to compute gradients.

import std/[tables, algorithm, hashes, options]
import ml_core
import ./tape

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

        ctx.accumGrad(inp, inputGrad)

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
    let ones = newTensorRef(target.shape, target.dtype)
    ctx.setGrad(target, ones)
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
  ## This is a placeholder - actual implementation would need
  ## tensor arithmetic which is not available yet
  nil

# Chain rule helpers

proc chainAdd*(grad: TensorRef, left, right: TensorRef): seq[TensorRef] =
  ## Gradient for addition: d/dx (a + b) = (1, 1)
  ## Both inputs get the full gradient
  @[grad, grad]

proc chainMul*(grad: TensorRef, left, right: TensorRef): seq[TensorRef] =
  ## Gradient for multiplication: d/dx (a * b) = (b, a)
  ## Placeholder - would need actual tensor multiplication
  @[right, left]

proc chainNeg*(grad: TensorRef, input: TensorRef): seq[TensorRef] =
  ## Gradient for negation: d/dx (-x) = -1
  ## Placeholder - would need negation
  @[grad]

proc chainMatMul*(grad: TensorRef, left, right: TensorRef): seq[TensorRef] =
  ## Gradient for matrix multiplication
  ## d/dA (A @ B) = grad @ B.T
  ## d/dB (A @ B) = A.T @ grad
  ## Placeholder - would need actual computation
  @[grad, grad]

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
