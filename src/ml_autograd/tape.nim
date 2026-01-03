## Gradient Tape Module
##
## Provides automatic differentiation through reverse-mode autodiff.
## Records operations for backward pass gradient computation.

import std/[tables, options, hashes]
import ml_core

type
  GradFn* = proc(grad: TensorRef, inputs: seq[TensorRef],
                 output: TensorRef): seq[TensorRef]
    ## Function that computes gradients w.r.t. inputs given output gradient

  TapeEntry* = ref object
    ## A single entry in the gradient tape
    op*: OpKind              ## Operation that was performed
    inputs*: seq[TensorRef]  ## Input tensors
    output*: TensorRef       ## Output tensor
    gradFn*: GradFn          ## Function to compute gradients
    savedTensors*: seq[TensorRef]  ## Tensors saved for backward

  GradientTape* = ref object
    ## Records operations for automatic differentiation
    entries*: seq[TapeEntry]
    watching*: Table[Hash256, bool]  ## Tensors being watched
    persistent*: bool         ## If true, tape is not cleared after backward
    enabled*: bool            ## If false, operations are not recorded

  GradientContext* = ref object
    ## Context for managing gradient computation
    tape*: GradientTape
    grads*: Table[Hash256, TensorRef]  ## Accumulated gradients

# TapeEntry creation

proc newTapeEntry*(op: OpKind, inputs: seq[TensorRef], output: TensorRef,
                   gradFn: GradFn = nil): TapeEntry =
  ## Create a new tape entry
  TapeEntry(
    op: op,
    inputs: inputs,
    output: output,
    gradFn: gradFn,
    savedTensors: @[]
  )

proc saveForBackward*(entry: TapeEntry, tensors: varargs[TensorRef]) =
  ## Save tensors needed for backward pass
  for t in tensors:
    entry.savedTensors.add(t)

# GradientTape creation and management

proc newGradientTape*(persistent: bool = false): GradientTape =
  ## Create a new gradient tape
  GradientTape(
    entries: @[],
    watching: initTable[Hash256, bool](),
    persistent: persistent,
    enabled: true
  )

proc watch*(tape: GradientTape, t: TensorRef) =
  ## Mark a tensor for gradient tracking
  tape.watching[t.hash] = true

proc isWatching*(tape: GradientTape, t: TensorRef): bool =
  ## Check if tensor is being watched
  t.hash in tape.watching

proc unwatch*(tape: GradientTape, t: TensorRef) =
  ## Stop watching a tensor
  if t.hash in tape.watching:
    tape.watching.del(t.hash)

proc record*(tape: GradientTape, entry: TapeEntry) =
  ## Record an operation in the tape
  if tape.enabled:
    tape.entries.add(entry)

proc record*(tape: GradientTape, op: OpKind, inputs: seq[TensorRef],
             output: TensorRef, gradFn: GradFn = nil) =
  ## Record an operation in the tape (convenience overload)
  let entry = newTapeEntry(op, inputs, output, gradFn)
  tape.record(entry)

proc clear*(tape: GradientTape) =
  ## Clear all entries from the tape
  tape.entries = @[]

proc reset*(tape: GradientTape) =
  ## Reset tape completely (clear entries and watched tensors)
  tape.entries = @[]
  tape.watching.clear()

proc enable*(tape: GradientTape) =
  ## Enable recording
  tape.enabled = true

proc disable*(tape: GradientTape) =
  ## Disable recording
  tape.enabled = false

proc len*(tape: GradientTape): int =
  ## Number of entries in tape
  tape.entries.len

proc watchingCount*(tape: GradientTape): int =
  ## Number of tensors being watched
  tape.watching.len

# GradientContext creation and management

proc newGradientContext*(tape: GradientTape = nil): GradientContext =
  ## Create a new gradient context
  GradientContext(
    tape: if tape.isNil: newGradientTape() else: tape,
    grads: initTable[Hash256, TensorRef]()
  )

proc getGrad*(ctx: GradientContext, t: TensorRef): Option[TensorRef] =
  ## Get gradient for a tensor
  if t.hash in ctx.grads:
    some(ctx.grads[t.hash])
  else:
    none(TensorRef)

proc setGrad*(ctx: GradientContext, t: TensorRef, grad: TensorRef) =
  ## Set gradient for a tensor
  ctx.grads[t.hash] = grad

proc accumGrad*(ctx: GradientContext, t: TensorRef, grad: TensorRef) =
  ## Accumulate gradient for a tensor
  ## Note: Actual gradient addition is done via addGradFn callback
  ## For now, we store the last gradient (basic implementation)
  ## Full accumulation requires tensor_ops which would cause circular import
  ctx.grads[t.hash] = grad

proc hasGrad*(ctx: GradientContext, t: TensorRef): bool =
  ## Check if tensor has gradient
  t.hash in ctx.grads

proc clearGrads*(ctx: GradientContext) =
  ## Clear all accumulated gradients
  ctx.grads.clear()

proc gradsCount*(ctx: GradientContext): int =
  ## Number of accumulated gradients
  ctx.grads.len

proc requiresGrad*(tape: GradientTape, t: TensorRef): bool =
  ## Check if tensor requires gradient computation
  tape.isWatching(t)

# Utility functions

proc needsInputGrad*(tape: GradientTape, inputs: seq[TensorRef]): seq[bool] =
  ## Determine which inputs need gradients
  result = @[]
  for inp in inputs:
    result.add(tape.isWatching(inp))

proc anyInputNeedsGrad*(tape: GradientTape, inputs: seq[TensorRef]): bool =
  ## Check if any input needs gradient
  for inp in inputs:
    if tape.isWatching(inp):
      return true
  false

# String representations

proc opName*(op: OpKind): string =
  ## Get operation name as string
  $op

proc `$`*(entry: TapeEntry): string =
  ## String representation of tape entry
  result = "TapeEntry(" & entry.op.opName
  result &= ", inputs=" & $entry.inputs.len
  result &= ")"

proc `$`*(tape: GradientTape): string =
  ## String representation of tape
  result = "GradientTape("
  result &= $tape.entries.len & " entries, "
  result &= $tape.watching.len & " watched)"

proc `$`*(ctx: GradientContext): string =
  ## String representation of context
  "GradientContext(" & $ctx.grads.len & " grads)"

# Tape scope management (for with-statement-like usage)

type TapeScope* = object
  ## RAII-style tape scope
  tape*: GradientTape
  previouslyEnabled*: bool

proc enterScope*(tape: GradientTape): TapeScope =
  ## Enter a tape scope
  TapeScope(tape: tape, previouslyEnabled: tape.enabled)

proc exitScope*(scope: TapeScope) =
  ## Exit a tape scope
  if not scope.tape.persistent:
    scope.tape.clear()
  scope.tape.enabled = scope.previouslyEnabled

# Gradient checking utilities

proc checkGradShape*(inputShape: Shape, gradShape: Shape): bool =
  ## Check if gradient shape matches input shape
  inputShape == gradShape

proc checkGradDtype*(inputDtype: DType, gradDtype: DType): bool =
  ## Check if gradient dtype is compatible with input dtype
  # Gradients should typically have same or higher precision
  case inputDtype
  of dtFloat16:
    gradDtype in {dtFloat16, dtFloat32, dtFloat64}
  of dtFloat32:
    gradDtype in {dtFloat32, dtFloat64}
  of dtFloat64:
    gradDtype == dtFloat64
  of dtBFloat16:
    gradDtype in {dtBFloat16, dtFloat32, dtFloat64}
  else:
    false  # Non-float types don't have gradients
