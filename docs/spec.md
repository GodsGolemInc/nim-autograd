# nim-autograd Specification

## Overview

自動微分（Automatic Differentiation）を提供する。
計算グラフを記録し、逆伝播で勾配を計算する。

---

## Module Structure

```
nim-autograd/
├── src/
│   ├── nimml_autograd.nim       # エントリポイント
│   └── nimml_autograd/
│       ├── tape.nim             # 計算グラフ記録
│       ├── backward.nim         # 逆伝播
│       └── grad.nim             # 勾配管理
└── nimml_autograd.nimble
```

---

## 1. Tape Module (`nimml_autograd/tape.nim`)

### Purpose

Forward計算を記録し、Backward用の計算グラフを構築する。

### Types

```nim
type
  TapeId* = distinct int

  TapeEntry* = object
    id*: TapeId
    op*: OpKind
    inputs*: seq[TensorRef]
    output*: TensorRef
    savedTensors*: seq[TensorRef]  # Tensors needed for backward
    gradFn*: GradFn

  Tape* = ref object
    entries*: seq[TapeEntry]
    tensorToEntry*: Table[Hash256, TapeId]
    isRecording*: bool
    retainGraph*: bool

  GradFn* = proc(gradOutput: TensorData,
                 savedTensors: seq[TensorData]): seq[TensorData]
    ## Computes gradients w.r.t. inputs given gradient w.r.t. output.

  RecordingContext* = object
    tape*: Tape
    prevTape*: Option[Tape]
```

### API

```nim
# Tape management
proc newTape*(): Tape

proc startRecording*(t: Tape): void
proc stopRecording*(t: Tape): void
proc isRecording*(t: Tape): bool
proc clear*(t: Tape): void

template withTape*(t: Tape, body: untyped): untyped
  ## Record operations within body.

template noGrad*(body: untyped): untyped
  ## Execute without recording gradients.

# Recording
proc record*(t: Tape, op: OpKind, inputs: seq[TensorRef],
             output: TensorRef, gradFn: GradFn,
             savedTensors: seq[TensorRef] = @[]): TapeId
  ## Record an operation.

proc getEntry*(t: Tape, id: TapeId): TapeEntry
proc getEntryForTensor*(t: Tape, ref: TensorRef): Option[TapeEntry]

# Graph queries
proc requiresGrad*(t: Tape, ref: TensorRef): bool
  ## Check if tensor requires gradient.

proc getCreator*(t: Tape, ref: TensorRef): Option[TapeEntry]
  ## Get the operation that created this tensor.

proc getDependencies*(t: Tape, ref: TensorRef): seq[TensorRef]
  ## Get tensors this tensor depends on.

# Serialization
proc toGraph*(t: Tape): Graph
  ## Convert tape to IR graph.
```

### Thread-Local Tape

```nim
var currentTape* {.threadvar.}: Option[Tape]

proc getDefaultTape*(): Tape
  ## Get or create thread-local default tape.

proc setDefaultTape*(t: Tape): void
proc clearDefaultTape*(): void
```

---

## 2. Backward Module (`nimml_autograd/backward.nim`)

### Purpose

逆伝播アルゴリズムを実装。勾配を効率的に計算。

### Types

```nim
type
  BackwardContext* = object
    tape*: Tape
    gradTensors*: Table[Hash256, TensorData]  # Accumulated gradients
    executor*: Executor
    retainGraph*: bool
    createGraph*: bool  # For higher-order derivatives

  BackwardOptions* = object
    retainGraph*: bool
    createGraph*: bool
    inputs*: seq[TensorRef]  # Specific inputs to compute grads for
    gradOutputs*: Option[seq[TensorData]]  # Custom output gradients
```

### API

```nim
proc backward*(output: TensorRef, tape: Tape,
               executor: Executor,
               options: BackwardOptions = BackwardOptions()): Table[Hash256, TensorData]
  ## Compute gradients of output w.r.t. all inputs.

proc backward*(outputs: seq[TensorRef], tape: Tape,
               executor: Executor,
               gradOutputs: seq[TensorData]): Table[Hash256, TensorData]
  ## Compute gradients for multiple outputs with custom grad_outputs.

proc grad*(output: TensorRef, inputs: seq[TensorRef],
           tape: Tape, executor: Executor,
           gradOutputs: Option[TensorData] = none(TensorData),
           retainGraph: bool = false,
           createGraph: bool = false): seq[TensorData]
  ## Compute gradients of output w.r.t. specific inputs.

# Gradient accumulation
proc accumulateGrad*(ctx: var BackwardContext,
                     ref: TensorRef, grad: TensorData): void
  ## Add gradient to accumulated value.

proc getGrad*(ctx: BackwardContext, ref: TensorRef): Option[TensorData]
  ## Get accumulated gradient.

# Backward order
proc topologicalOrder*(tape: Tape, outputs: seq[TensorRef]): seq[TapeId]
  ## Get backward execution order.
```

### Gradient Functions

```nim
# Built-in gradient functions for each op
proc gradAdd*(gradOutput: TensorData,
              savedTensors: seq[TensorData]): seq[TensorData]
  ## d(a+b)/da = 1, d(a+b)/db = 1

proc gradMul*(gradOutput: TensorData,
              savedTensors: seq[TensorData]): seq[TensorData]
  ## d(a*b)/da = b, d(a*b)/db = a

proc gradMatMul*(gradOutput: TensorData,
                 savedTensors: seq[TensorData]): seq[TensorData]
  ## d(A@B)/dA = grad @ B^T, d(A@B)/dB = A^T @ grad

proc gradRelu*(gradOutput: TensorData,
               savedTensors: seq[TensorData]): seq[TensorData]
  ## d(relu(x))/dx = 1 if x > 0 else 0

proc gradSoftmax*(gradOutput: TensorData,
                  savedTensors: seq[TensorData]): seq[TensorData]

proc gradCrossEntropy*(gradOutput: TensorData,
                       savedTensors: seq[TensorData]): seq[TensorData]

# Registry
proc registerGradFn*(op: OpKind, gradFn: GradFn): void
proc getGradFn*(op: OpKind): Option[GradFn]
proc hasGradFn*(op: OpKind): bool
```

---

## 3. Grad Module (`nimml_autograd/grad.nim`)

### Purpose

勾配テンソルの管理とユーティリティ。

### Types

```nim
type
  GradMode* = enum
    gmEnabled     # Record gradients
    gmDisabled    # No gradient recording
    gmInference   # Inference mode (no grad, optimized)

  GradientInfo* = object
    tensorRef*: TensorRef
    gradRef*: Option[TensorRef]
    requiresGrad*: bool
    isLeaf*: bool           # Created by user, not by op
    retainGrad*: bool       # Keep grad after backward

  GradManager* = ref object
    gradients*: Table[Hash256, TensorData]
    gradInfo*: Table[Hash256, GradientInfo]
    mode*: GradMode
```

### API

```nim
proc newGradManager*(): GradManager

# Gradient mode
proc setMode*(gm: GradManager, mode: GradMode): void
proc getMode*(gm: GradManager): GradMode

template enableGrad*(gm: GradManager, body: untyped): untyped
template disableGrad*(gm: GradManager, body: untyped): untyped
template inferenceMode*(gm: GradManager, body: untyped): untyped

# Gradient management
proc setRequiresGrad*(gm: GradManager, ref: TensorRef, requires: bool): void
proc requiresGrad*(gm: GradManager, ref: TensorRef): bool

proc retainGrad*(gm: GradManager, ref: TensorRef): void
  ## Keep gradient for non-leaf tensor.

proc getGrad*(gm: GradManager, ref: TensorRef): Option[TensorData]
proc setGrad*(gm: GradManager, ref: TensorRef, grad: TensorData): void
proc clearGrad*(gm: GradManager, ref: TensorRef): void
proc clearAllGrads*(gm: GradManager): void

proc zeroGrad*(gm: GradManager, refs: seq[TensorRef]): void
  ## Zero out gradients for given tensors.

# Gradient utilities
proc clipGradNorm*(gm: GradManager, refs: seq[TensorRef],
                   maxNorm: float, normType: float = 2.0): float
  ## Clip gradients by total norm. Returns actual norm.

proc clipGradValue*(gm: GradManager, refs: seq[TensorRef],
                    clipValue: float): void
  ## Clip gradients by value.

# Gradient checking
proc checkGrad*(fn: proc(inputs: seq[TensorData]): TensorData,
                inputs: seq[TensorData],
                eps: float = 1e-5,
                atol: float = 1e-4,
                rtol: float = 1e-3): bool
  ## Numerical gradient check.
```

---

## Dependencies

```nim
# nimml_autograd.nimble
requires "nim >= 2.0.0"
requires "nimml_core >= 0.1.0"
requires "nimml_executor >= 0.1.0"
```

---

## Error Handling

```nim
type
  AutogradError* = object of CatchableError

  NoGradError* = object of AutogradError
  TapeNotRecordingError* = object of AutogradError
  NoGradFnError* = object of AutogradError
  ShapeMismatchError* = object of AutogradError
  GradCheckFailedError* = object of AutogradError
```

---

## Usage Example

```nim
import nimml_autograd
import nimml_core
import nimml_executor

let executor = newCpuExecutor()
let tape = newTape()
let gm = newGradManager()

# Create leaf tensors with requires_grad
let x = newTensorRef(newShape(2, 3), dtFloat32)
let w = newTensorRef(newShape(3, 4), dtFloat32)
gm.setRequiresGrad(x, true)
gm.setRequiresGrad(w, true)

# Forward with recording
tape.withTape:
  # y = x @ w
  let yOp = newOpSpec(opMatMul, @[x, w])
  let y = yOp.output
  tape.record(opMatMul, @[x, w], y, gradMatMul, @[x, w])

  # z = relu(y)
  let zOp = newOpSpec(opRelu, @[y])
  let z = zOp.output
  tape.record(opRelu, @[y], z, gradRelu, @[y])

  # loss = sum(z)
  let lossOp = newOpSpec(opSum, @[z])
  let loss = lossOp.output
  tape.record(opSum, @[z], loss, gradSum, @[z])

# Backward
let grads = backward(loss, tape, executor)

# Get gradients
let dw = grads[w.hash]
let dx = grads[x.hash]

# Gradient clipping
discard gm.clipGradNorm(@[x, w], maxNorm = 1.0)
```
