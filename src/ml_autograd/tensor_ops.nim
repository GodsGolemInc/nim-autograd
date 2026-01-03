## Tensor Operations for Gradient Computation
##
## Provides tensor arithmetic operations using TensorData.
## These operations are used during backward pass to compute gradients.

import std/[tables, math, hashes]
import ml_core

type
  TensorStore* = ref object
    ## Storage for tensor data, keyed by hash
    store*: Table[Hash256, TensorData]

var globalTensorStore* {.global.} = TensorStore(
  store: initTable[Hash256, TensorData]()
)

# Tensor counter for unique hash generation
var tensorOpCounter {.global.} = 0

proc generateHash*(): Hash256 =
  ## Generate a unique hash for computed tensors
  inc tensorOpCounter
  let counter = tensorOpCounter
  result[0] = byte(counter and 0xFF)
  result[1] = byte((counter shr 8) and 0xFF)
  result[2] = byte((counter shr 16) and 0xFF)
  result[3] = byte((counter shr 24) and 0xFF)
  result[4] = byte((counter shr 32) and 0xFF)
  result[5] = byte((counter shr 40) and 0xFF)
  result[6] = byte((counter shr 48) and 0xFF)
  result[7] = byte((counter shr 56) and 0xFF)
  # Mark as computed gradient (for debugging)
  result[31] = 0xAD  # "AD" for autodiff

proc registerTensor*(tr: TensorRef, td: TensorData) =
  ## Register tensor data in the global store
  globalTensorStore.store[tr.hash] = td

proc getTensorData*(tr: TensorRef): TensorData =
  ## Get tensor data from global store
  if tr.hash in globalTensorStore.store:
    return globalTensorStore.store[tr.hash]
  # Return zeros if not found
  result = newTensorDataZeros(tr.shape, tr.dtype)

proc hasTensorData*(tr: TensorRef): bool =
  ## Check if tensor has data in store
  tr.hash in globalTensorStore.store

# Create new TensorRef with data

proc newComputedTensor*(shape: Shape, dtype: DType): (TensorRef, TensorData) =
  ## Create a new tensor with data storage
  let td = newTensorDataZeros(shape, dtype)
  let h = generateHash()
  let tr = newTensorRef(h, shape, dtype)
  registerTensor(tr, td)
  (tr, td)

proc newComputedTensorRef*(td: TensorData): TensorRef =
  ## Create a tensor ref from existing data
  let h = generateHash()
  result = newTensorRef(h, td.shape, td.dtype)
  registerTensor(result, td)

# Element-wise operations on TensorData

proc negData*(input: TensorData): TensorData =
  ## Negate tensor data
  result = newTensorDataZeros(input.shape, input.dtype)
  let size = input.size
  case input.dtype
  of dtFloat32:
    let inArr = input.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      outArr[i] = -inArr[i]
  of dtFloat64:
    let inArr = input.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      outArr[i] = -inArr[i]
  else:
    discard

proc addData*(a, b: TensorData): TensorData =
  ## Add two tensor data
  assert a.shape == b.shape, "Shapes must match for addition"
  result = newTensorDataZeros(a.shape, a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let bArr = b.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      outArr[i] = aArr[i] + bArr[i]
  of dtFloat64:
    let aArr = a.asFloat64
    let bArr = b.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      outArr[i] = aArr[i] + bArr[i]
  else:
    discard

# Aliases for tape.nim forward declarations
proc addTensorData*(a, b: TensorData): TensorData =
  ## Alias for addData (used by tape.nim)
  addData(a, b)

proc getTensorDataFromStore*(tr: TensorRef): TensorData =
  ## Alias for getTensorData (used by tape.nim)
  getTensorData(tr)

proc registerTensorInStore*(tr: TensorRef, td: TensorData) =
  ## Alias for registerTensor (used by tape.nim)
  registerTensor(tr, td)

proc generateUniqueHash*(): Hash256 =
  ## Alias for generateHash (used by tape.nim)
  generateHash()

proc newComputedTensorRefFromData*(td: TensorData): TensorRef =
  ## Alias for newComputedTensorRef (used by tape.nim)
  newComputedTensorRef(td)

proc subData*(a, b: TensorData): TensorData =
  ## Subtract tensor data
  assert a.shape == b.shape, "Shapes must match for subtraction"
  result = newTensorDataZeros(a.shape, a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let bArr = b.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      outArr[i] = aArr[i] - bArr[i]
  of dtFloat64:
    let aArr = a.asFloat64
    let bArr = b.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      outArr[i] = aArr[i] - bArr[i]
  else:
    discard

proc mulData*(a, b: TensorData): TensorData =
  ## Element-wise multiply tensor data
  assert a.shape == b.shape, "Shapes must match for multiplication"
  result = newTensorDataZeros(a.shape, a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let bArr = b.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      outArr[i] = aArr[i] * bArr[i]
  of dtFloat64:
    let aArr = a.asFloat64
    let bArr = b.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      outArr[i] = aArr[i] * bArr[i]
  else:
    discard

proc divData*(a, b: TensorData): TensorData =
  ## Element-wise divide tensor data
  assert a.shape == b.shape, "Shapes must match for division"
  result = newTensorDataZeros(a.shape, a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let bArr = b.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      if bArr[i] != 0.0'f32:
        outArr[i] = aArr[i] / bArr[i]
  of dtFloat64:
    let aArr = a.asFloat64
    let bArr = b.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      if bArr[i] != 0.0'f64:
        outArr[i] = aArr[i] / bArr[i]
  else:
    discard

proc scaleData*(a: TensorData, scalar: float): TensorData =
  ## Scale tensor data by scalar
  result = newTensorDataZeros(a.shape, a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      outArr[i] = aArr[i] * scalar.float32
  of dtFloat64:
    let aArr = a.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      outArr[i] = aArr[i] * scalar
  else:
    discard

proc reciprocalData*(a: TensorData): TensorData =
  ## Compute 1/x for each element
  result = newTensorDataZeros(a.shape, a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      if aArr[i] != 0.0'f32:
        outArr[i] = 1.0'f32 / aArr[i]
  of dtFloat64:
    let aArr = a.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      if aArr[i] != 0.0'f64:
        outArr[i] = 1.0'f64 / aArr[i]
  else:
    discard

proc squareData*(a: TensorData): TensorData =
  ## Compute x^2 for each element
  result = newTensorDataZeros(a.shape, a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      outArr[i] = aArr[i] * aArr[i]
  of dtFloat64:
    let aArr = a.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      outArr[i] = aArr[i] * aArr[i]
  else:
    discard

proc sqrtData*(a: TensorData): TensorData =
  ## Compute sqrt(x) for each element
  result = newTensorDataZeros(a.shape, a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      if aArr[i] >= 0.0'f32:
        outArr[i] = sqrt(aArr[i])
  of dtFloat64:
    let aArr = a.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      if aArr[i] >= 0.0'f64:
        outArr[i] = sqrt(aArr[i])
  else:
    discard

proc expData*(a: TensorData): TensorData =
  ## Compute exp(x) for each element
  result = newTensorDataZeros(a.shape, a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      outArr[i] = exp(aArr[i])
  of dtFloat64:
    let aArr = a.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      outArr[i] = exp(aArr[i])
  else:
    discard

proc logData*(a: TensorData): TensorData =
  ## Compute log(x) for each element
  result = newTensorDataZeros(a.shape, a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      if aArr[i] > 0.0'f32:
        outArr[i] = ln(aArr[i])
  of dtFloat64:
    let aArr = a.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      if aArr[i] > 0.0'f64:
        outArr[i] = ln(aArr[i])
  else:
    discard

proc sinData*(a: TensorData): TensorData =
  ## Compute sin(x) for each element
  result = newTensorDataZeros(a.shape, a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      outArr[i] = sin(aArr[i])
  of dtFloat64:
    let aArr = a.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      outArr[i] = sin(aArr[i])
  else:
    discard

proc cosData*(a: TensorData): TensorData =
  ## Compute cos(x) for each element
  result = newTensorDataZeros(a.shape, a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      outArr[i] = cos(aArr[i])
  of dtFloat64:
    let aArr = a.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      outArr[i] = cos(aArr[i])
  else:
    discard

proc tanhData*(a: TensorData): TensorData =
  ## Compute tanh(x) for each element
  result = newTensorDataZeros(a.shape, a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      outArr[i] = tanh(aArr[i])
  of dtFloat64:
    let aArr = a.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      outArr[i] = tanh(aArr[i])
  else:
    discard

proc signData*(a: TensorData): TensorData =
  ## Compute sign(x) for each element
  result = newTensorDataZeros(a.shape, a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      if aArr[i] > 0.0'f32:
        outArr[i] = 1.0'f32
      elif aArr[i] < 0.0'f32:
        outArr[i] = -1.0'f32
      else:
        outArr[i] = 0.0'f32
  of dtFloat64:
    let aArr = a.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      if aArr[i] > 0.0'f64:
        outArr[i] = 1.0'f64
      elif aArr[i] < 0.0'f64:
        outArr[i] = -1.0'f64
      else:
        outArr[i] = 0.0'f64
  else:
    discard

proc powData*(base, exp: TensorData): TensorData =
  ## Compute base^exp for each element
  assert base.shape == exp.shape, "Shapes must match for power"
  result = newTensorDataZeros(base.shape, base.dtype)
  let size = base.size
  case base.dtype
  of dtFloat32:
    let baseArr = base.asFloat32
    let expArr = exp.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      outArr[i] = pow(baseArr[i], expArr[i])
  of dtFloat64:
    let baseArr = base.asFloat64
    let expArr = exp.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      outArr[i] = pow(baseArr[i], expArr[i])
  else:
    discard

# Activation functions

proc reluData*(a: TensorData): TensorData =
  ## Compute ReLU(x) = max(0, x)
  result = newTensorDataZeros(a.shape, a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      outArr[i] = max(0.0'f32, aArr[i])
  of dtFloat64:
    let aArr = a.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      outArr[i] = max(0.0'f64, aArr[i])
  else:
    discard

proc reluMaskData*(a: TensorData): TensorData =
  ## Compute ReLU mask: 1 if x > 0, else 0
  result = newTensorDataZeros(a.shape, a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      outArr[i] = if aArr[i] > 0.0'f32: 1.0'f32 else: 0.0'f32
  of dtFloat64:
    let aArr = a.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      outArr[i] = if aArr[i] > 0.0'f64: 1.0'f64 else: 0.0'f64
  else:
    discard

proc sigmoidData*(a: TensorData): TensorData =
  ## Compute sigmoid(x) = 1 / (1 + exp(-x))
  result = newTensorDataZeros(a.shape, a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      outArr[i] = 1.0'f32 / (1.0'f32 + exp(-aArr[i]))
  of dtFloat64:
    let aArr = a.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      outArr[i] = 1.0'f64 / (1.0'f64 + exp(-aArr[i]))
  else:
    discard

proc leakyReluMaskData*(a: TensorData, alpha: float = 0.01): TensorData =
  ## Compute LeakyReLU mask: 1 if x > 0, else alpha
  result = newTensorDataZeros(a.shape, a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< size:
      outArr[i] = if aArr[i] > 0.0'f32: 1.0'f32 else: alpha.float32
  of dtFloat64:
    let aArr = a.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< size:
      outArr[i] = if aArr[i] > 0.0'f64: 1.0'f64 else: alpha.float64
  else:
    discard

# Matrix operations

proc transposeData*(a: TensorData): TensorData =
  ## Transpose 2D tensor
  assert a.shape.rank == 2, "Transpose requires 2D tensor"
  let rows = a.shape.dims[0]
  let cols = a.shape.dims[1]
  result = newTensorDataZeros(newShape(cols, rows), a.dtype)
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< rows:
      for j in 0 ..< cols:
        outArr[j * rows + i] = aArr[i * cols + j]
  of dtFloat64:
    let aArr = a.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< rows:
      for j in 0 ..< cols:
        outArr[j * rows + i] = aArr[i * cols + j]
  else:
    discard

proc matmulData*(a, b: TensorData): TensorData =
  ## Matrix multiplication
  assert a.shape.rank == 2 and b.shape.rank == 2, "MatMul requires 2D tensors"
  assert a.shape.dims[1] == b.shape.dims[0], "Inner dimensions must match"
  let m = a.shape.dims[0]
  let k = a.shape.dims[1]
  let n = b.shape.dims[1]
  result = newTensorDataZeros(newShape(m, n), a.dtype)
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let bArr = b.asFloat32
    let outArr = result.asFloat32
    for i in 0 ..< m:
      for j in 0 ..< n:
        var sum = 0.0'f32
        for l in 0 ..< k:
          sum += aArr[i * k + l] * bArr[l * n + j]
        outArr[i * n + j] = sum
  of dtFloat64:
    let aArr = a.asFloat64
    let bArr = b.asFloat64
    let outArr = result.asFloat64
    for i in 0 ..< m:
      for j in 0 ..< n:
        var sum = 0.0'f64
        for l in 0 ..< k:
          sum += aArr[i * k + l] * bArr[l * n + j]
        outArr[i * n + j] = sum
  else:
    discard

# Reduction operations

proc sumData*(a: TensorData): TensorData =
  ## Sum all elements
  result = newTensorDataZeros(newShape(1), a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let outArr = result.asFloat32
    var sum = 0.0'f32
    for i in 0 ..< size:
      sum += aArr[i]
    outArr[0] = sum
  of dtFloat64:
    let aArr = a.asFloat64
    let outArr = result.asFloat64
    var sum = 0.0'f64
    for i in 0 ..< size:
      sum += aArr[i]
    outArr[0] = sum
  else:
    discard

proc meanData*(a: TensorData): TensorData =
  ## Mean of all elements
  result = newTensorDataZeros(newShape(1), a.dtype)
  let size = a.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let outArr = result.asFloat32
    var sum = 0.0'f32
    for i in 0 ..< size:
      sum += aArr[i]
    outArr[0] = sum / size.float32
  of dtFloat64:
    let aArr = a.asFloat64
    let outArr = result.asFloat64
    var sum = 0.0'f64
    for i in 0 ..< size:
      sum += aArr[i]
    outArr[0] = sum / size.float64
  else:
    discard

proc broadcastData*(a: TensorData, targetShape: Shape): TensorData =
  ## Broadcast scalar or 1D to target shape
  result = newTensorDataZeros(targetShape, a.dtype)
  let targetSize = targetShape.size
  case a.dtype
  of dtFloat32:
    let aArr = a.asFloat32
    let outArr = result.asFloat32
    if a.size == 1:
      # Scalar broadcast
      for i in 0 ..< targetSize:
        outArr[i] = aArr[0]
    else:
      # General broadcast (copy values)
      for i in 0 ..< targetSize:
        outArr[i] = aArr[i mod a.size]
  of dtFloat64:
    let aArr = a.asFloat64
    let outArr = result.asFloat64
    if a.size == 1:
      for i in 0 ..< targetSize:
        outArr[i] = aArr[0]
    else:
      for i in 0 ..< targetSize:
        outArr[i] = aArr[i mod a.size]
  else:
    discard

proc onesData*(shape: Shape, dtype: DType): TensorData =
  ## Create tensor filled with ones
  result = newTensorDataZeros(shape, dtype)
  case dtype
  of dtFloat32:
    result.fillFloat32(1.0'f32)
  of dtFloat64:
    result.fillFloat64(1.0'f64)
  else:
    discard

proc zerosData*(shape: Shape, dtype: DType): TensorData =
  ## Create tensor filled with zeros (already zero by default)
  newTensorDataZeros(shape, dtype)

# TensorRef operations (create new TensorRef with computed data)

proc neg*(a: TensorRef): TensorRef =
  ## Negate tensor
  if a.isNil:
    return nil
  let aData = getTensorData(a)
  let resultData = negData(aData)
  newComputedTensorRef(resultData)

proc add*(a, b: TensorRef): TensorRef =
  ## Add tensors
  if a.isNil or b.isNil:
    return nil
  let aData = getTensorData(a)
  let bData = getTensorData(b)
  let resultData = addData(aData, bData)
  newComputedTensorRef(resultData)

proc sub*(a, b: TensorRef): TensorRef =
  ## Subtract tensors
  if a.isNil or b.isNil:
    return nil
  let aData = getTensorData(a)
  let bData = getTensorData(b)
  let resultData = subData(aData, bData)
  newComputedTensorRef(resultData)

proc mul*(a, b: TensorRef): TensorRef =
  ## Multiply tensors element-wise
  if a.isNil or b.isNil:
    return nil
  let aData = getTensorData(a)
  let bData = getTensorData(b)
  let resultData = mulData(aData, bData)
  newComputedTensorRef(resultData)

proc `div`*(a, b: TensorRef): TensorRef =
  ## Divide tensors element-wise
  if a.isNil or b.isNil:
    return nil
  let aData = getTensorData(a)
  let bData = getTensorData(b)
  let resultData = divData(aData, bData)
  newComputedTensorRef(resultData)

proc scale*(a: TensorRef, scalar: float): TensorRef =
  ## Scale tensor by scalar
  if a.isNil:
    return nil
  let aData = getTensorData(a)
  let resultData = scaleData(aData, scalar)
  newComputedTensorRef(resultData)

proc sign*(a: TensorRef): TensorRef =
  ## Sign of tensor elements
  if a.isNil:
    return nil
  let aData = getTensorData(a)
  let resultData = signData(aData)
  newComputedTensorRef(resultData)

proc square*(a: TensorRef): TensorRef =
  ## Square tensor elements
  if a.isNil:
    return nil
  let aData = getTensorData(a)
  let resultData = squareData(aData)
  newComputedTensorRef(resultData)

proc reciprocal*(a: TensorRef): TensorRef =
  ## Reciprocal of tensor elements
  if a.isNil:
    return nil
  let aData = getTensorData(a)
  let resultData = reciprocalData(aData)
  newComputedTensorRef(resultData)

proc sqrtRef*(a: TensorRef): TensorRef =
  ## Square root of tensor elements
  if a.isNil:
    return nil
  let aData = getTensorData(a)
  let resultData = sqrtData(aData)
  newComputedTensorRef(resultData)

proc expRef*(a: TensorRef): TensorRef =
  ## Exp of tensor elements
  if a.isNil:
    return nil
  let aData = getTensorData(a)
  let resultData = expData(aData)
  newComputedTensorRef(resultData)

proc logRef*(a: TensorRef): TensorRef =
  ## Log of tensor elements
  if a.isNil:
    return nil
  let aData = getTensorData(a)
  let resultData = logData(aData)
  newComputedTensorRef(resultData)

proc sinRef*(a: TensorRef): TensorRef =
  ## Sin of tensor elements
  if a.isNil:
    return nil
  let aData = getTensorData(a)
  let resultData = sinData(aData)
  newComputedTensorRef(resultData)

proc cosRef*(a: TensorRef): TensorRef =
  ## Cos of tensor elements
  if a.isNil:
    return nil
  let aData = getTensorData(a)
  let resultData = cosData(aData)
  newComputedTensorRef(resultData)

proc tanhRef*(a: TensorRef): TensorRef =
  ## Tanh of tensor elements
  if a.isNil:
    return nil
  let aData = getTensorData(a)
  let resultData = tanhData(aData)
  newComputedTensorRef(resultData)

proc transpose*(a: TensorRef): TensorRef =
  ## Transpose 2D tensor
  if a.isNil:
    return nil
  let aData = getTensorData(a)
  let resultData = transposeData(aData)
  newComputedTensorRef(resultData)

proc matmul*(a, b: TensorRef): TensorRef =
  ## Matrix multiplication
  if a.isNil or b.isNil:
    return nil
  let aData = getTensorData(a)
  let bData = getTensorData(b)
  let resultData = matmulData(aData, bData)
  newComputedTensorRef(resultData)

proc reluMask*(a: TensorRef): TensorRef =
  ## ReLU mask (1 where x > 0, 0 otherwise)
  if a.isNil:
    return nil
  let aData = getTensorData(a)
  let resultData = reluMaskData(aData)
  newComputedTensorRef(resultData)

proc sigmoidRef*(a: TensorRef): TensorRef =
  ## Sigmoid function
  if a.isNil:
    return nil
  let aData = getTensorData(a)
  let resultData = sigmoidData(aData)
  newComputedTensorRef(resultData)

proc leakyReluMask*(a: TensorRef, alpha: float = 0.01): TensorRef =
  ## LeakyReLU mask
  if a.isNil:
    return nil
  let aData = getTensorData(a)
  let resultData = leakyReluMaskData(aData, alpha)
  newComputedTensorRef(resultData)

proc broadcast*(a: TensorRef, targetShape: Shape): TensorRef =
  ## Broadcast tensor to target shape
  if a.isNil:
    return nil
  let aData = getTensorData(a)
  let resultData = broadcastData(aData, targetShape)
  newComputedTensorRef(resultData)

proc ones*(shape: Shape, dtype: DType): TensorRef =
  ## Create tensor filled with ones
  let resultData = onesData(shape, dtype)
  newComputedTensorRef(resultData)

proc zeros*(shape: Shape, dtype: DType): TensorRef =
  ## Create tensor filled with zeros
  let resultData = zerosData(shape, dtype)
  newComputedTensorRef(resultData)
