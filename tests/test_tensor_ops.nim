## Tests for tensor_ops module
## Covers all public functions with normal and error cases

import unittest
import std/[math]
import ml_core
import ../src/ml_autograd/tensor_ops

# Helper to create TensorData with specific values
proc newTestTensorData(shape: Shape, dtype: DType, values: seq[float32]): TensorData =
  result = newTensorDataZeros(shape, dtype)
  let arr = result.asFloat32
  for i, v in values:
    if i < result.size:
      arr[i] = v

proc newTestTensorDataF64(shape: Shape, values: seq[float64]): TensorData =
  result = newTensorDataZeros(shape, dtFloat64)
  let arr = result.asFloat64
  for i, v in values:
    if i < result.size:
      arr[i] = v

# Helper to create TensorRef with data
proc newTestTensorRef(shape: Shape, dtype: DType, values: seq[float32]): TensorRef =
  let td = newTestTensorData(shape, dtype, values)
  newComputedTensorRef(td)

proc getValues(tr: TensorRef): seq[float32] =
  let td = getTensorData(tr)
  result = @[]
  let arr = td.asFloat32
  for i in 0 ..< td.size:
    result.add(arr[i])

proc getValuesF64(tr: TensorRef): seq[float64] =
  let td = getTensorData(tr)
  result = @[]
  let arr = td.asFloat64
  for i in 0 ..< td.size:
    result.add(arr[i])

suite "TensorStore and Hash Generation":
  test "generateHash returns unique hashes":
    let h1 = generateHash()
    let h2 = generateHash()
    let h3 = generateHash()
    check h1 != h2
    check h2 != h3
    check h1 != h3

  test "registerTensor and getTensorData":
    let shape = newShape(3)
    let td = newTestTensorData(shape, dtFloat32, @[1.0'f32, 2.0, 3.0])
    let h = generateHash()
    let tr = newTensorRef(h, shape, dtFloat32)
    registerTensor(tr, td)

    let retrieved = getTensorData(tr)
    check retrieved.size == 3
    let arr = retrieved.asFloat32
    check arr[0] == 1.0'f32
    check arr[1] == 2.0'f32
    check arr[2] == 3.0'f32

  test "hasTensorData":
    let shape = newShape(2)
    let td = newTestTensorData(shape, dtFloat32, @[1.0'f32, 2.0])
    let h = generateHash()
    let tr = newTensorRef(h, shape, dtFloat32)

    check not hasTensorData(tr)
    registerTensor(tr, td)
    check hasTensorData(tr)

  test "getTensorData returns zeros for unregistered":
    let h = generateHash()
    let tr = newTensorRef(h, newShape(3), dtFloat32)
    let td = getTensorData(tr)
    check td.size == 3
    let arr = td.asFloat32
    check arr[0] == 0.0'f32
    check arr[1] == 0.0'f32

  test "newComputedTensor":
    let (tr, td) = newComputedTensor(newShape(4), dtFloat32)
    check tr.shape == newShape(4)
    check tr.dtype == dtFloat32
    check td.size == 4

  test "newComputedTensorRef":
    let td = newTestTensorData(newShape(2, 3), dtFloat32, @[1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0])
    let tr = newComputedTensorRef(td)
    check tr.shape == newShape(2, 3)
    check hasTensorData(tr)

suite "TensorData Element-wise Unary Operations":
  test "negData float32":
    let td = newTestTensorData(newShape(3), dtFloat32, @[1.0'f32, -2.0, 3.0])
    let result = negData(td)
    let arr = result.asFloat32
    check arr[0] == -1.0'f32
    check arr[1] == 2.0'f32
    check arr[2] == -3.0'f32

  test "negData float64":
    let td = newTestTensorDataF64(newShape(3), @[1.0'f64, -2.0, 3.0])
    let result = negData(td)
    let arr = result.asFloat64
    check arr[0] == -1.0'f64
    check arr[1] == 2.0'f64
    check arr[2] == -3.0'f64

  test "signData":
    let td = newTestTensorData(newShape(5), dtFloat32, @[-3.0'f32, -0.1, 0.0, 0.5, 2.0])
    let result = signData(td)
    let arr = result.asFloat32
    check arr[0] == -1.0'f32
    check arr[1] == -1.0'f32
    check arr[2] == 0.0'f32
    check arr[3] == 1.0'f32
    check arr[4] == 1.0'f32

  test "squareData":
    let td = newTestTensorData(newShape(3), dtFloat32, @[2.0'f32, -3.0, 0.5])
    let result = squareData(td)
    let arr = result.asFloat32
    check abs(arr[0] - 4.0'f32) < 0.001
    check abs(arr[1] - 9.0'f32) < 0.001
    check abs(arr[2] - 0.25'f32) < 0.001

  test "sqrtData":
    let td = newTestTensorData(newShape(3), dtFloat32, @[4.0'f32, 9.0, 16.0])
    let result = sqrtData(td)
    let arr = result.asFloat32
    check abs(arr[0] - 2.0'f32) < 0.001
    check abs(arr[1] - 3.0'f32) < 0.001
    check abs(arr[2] - 4.0'f32) < 0.001

  test "sqrtData negative values returns zero":
    let td = newTestTensorData(newShape(2), dtFloat32, @[-4.0'f32, 4.0])
    let result = sqrtData(td)
    let arr = result.asFloat32
    check arr[0] == 0.0'f32  # Negative input returns 0
    check abs(arr[1] - 2.0'f32) < 0.001

  test "expData":
    let td = newTestTensorData(newShape(3), dtFloat32, @[0.0'f32, 1.0, 2.0])
    let result = expData(td)
    let arr = result.asFloat32
    check abs(arr[0] - 1.0'f32) < 0.001
    check abs(arr[1] - E.float32) < 0.001
    check abs(arr[2] - (E * E).float32) < 0.01

  test "logData":
    let td = newTestTensorData(newShape(3), dtFloat32, @[1.0'f32, E.float32, (E*E).float32])
    let result = logData(td)
    let arr = result.asFloat32
    check abs(arr[0] - 0.0'f32) < 0.001
    check abs(arr[1] - 1.0'f32) < 0.001
    check abs(arr[2] - 2.0'f32) < 0.01

  test "logData zero and negative":
    let td = newTestTensorData(newShape(2), dtFloat32, @[0.0'f32, -1.0])
    let result = logData(td)
    let arr = result.asFloat32
    check arr[0] == 0.0'f32  # log(0) returns 0 (handled)
    check arr[1] == 0.0'f32  # log(-1) returns 0 (handled)

  test "sinData":
    let td = newTestTensorData(newShape(3), dtFloat32, @[0.0'f32, PI.float32/2, PI.float32])
    let result = sinData(td)
    let arr = result.asFloat32
    check abs(arr[0] - 0.0'f32) < 0.001
    check abs(arr[1] - 1.0'f32) < 0.001
    check abs(arr[2] - 0.0'f32) < 0.001

  test "cosData":
    let td = newTestTensorData(newShape(3), dtFloat32, @[0.0'f32, PI.float32/2, PI.float32])
    let result = cosData(td)
    let arr = result.asFloat32
    check abs(arr[0] - 1.0'f32) < 0.001
    check abs(arr[1] - 0.0'f32) < 0.001
    check abs(arr[2] - (-1.0'f32)) < 0.001

  test "tanhData":
    let td = newTestTensorData(newShape(3), dtFloat32, @[0.0'f32, 1.0, -1.0])
    let result = tanhData(td)
    let arr = result.asFloat32
    check abs(arr[0] - 0.0'f32) < 0.001
    check abs(arr[1] - tanh(1.0'f32)) < 0.001
    check abs(arr[2] - tanh(-1.0'f32)) < 0.001

  test "reciprocalData":
    let td = newTestTensorData(newShape(3), dtFloat32, @[1.0'f32, 2.0, 4.0])
    let result = reciprocalData(td)
    let arr = result.asFloat32
    check abs(arr[0] - 1.0'f32) < 0.001
    check abs(arr[1] - 0.5'f32) < 0.001
    check abs(arr[2] - 0.25'f32) < 0.001

  test "reciprocalData zero":
    let td = newTestTensorData(newShape(2), dtFloat32, @[0.0'f32, 2.0])
    let result = reciprocalData(td)
    let arr = result.asFloat32
    check arr[0] == 0.0'f32  # 1/0 returns 0 (handled)
    check abs(arr[1] - 0.5'f32) < 0.001

suite "TensorData Binary Operations":
  test "addData":
    let a = newTestTensorData(newShape(3), dtFloat32, @[1.0'f32, 2.0, 3.0])
    let b = newTestTensorData(newShape(3), dtFloat32, @[4.0'f32, 5.0, 6.0])
    let result = addData(a, b)
    let arr = result.asFloat32
    check arr[0] == 5.0'f32
    check arr[1] == 7.0'f32
    check arr[2] == 9.0'f32

  test "subData":
    let a = newTestTensorData(newShape(3), dtFloat32, @[5.0'f32, 7.0, 9.0])
    let b = newTestTensorData(newShape(3), dtFloat32, @[1.0'f32, 2.0, 3.0])
    let result = subData(a, b)
    let arr = result.asFloat32
    check arr[0] == 4.0'f32
    check arr[1] == 5.0'f32
    check arr[2] == 6.0'f32

  test "mulData":
    let a = newTestTensorData(newShape(3), dtFloat32, @[2.0'f32, 3.0, 4.0])
    let b = newTestTensorData(newShape(3), dtFloat32, @[5.0'f32, 6.0, 7.0])
    let result = mulData(a, b)
    let arr = result.asFloat32
    check arr[0] == 10.0'f32
    check arr[1] == 18.0'f32
    check arr[2] == 28.0'f32

  test "divData":
    let a = newTestTensorData(newShape(3), dtFloat32, @[10.0'f32, 18.0, 28.0])
    let b = newTestTensorData(newShape(3), dtFloat32, @[2.0'f32, 3.0, 4.0])
    let result = divData(a, b)
    let arr = result.asFloat32
    check abs(arr[0] - 5.0'f32) < 0.001
    check abs(arr[1] - 6.0'f32) < 0.001
    check abs(arr[2] - 7.0'f32) < 0.001

  test "divData by zero":
    let a = newTestTensorData(newShape(2), dtFloat32, @[10.0'f32, 5.0])
    let b = newTestTensorData(newShape(2), dtFloat32, @[0.0'f32, 5.0])
    let result = divData(a, b)
    let arr = result.asFloat32
    check arr[0] == 0.0'f32  # Division by zero returns 0
    check abs(arr[1] - 1.0'f32) < 0.001

  test "powData":
    let base = newTestTensorData(newShape(3), dtFloat32, @[2.0'f32, 3.0, 4.0])
    let exp = newTestTensorData(newShape(3), dtFloat32, @[2.0'f32, 2.0, 0.5])
    let result = powData(base, exp)
    let arr = result.asFloat32
    check abs(arr[0] - 4.0'f32) < 0.001
    check abs(arr[1] - 9.0'f32) < 0.001
    check abs(arr[2] - 2.0'f32) < 0.001

  test "scaleData":
    let td = newTestTensorData(newShape(3), dtFloat32, @[1.0'f32, 2.0, 3.0])
    let result = scaleData(td, 2.5)
    let arr = result.asFloat32
    check abs(arr[0] - 2.5'f32) < 0.001
    check abs(arr[1] - 5.0'f32) < 0.001
    check abs(arr[2] - 7.5'f32) < 0.001

  test "scaleData float64":
    let td = newTestTensorDataF64(newShape(3), @[1.0'f64, 2.0, 3.0])
    let result = scaleData(td, 2.5)
    let arr = result.asFloat64
    check abs(arr[0] - 2.5'f64) < 0.001
    check abs(arr[1] - 5.0'f64) < 0.001
    check abs(arr[2] - 7.5'f64) < 0.001

suite "TensorData Activation Functions":
  test "reluData":
    let td = newTestTensorData(newShape(5), dtFloat32, @[-2.0'f32, -0.1, 0.0, 0.5, 3.0])
    let result = reluData(td)
    let arr = result.asFloat32
    check arr[0] == 0.0'f32
    check arr[1] == 0.0'f32
    check arr[2] == 0.0'f32
    check arr[3] == 0.5'f32
    check arr[4] == 3.0'f32

  test "reluMaskData":
    let td = newTestTensorData(newShape(5), dtFloat32, @[-2.0'f32, -0.1, 0.0, 0.5, 3.0])
    let result = reluMaskData(td)
    let arr = result.asFloat32
    check arr[0] == 0.0'f32
    check arr[1] == 0.0'f32
    check arr[2] == 0.0'f32
    check arr[3] == 1.0'f32
    check arr[4] == 1.0'f32

  test "sigmoidData":
    let td = newTestTensorData(newShape(3), dtFloat32, @[0.0'f32, 100.0, -100.0])
    let result = sigmoidData(td)
    let arr = result.asFloat32
    check abs(arr[0] - 0.5'f32) < 0.001
    check abs(arr[1] - 1.0'f32) < 0.001  # Very large positive -> ~1
    check abs(arr[2] - 0.0'f32) < 0.001  # Very large negative -> ~0

  test "leakyReluMaskData":
    let td = newTestTensorData(newShape(4), dtFloat32, @[-2.0'f32, 0.0, 0.5, 3.0])
    let result = leakyReluMaskData(td, 0.1)
    let arr = result.asFloat32
    check abs(arr[0] - 0.1'f32) < 0.001  # Negative -> alpha
    check abs(arr[1] - 0.1'f32) < 0.001  # Zero -> alpha
    check arr[2] == 1.0'f32  # Positive -> 1
    check arr[3] == 1.0'f32

suite "TensorData Matrix Operations":
  test "transposeData":
    # 2x3 matrix: [[1,2,3], [4,5,6]]
    let td = newTestTensorData(newShape(2, 3), dtFloat32,
      @[1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0])
    let result = transposeData(td)
    check result.shape == newShape(3, 2)
    let arr = result.asFloat32
    # Should be [[1,4], [2,5], [3,6]]
    check arr[0] == 1.0'f32
    check arr[1] == 4.0'f32
    check arr[2] == 2.0'f32
    check arr[3] == 5.0'f32
    check arr[4] == 3.0'f32
    check arr[5] == 6.0'f32

  test "matmulData":
    # A: 2x3, B: 3x2 -> Result: 2x2
    let a = newTestTensorData(newShape(2, 3), dtFloat32,
      @[1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0])
    let b = newTestTensorData(newShape(3, 2), dtFloat32,
      @[1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0])
    let result = matmulData(a, b)
    check result.shape == newShape(2, 2)
    let arr = result.asFloat32
    # [1,2,3] @ [1,3,5; 2,4,6] = [22, 28; 49, 64]
    check abs(arr[0] - 22.0'f32) < 0.001
    check abs(arr[1] - 28.0'f32) < 0.001
    check abs(arr[2] - 49.0'f32) < 0.001
    check abs(arr[3] - 64.0'f32) < 0.001

  test "matmulData identity":
    # A @ I = A
    let a = newTestTensorData(newShape(2, 2), dtFloat32,
      @[1.0'f32, 2.0, 3.0, 4.0])
    let identity = newTestTensorData(newShape(2, 2), dtFloat32,
      @[1.0'f32, 0.0, 0.0, 1.0])
    let result = matmulData(a, identity)
    let arr = result.asFloat32
    check abs(arr[0] - 1.0'f32) < 0.001
    check abs(arr[1] - 2.0'f32) < 0.001
    check abs(arr[2] - 3.0'f32) < 0.001
    check abs(arr[3] - 4.0'f32) < 0.001

suite "TensorData Reduction Operations":
  test "sumData":
    let td = newTestTensorData(newShape(4), dtFloat32, @[1.0'f32, 2.0, 3.0, 4.0])
    let result = sumData(td)
    check result.shape == newShape(1)
    let arr = result.asFloat32
    check abs(arr[0] - 10.0'f32) < 0.001

  test "meanData":
    let td = newTestTensorData(newShape(4), dtFloat32, @[1.0'f32, 2.0, 3.0, 4.0])
    let result = meanData(td)
    check result.shape == newShape(1)
    let arr = result.asFloat32
    check abs(arr[0] - 2.5'f32) < 0.001

  test "broadcastData scalar to vector":
    let td = newTestTensorData(newShape(1), dtFloat32, @[5.0'f32])
    let result = broadcastData(td, newShape(4))
    check result.shape == newShape(4)
    let arr = result.asFloat32
    check arr[0] == 5.0'f32
    check arr[1] == 5.0'f32
    check arr[2] == 5.0'f32
    check arr[3] == 5.0'f32

  test "onesData":
    let result = onesData(newShape(3), dtFloat32)
    let arr = result.asFloat32
    check arr[0] == 1.0'f32
    check arr[1] == 1.0'f32
    check arr[2] == 1.0'f32

  test "zerosData":
    let result = zerosData(newShape(3), dtFloat32)
    let arr = result.asFloat32
    check arr[0] == 0.0'f32
    check arr[1] == 0.0'f32
    check arr[2] == 0.0'f32

suite "TensorRef Operations":
  test "neg TensorRef":
    let tr = newTestTensorRef(newShape(3), dtFloat32, @[1.0'f32, -2.0, 3.0])
    let result = neg(tr)
    let values = getValues(result)
    check values[0] == -1.0'f32
    check values[1] == 2.0'f32
    check values[2] == -3.0'f32

  test "neg nil returns nil":
    let result = neg(nil)
    check result.isNil

  test "add TensorRef":
    let a = newTestTensorRef(newShape(3), dtFloat32, @[1.0'f32, 2.0, 3.0])
    let b = newTestTensorRef(newShape(3), dtFloat32, @[4.0'f32, 5.0, 6.0])
    let result = add(a, b)
    let values = getValues(result)
    check values[0] == 5.0'f32
    check values[1] == 7.0'f32
    check values[2] == 9.0'f32

  test "add nil returns nil":
    let a = newTestTensorRef(newShape(3), dtFloat32, @[1.0'f32, 2.0, 3.0])
    check add(a, nil).isNil
    check add(nil, a).isNil
    check add(nil, nil).isNil

  test "sub TensorRef":
    let a = newTestTensorRef(newShape(3), dtFloat32, @[5.0'f32, 7.0, 9.0])
    let b = newTestTensorRef(newShape(3), dtFloat32, @[1.0'f32, 2.0, 3.0])
    let result = sub(a, b)
    let values = getValues(result)
    check values[0] == 4.0'f32
    check values[1] == 5.0'f32
    check values[2] == 6.0'f32

  test "mul TensorRef":
    let a = newTestTensorRef(newShape(3), dtFloat32, @[2.0'f32, 3.0, 4.0])
    let b = newTestTensorRef(newShape(3), dtFloat32, @[5.0'f32, 6.0, 7.0])
    let result = mul(a, b)
    let values = getValues(result)
    check values[0] == 10.0'f32
    check values[1] == 18.0'f32
    check values[2] == 28.0'f32

  test "div TensorRef":
    let a = newTestTensorRef(newShape(3), dtFloat32, @[10.0'f32, 18.0, 28.0])
    let b = newTestTensorRef(newShape(3), dtFloat32, @[2.0'f32, 3.0, 4.0])
    let result = `div`(a, b)
    let values = getValues(result)
    check abs(values[0] - 5.0'f32) < 0.001
    check abs(values[1] - 6.0'f32) < 0.001
    check abs(values[2] - 7.0'f32) < 0.001

  test "scale TensorRef":
    let tr = newTestTensorRef(newShape(3), dtFloat32, @[1.0'f32, 2.0, 3.0])
    let result = scale(tr, 2.5)
    let values = getValues(result)
    check abs(values[0] - 2.5'f32) < 0.001
    check abs(values[1] - 5.0'f32) < 0.001
    check abs(values[2] - 7.5'f32) < 0.001

  test "sign TensorRef":
    let tr = newTestTensorRef(newShape(3), dtFloat32, @[-2.0'f32, 0.0, 3.0])
    let result = sign(tr)
    let values = getValues(result)
    check values[0] == -1.0'f32
    check values[1] == 0.0'f32
    check values[2] == 1.0'f32

  test "square TensorRef":
    let tr = newTestTensorRef(newShape(3), dtFloat32, @[2.0'f32, 3.0, 4.0])
    let result = square(tr)
    let values = getValues(result)
    check abs(values[0] - 4.0'f32) < 0.001
    check abs(values[1] - 9.0'f32) < 0.001
    check abs(values[2] - 16.0'f32) < 0.001

  test "reciprocal TensorRef":
    let tr = newTestTensorRef(newShape(3), dtFloat32, @[1.0'f32, 2.0, 4.0])
    let result = reciprocal(tr)
    let values = getValues(result)
    check abs(values[0] - 1.0'f32) < 0.001
    check abs(values[1] - 0.5'f32) < 0.001
    check abs(values[2] - 0.25'f32) < 0.001

  test "sqrtRef TensorRef":
    let tr = newTestTensorRef(newShape(3), dtFloat32, @[4.0'f32, 9.0, 16.0])
    let result = sqrtRef(tr)
    let values = getValues(result)
    check abs(values[0] - 2.0'f32) < 0.001
    check abs(values[1] - 3.0'f32) < 0.001
    check abs(values[2] - 4.0'f32) < 0.001

  test "expRef TensorRef":
    let tr = newTestTensorRef(newShape(2), dtFloat32, @[0.0'f32, 1.0])
    let result = expRef(tr)
    let values = getValues(result)
    check abs(values[0] - 1.0'f32) < 0.001
    check abs(values[1] - E.float32) < 0.001

  test "logRef TensorRef":
    let tr = newTestTensorRef(newShape(2), dtFloat32, @[1.0'f32, E.float32])
    let result = logRef(tr)
    let values = getValues(result)
    check abs(values[0] - 0.0'f32) < 0.001
    check abs(values[1] - 1.0'f32) < 0.001

  test "sinRef TensorRef":
    let tr = newTestTensorRef(newShape(2), dtFloat32, @[0.0'f32, PI.float32/2])
    let result = sinRef(tr)
    let values = getValues(result)
    check abs(values[0] - 0.0'f32) < 0.001
    check abs(values[1] - 1.0'f32) < 0.001

  test "cosRef TensorRef":
    let tr = newTestTensorRef(newShape(2), dtFloat32, @[0.0'f32, PI.float32])
    let result = cosRef(tr)
    let values = getValues(result)
    check abs(values[0] - 1.0'f32) < 0.001
    check abs(values[1] - (-1.0'f32)) < 0.001

  test "tanhRef TensorRef":
    let tr = newTestTensorRef(newShape(2), dtFloat32, @[0.0'f32, 1.0])
    let result = tanhRef(tr)
    let values = getValues(result)
    check abs(values[0] - 0.0'f32) < 0.001
    check abs(values[1] - tanh(1.0'f32)) < 0.001

  test "transpose TensorRef":
    let td = newTestTensorData(newShape(2, 3), dtFloat32,
      @[1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0])
    let tr = newComputedTensorRef(td)
    let result = transpose(tr)
    check result.shape == newShape(3, 2)

  test "matmul TensorRef":
    let aTd = newTestTensorData(newShape(2, 3), dtFloat32,
      @[1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0])
    let bTd = newTestTensorData(newShape(3, 2), dtFloat32,
      @[1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0])
    let a = newComputedTensorRef(aTd)
    let b = newComputedTensorRef(bTd)
    let result = matmul(a, b)
    check result.shape == newShape(2, 2)

  test "reluMask TensorRef":
    let tr = newTestTensorRef(newShape(4), dtFloat32, @[-2.0'f32, 0.0, 0.5, 3.0])
    let result = reluMask(tr)
    let values = getValues(result)
    check values[0] == 0.0'f32
    check values[1] == 0.0'f32
    check values[2] == 1.0'f32
    check values[3] == 1.0'f32

  test "sigmoidRef TensorRef":
    let tr = newTestTensorRef(newShape(2), dtFloat32, @[0.0'f32, 100.0])
    let result = sigmoidRef(tr)
    let values = getValues(result)
    check abs(values[0] - 0.5'f32) < 0.001
    check abs(values[1] - 1.0'f32) < 0.001

  test "leakyReluMask TensorRef":
    let tr = newTestTensorRef(newShape(3), dtFloat32, @[-2.0'f32, 0.0, 3.0])
    let result = leakyReluMask(tr, 0.1)
    let values = getValues(result)
    check abs(values[0] - 0.1'f32) < 0.001
    check abs(values[1] - 0.1'f32) < 0.001
    check values[2] == 1.0'f32

  test "broadcast TensorRef":
    let tr = newTestTensorRef(newShape(1), dtFloat32, @[5.0'f32])
    let result = broadcast(tr, newShape(4))
    check result.shape == newShape(4)
    let values = getValues(result)
    check values[0] == 5.0'f32
    check values[3] == 5.0'f32

  test "ones TensorRef":
    let result = ones(newShape(3), dtFloat32)
    let values = getValues(result)
    check values[0] == 1.0'f32
    check values[1] == 1.0'f32
    check values[2] == 1.0'f32

  test "zeros TensorRef":
    let result = zeros(newShape(3), dtFloat32)
    let values = getValues(result)
    check values[0] == 0.0'f32
    check values[1] == 0.0'f32
    check values[2] == 0.0'f32

suite "Error Cases and Edge Cases":
  test "single element tensor operations":
    let td = newTestTensorData(newShape(1), dtFloat32, @[5.0'f32])
    let neg_result = negData(td)
    check neg_result.size == 1
    let arr = neg_result.asFloat32
    check arr[0] == -5.0'f32

  test "operations preserve dtype":
    let td32 = newTestTensorData(newShape(2), dtFloat32, @[1.0'f32, 2.0])
    let td64 = newTestTensorDataF64(newShape(2), @[1.0'f64, 2.0])

    check negData(td32).dtype == dtFloat32
    check negData(td64).dtype == dtFloat64

  test "2D operations on non-2D tensors fail gracefully":
    # transposeData requires 2D
    let td1d = newTestTensorData(newShape(5), dtFloat32, @[1.0'f32, 2.0, 3.0, 4.0, 5.0])
    expect AssertionDefect:
      discard transposeData(td1d)

  test "matmul dimension mismatch fails":
    let a = newTestTensorData(newShape(2, 3), dtFloat32,
      @[1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0])
    let b = newTestTensorData(newShape(4, 2), dtFloat32,
      @[1.0'f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    expect AssertionDefect:
      discard matmulData(a, b)

  test "binary ops shape mismatch fails":
    let a = newTestTensorData(newShape(3), dtFloat32, @[1.0'f32, 2.0, 3.0])
    let b = newTestTensorData(newShape(4), dtFloat32, @[1.0'f32, 2.0, 3.0, 4.0])
    expect AssertionDefect:
      discard addData(a, b)
    expect AssertionDefect:
      discard mulData(a, b)
