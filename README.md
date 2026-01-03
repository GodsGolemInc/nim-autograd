# nim-autograd

Automatic differentiation engine for the Nim ML framework.

## Features

- **Gradient Tape** - Records operations for automatic differentiation
- **Backward Pass** - Computes gradients through the computation graph
- **Gradient Functions** - Comprehensive set of gradient implementations
- **Chain Rule Helpers** - Optimized gradient propagation utilities

## Installation

```bash
nimble install ml_autograd
```

Or add to your `.nimble` file:
```nim
requires "ml_autograd >= 0.0.3"
```

## Dependencies

- `nim >= 2.0.0`
- `ml_core >= 0.0.4`

## Usage

```nim
import ml_autograd
import ml_core

# Create a gradient tape
var tape = newGradientTape()

# Watch variables for differentiation
let x = newTensorRef(newTensorData(DType.float32, newShape(3, 3)))
tape.watch(x)

# Record operations
tape.record("matmul", @[x, weight], output)

# Compute gradients
let ctx = newGradientContext(tape)
ctx.setGrad(output, gradOutput)
backward(tape, ctx)

# Get gradients
let gradX = ctx.getGrad(x)
```

## Gradient Functions

### Unary Operations
- `neg`, `abs`, `exp`, `log`, `sqrt`, `tanh`

### Binary Operations
- `add`, `sub`, `mul`, `div`, `pow`

### Matrix Operations
- `matmul`, `transpose`

### Activation Functions
- `relu`, `sigmoid`, `softmax`
- `leaky_relu`, `elu`, `gelu`

### Reduction Operations
- `sum`, `mean`

### Loss Functions
- `mse_loss`, `cross_entropy_loss`

## Modules

| Module | Description |
|--------|-------------|
| `tensor_ops` | Tensor arithmetic operations for gradients |
| `tape` | Gradient tape for recording operations |
| `backward` | Backward pass implementation |
| `gradients` | Gradient function implementations |

## Test Coverage

| Test Type | File | Tests | Status |
|-----------|------|-------|--------|
| Unit | test_tape.nim | 34 | Pass |
| Unit | test_backward.nim | 19 | Pass |
| Unit | test_gradients.nim | 25 | Pass |
| Unit | test_tensor_ops.nim | 70 | Pass |
| Integration | test_integration.nim | 21 | Pass |
| Use Case | test_usecases.nim | 14 | Pass |
| **Total** | | **183** | **100%** |

### Test Categories

- **Unit Tests (148)**: Individual function testing
- **Integration Tests (21)**: Component interaction testing
- **Use Case Tests (14)**: Real-world ML scenarios

See [docs/testing.md](docs/testing.md) for detailed test documentation.

## Running Tests

```bash
nimble test
```

Or run individual test files:
```bash
nim c -r --path:../nim-ml-core/src tests/test_tape.nim
nim c -r --path:../nim-ml-core/src tests/test_backward.nim
nim c -r --path:../nim-ml-core/src tests/test_gradients.nim
nim c -r --path:../nim-ml-core/src tests/test_tensor_ops.nim
nim c -r --path:../nim-ml-core/src tests/test_integration.nim
nim c -r --path:../nim-ml-core/src tests/test_usecases.nim
```

## Architecture

```
ml_autograd
├── tensor_ops.nim # TensorData/TensorRef arithmetic operations
├── tape.nim       # GradientTape, TapeEntry, watch/record
├── backward.nim   # backward(), computeGradients(), chain rules
└── gradients.nim  # Gradient functions for all operations
```

## License

Apache-2.0
