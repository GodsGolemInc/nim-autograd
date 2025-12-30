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
| `tape` | Gradient tape for recording operations |
| `backward` | Backward pass implementation |
| `gradients` | Gradient function implementations |

## Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| tape | 34 | Pass |
| backward | 19 | Pass |
| gradients | 25 | Pass |
| **Total** | **78** | **100%** |

## Running Tests

```bash
nimble test
```

Or run individual test files:
```bash
nim c -r --path:../nim-ml-core/src tests/test_tape.nim
nim c -r --path:../nim-ml-core/src tests/test_backward.nim
nim c -r --path:../nim-ml-core/src tests/test_gradients.nim
```

## Architecture

```
ml_autograd
├── tape.nim       # GradientTape, TapeEntry, watch/record
├── backward.nim   # backward(), computeGradients(), chain rules
└── gradients.nim  # Gradient functions for all operations
```

## License

Apache-2.0
