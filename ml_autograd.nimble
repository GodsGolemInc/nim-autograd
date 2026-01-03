# Package

version       = "0.0.6"
author        = "jasagiri"
description   = "Automatic differentiation for ML framework"
license       = "Apache-2.0"
srcDir        = "src"

# Dependencies

requires "nim >= 2.0.0"
requires "ml_core >= 0.0.4"

# Tasks
task test, "Run tests":
  echo "Running unit tests..."
  exec "nim c -r --path:../nim-ml-core/src tests/test_tape.nim"
  exec "nim c -r --path:../nim-ml-core/src tests/test_backward.nim"
  exec "nim c -r --path:../nim-ml-core/src tests/test_gradients.nim"
  exec "nim c -r --path:../nim-ml-core/src tests/test_tensor_ops.nim"
  echo "Running integration tests..."
  exec "nim c -r --path:../nim-ml-core/src tests/test_integration.nim"
  echo "Running use case tests..."
  exec "nim c -r --path:../nim-ml-core/src tests/test_usecases.nim"
  echo "All tests passed!"
