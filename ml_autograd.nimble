# Package

version       = "0.0.5"
author        = "jasagiri"
description   = "Automatic differentiation for ML framework"
license       = "Apache-2.0"
srcDir        = "src"

# Dependencies

requires "nim >= 2.0.0"
requires "ml_core >= 0.0.4"

# Tasks
task test, "Run tests":
  exec "nim c -r --path:../nim-ml-core/src tests/test_tape.nim"
  exec "nim c -r --path:../nim-ml-core/src tests/test_backward.nim"
  exec "nim c -r --path:../nim-ml-core/src tests/test_gradients.nim"
