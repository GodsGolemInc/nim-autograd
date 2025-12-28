# Package

version       = "0.0.2"
author        = "jasagiri"
description   = "Automatic differentiation for ML framework"
license       = "MIT"
srcDir        = "src"

# Dependencies

requires "nim >= 2.0.0"
requires "nimml_core >= 0.0.4"

# Tasks
task test, "Run tests":
  exec "nim c -r --path:../nim-ml-core/src tests/test_tape.nim"
  exec "nim c -r --path:../nim-ml-core/src tests/test_backward.nim"
