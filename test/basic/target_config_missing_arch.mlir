// RUN: not ptoas %s 2>&1 | FileCheck %s

module attributes {
  // Missing required field: arch
  pto.test_target = #pto.target_config<isa="kirin9030">
} {
}

// CHECK: error:
// CHECK-SAME: target_config
// CHECK: arch
