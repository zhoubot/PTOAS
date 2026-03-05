// RUN: not ptoas %s 2>&1 | FileCheck %s

module attributes {
  // Unknown key: foo
  pto.test_target = #pto.target_config<arch=a3, foo=1>
} {
}

// CHECK: error:
// CHECK-SAME: target_config
// CHECK: foo
