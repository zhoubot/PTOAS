// RUN: ptoas %s | FileCheck %s

// Verify stable custom parse/print formats:
//   - !pto.vtile<lanes x elementType>
//   - !pto.uscalar<elementType>
//   - #pto.target_config<arch=..., ...>

module attributes {
  // Attribute name is arbitrary; we only care about the attr payload printing.
  pto.test_target = #pto.target_config<arch=a3, isa="kirin9030", variant="v1", repeat_bytes=256, block_bytes=32, caps={foo = 1 : i32, bar = "baz"}>
} {
  func.func @type_roundtrip(%v: !pto.vtile<64xf32>, %s: !pto.uscalar<f32>) {
    return
  }
}

// CHECK: module attributes
// CHECK: #pto.target_config<{{.*}}arch=a3{{.*}}isa="kirin9030"{{.*}}variant="v1"{{.*}}repeat_bytes=256{{.*}}block_bytes=32{{.*}}caps={{\{}}[[CAPS:.*]]{{\}}}>
// CHECK: func.func @type_roundtrip(%{{.*}}: !pto.vtile<64xf32>, %{{.*}}: !pto.uscalar<f32>)
