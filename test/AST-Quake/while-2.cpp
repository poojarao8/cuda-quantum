/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// RUN: cudaq-quake %s | FileCheck --check-prefixes=EARLY %s
// RUN: cudaq-quake %s | cudaq-opt --canonicalize | FileCheck %s

#include <cudaq.h>

void fun(int);

void uma(cudaq::qubit &, cudaq::qubit &, cudaq::qubit &);

__qpu__ void test1(cudaq::qspan<> a, cudaq::qspan<> b) {
  uint32_t i = a.size();
  while (i > 0) {
    uma(a[i - 1ul], b[i - 1ul], a[i]);
    break;
  }
}

__qpu__ void test2(cudaq::qspan<> a, cudaq::qspan<> b) {
  uint32_t i = a.size();
  while (i > 0) {
    uma(a[i - 1ul], b[i - 1ul], a[i]);
    if (i == 57) {
      int x = 42;
      fun(x);
      break;
    }
  }
}

__qpu__ void test3(cudaq::qspan<> a, cudaq::qspan<> b) {
  uint32_t i = a.size();
  while (i > 0) {
    uma(a[i - 1ul], b[i - 1ul], a[i]);
    continue;
  }
}

__qpu__ void test4(cudaq::qspan<> a, cudaq::qspan<> b) {
  uint32_t i = a.size();
  while (i > 0) {
    uma(a[i - 1ul], b[i - 1ul], a[i]);
    if (i == 57) {
      fun(87);
      continue;
    }
  }
}

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test1
// CHECK:           cc.loop while {
// CHECK:             cc.condition %
// CHECK:           } do {
// CHECK:             func.call @
// CHECK:             cc.break
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test2
// CHECK:           cc.loop while {
// CHECK:             cc.condition %
// CHECK:           } do {
// CHECK:             func.call @
// CHECK:             cc.if(%{{.*}}) {
// CHECK:               cc.scope {
// CHECK:                 func.call @
// CHECK:                 cc.unwind_break
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test3
// CHECK:           cc.loop while {
// CHECK:             cc.condition %
// CHECK:           } do {
// CHECK:             func.call @
// CHECK:             cc.continue
// CHECK:           }
// CHECK:           return
// CHECK:         }

// CHECK-LABEL:   func.func @__nvqpp__mlirgen__function_test4
// CHECK:           cc.loop while {
// CHECK:             cc.condition %
// CHECK:           } do {
// CHECK:             func.call @
// CHECK:             cc.if(%{{.*}}) {
// CHECK:               func.call @
// CHECK:               cc.unwind_continue
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }

// EARLY-LABEL:   func.func @__nvqpp__mlirgen__function_test1
// EARLY:           cc.loop while {
// EARLY:             cc.condition %
// EARLY:           } do {
// EARLY:             cc.scope {
// EARLY:               func.call @
// EARLY:               cc.unwind_break
// EARLY:             }
// EARLY:             cc.continue
// EARLY:           }
// EARLY:           return
// EARLY:         }

// EARLY-LABEL:   func.func @__nvqpp__mlirgen__function_test2
// EARLY:           cc.loop while {
// EARLY:             cc.condition %
// EARLY:           } do {
// EARLY:             cc.scope {
// EARLY:               func.call @
// EARLY:               cc.if(%{{.*}}) {
// EARLY:                 cc.scope {
// EARLY:                   func.call @
// EARLY:                   cc.unwind_break
// EARLY:                 }
// EARLY:               }
// EARLY:             }
// EARLY:             cc.continue
// EARLY:           }
// EARLY:           return
// EARLY:         }

// EARLY-LABEL:   func.func @__nvqpp__mlirgen__function_test3
// EARLY:           cc.loop while {
// EARLY:             cc.condition %
// EARLY:           } do {
// EARLY:             cc.scope {
// EARLY:               func.call @
// EARLY:               cc.unwind_continue
// EARLY:             }
// EARLY:             cc.continue
// EARLY:           }
// EARLY:           return
// EARLY:         }

// EARLY-LABEL:   func.func @__nvqpp__mlirgen__function_test4
// EARLY:           cc.loop while {
// EARLY:             cc.condition %
// EARLY:           } do {
// EARLY:             cc.scope {
// EARLY:               func.call @
// EARLY:               cc.if(%{{.*}}) {
// EARLY:                 func.call @
// EARLY:                 cc.unwind_continue
// EARLY:               }
// EARLY:             }
// EARLY:             cc.continue
// EARLY:           }
// EARLY:           return
// EARLY:         }
