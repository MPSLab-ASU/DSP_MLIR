//===- ShapedOpInterfaces.h - Interfaces for Shaped Ops ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a set of interfaces for ops that operate on shaped values.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_INTERFACES_SHAPEINFERENCEOPINTERFACES_H_
#define MLIR_INTERFACES_SHAPEINFERENCEOPINTERFACES_H_

#include "mlir/IR/OpDefinition.h"

//namespace mlir {
//namespace dsp {

/// Verify invariants of ops that implement the ShapedDimOpInterface.
// LogicalResult verifyShapedDimOpInterface(Operation *op);
#include "mlir/Interfaces/ShapeInferenceOpInterfaces.h.inc"
//} // namespace detail
//} // namespace mlir

/// Include the generated interface declarations.
// #include "mlir/Interfaces/ShapeInferenceOpInterfaces.h.inc"

#endif // MLIR_INTERFACES_SHAPEDOPINTERFACES_H_
