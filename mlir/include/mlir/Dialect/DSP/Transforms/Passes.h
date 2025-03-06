//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_DIALECT_DSP_TRANSFORMS_PASSES_H_
#define MLIR_DIALECT_DSP_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
class ConversionTarget;
namespace dsp {

#define GEN_PASS_DECL_DSPINFERSHAPES
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/DSP/Transforms/Passes.h.inc"

// void createDSPInferShapesPass();
} // namespace dsp
} // namespace mlir

#endif // MLIR_DIALECT_DSP_TRANSFORMS_PASSES_H_
