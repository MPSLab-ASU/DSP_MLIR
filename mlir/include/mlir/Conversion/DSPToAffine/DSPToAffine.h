//===-- DSPToAffine.h - DSP optimization pass declarations --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the passes for the DSP to Affine Dialect conversion.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONVERSION_DSPTOAFFINE_DSPTOAFFINE_H
#define MLIR_CONVERSION_DSPTOAFFINE_DSPTOAFFINE_H
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
class ModuleOp;

#define GEN_PASS_DECL_CONVERTDSPTOAFFINE
#include "mlir/Conversion/Passes.h.inc"
namespace dsp {
void populateDSPToAffineConversionPatterns(RewritePatternSet &patterns);
} // namespace dsp
} // namespace mlir

#endif // MLIR_CONVERSION_DSPTOAFFINE_DSPTOAFFINE_H
