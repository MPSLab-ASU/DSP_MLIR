//===-- Passes.td - DSP pass definition file ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DSP_TRANSFORMS_PASSES_TD
#define MLIR_DIALECT_DSP_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def DSPInferShapes : Pass<"dsp-infer-shapes", "func::FuncOp"> {
  let summary = "Propagate shapes across DSP operations";
  let description = [{
    Pass that uses operand types and propagates shapes to DSP operations.
    This includes legalizing rankless and dynamic shapes towards static.
  }];

  // let constructor = "createDSPInferShapesPass()";
  let dependentDialects = [
    "func::FuncDialect",
    "tensor::TensorDialect",
    "dsp::DSPDialect",
  ];
}

#endif // MLIR_DIALECT_DSP_TRANSFORMS_PASSES_TD

