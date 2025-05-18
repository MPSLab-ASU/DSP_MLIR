//===- Passes.h - DSP pipeline entry points -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes of all DSP pipelines.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DSP_PIPELINES_PASSES_H
#define MLIR_DIALECT_DSP_PIPELINES_PASSES_H

#include "mlir/Dialect/DSP/IR/DSPDialect.h"
#include "mlir/Pass/PassOptions.h"

namespace mlir {
class Pass;

namespace dsp {

/// Options for the dsp pipeline.
struct DSPPipelineOptions
    : public PassPipelineOptions<DSPPipelineOptions> {
  /* This is from Bufferization dialect.
  PassOptions::Option<bool> privateFunctionDynamicOwnership{
      *this, "private-function-dynamic-ownership",
      llvm::cl::desc(
          "Allows to add additional results to private functions to return "
          "ownership of returned memrefs to callers. This can avoid spurious "
          "buffer clones in the callee."),
      llvm::cl::init(false)};

  /// Implicit conversion to `DeallocationOptions`.
  operator DeallocationOptions() const {
    DeallocationOptions options;
    options.privateFuncDynamicOwnership = privateFunctionDynamicOwnership;
    return options;
  }
  */
};

//===----------------------------------------------------------------------===//
// Building and Registering.
//===----------------------------------------------------------------------===//

/// Adds the dsp pipeline to the `OpPassManager`.
void buildDSPPipeline(
    OpPassManager &pm, const DSPPipelineOptions &options);

/// Registers all pipelines for the `dsp` dialect. Currently,
/// this includes only the "dsp-pipeline".
void registerDSPPipelines();

std::unique_ptr<Pass> createShapeInferencePass();

} // namespace dsp
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_PIPELINES_PASSES_H
