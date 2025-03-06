//===- DSPToAffinePass.cpp - Lowering DSP to Affine Dialect -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This  file implements a pass to convert DSP dialect to the Affine dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/DSPToAffine/DSPToAffinePass.h"
#include "mlir/Conversion/DSPToAffine/DSPToAffine.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#define GEN_PASS_DEF_CONVERTDSPTOAFFINE

namespace mlir {
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
class ConvertDSPToAffinePass
    : public impl::ConvertDSPToAffineBase<ConvertDSPToAffinePass> {

  void runOnOperation() override {
    auto &context = getContext();
    ConversionTarget target(context);

    target.addLegalDialect<mlir::affine::AffineDialect, mlir::BuiltinDialect,
                           mlir::arith::ArithDialect,  mlir::func::FuncDialect,
                           mlir::memref::MemRefDialect,  mlir::math::MathDialect,
                           mlir::scf::SCFDialect>();

    target.addIllegalDialect<dsp::DSPDialect>();
    target.addDynamicallyLegalOp<dsp::PrintOp>([](dsp::PrintOp op) {
      // return ::nollvmne_of(op->getOperandTypes(), [](Type type) {
      //   return llvm::isa<TensorType>(type);
      // });
      return llvm::none_of(op->getOperandTypes(), [](Type type) {
        return llvm::isa<TensorType>(type);
      });
      
    });
    RewritePatternSet patterns(&context);
    populateDSPToAffineConversionPatterns(patterns);

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      return signalPassFailure();
  }
};
} // namespace
std::unique_ptr<OperationPass<ModuleOp>>
mlir::createConvertDSPToAffinePass() {
  return std::make_unique<ConvertDSPToAffinePass>();
}