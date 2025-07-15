//===- DSPPipelines.cpp - Pipelines for bufferization -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DSP/Pipelines/Passes.h"

#include "mlir/Dialect/DSP/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "mlir/Dialect/DSP/IR/DSPDialect.h"
//#include "toy/Passes.h"
//#include "toy/ShapeInferenceInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"
#include <memory>

#include "mlir/Interfaces/ShapeInferenceOpInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Pipeline implementation.
//===----------------------------------------------------------------------===//

void mlir::dsp::buildDSPPipeline(
    OpPassManager &pm, const DSPPipelineOptions &options) {

  pm.addPass(mlir::createInlinerPass());  
  // Now that there is only one function, we can infer the shapes of each of
  // the operations.                                                 
  
  mlir::OpPassManager &optPM = pm.nest<mlir::dsp::FuncOp>();

  optPM.addPass(mlir::dsp::createShapeInferencePass());

  
}

//===----------------------------------------------------------------------===//
// Pipeline registration.
//===----------------------------------------------------------------------===//

void mlir::dsp::registerDSPPipelines() {
  PassPipelineRegistration<DSPPipelineOptions>(
      "dsp-shapeinference",
      "Implements Shape Inference and Inlining for DSP dialect operations.",
      buildDSPPipeline);
}



using namespace mlir;
using namespace dsp;

namespace {
	
/// The ShapeInferencePass is a pass that performs intra-procedural
/// shape inference.
///
///    Algorithm:
///
///   1) Build a worklist containing all the operations that return a
///      dynamically shaped tensor: these are the operations that need shape
///      inference.
///   2) Iterate on the worklist:
///     a) find an operation to process: the next ready operation in the
///        worklist has all of its arguments non-generic,
///     b) if no operation is found, break out of the loop,
///     c) remove the operation from the worklist,
///     d) infer the shape of its output from the argument types.
///   3) If the worklist is empty, the algorithm succeeded.
///
struct ShapeInferencePass
    : public mlir::PassWrapper<ShapeInferencePass, OperationPass<dsp::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ShapeInferencePass)

  void runOnOperation() override {
    auto f = getOperation();

    // Populate the worklist with the operations that need shape inference:
    // these are operations that return a dynamic shape.
    llvm::SmallPtrSet<mlir::Operation *, 16> opWorklist;
    f.walk([&](mlir::Operation *op) {
      if (returnsDynamicShape(op))
        opWorklist.insert(op);
    });

    // Iterate on the operations in the worklist until all operations have been
    // inferred or no change happened (fix point).
    while (!opWorklist.empty()) {
      // Find the next operation ready for inference, that is an operation
      // with all operands already resolved (non-generic).
      auto nextop = llvm::find_if(opWorklist, allOperandsInferred);
      if (nextop == opWorklist.end())
        break;

      Operation *op = *nextop;
      opWorklist.erase(op);

      // Ask the operation to infer its output shapes.
      //LLVM_DEBUG(llvm::dbgs() << "Inferring shape for: " << *op << "\n");
      if (auto shapeOp = dyn_cast<ShapeInference>(op)) {
        shapeOp.inferShapes();
      } else {
        op->emitError("unable to infer shape of operation without shape "
                      "inference interface");
        return signalPassFailure();
      }
    }

    // If the operation worklist isn't empty, this indicates a failure.
    if (!opWorklist.empty()) {
      f.emitError("Shape inference failed, ")
          << opWorklist.size() << " operations couldn't be inferred\n";
      signalPassFailure();
    }
  }

  /// A utility method that returns if the given operation has all of its
  /// operands inferred.
  static bool allOperandsInferred(Operation *op) {
    return llvm::all_of(op->getOperandTypes(), [](Type operandType) {
      return llvm::isa<RankedTensorType>(operandType);
    });
  }

  /// A utility method that returns if the given operation has a dynamically
  /// shaped result.
  static bool returnsDynamicShape(Operation *op) {
    return llvm::any_of(op->getResultTypes(), [](Type resultType) {
      return !llvm::isa<RankedTensorType>(resultType);
    });
  }
};
} // namespace

/// Create a Shape Inference pass.
std::unique_ptr<mlir::Pass> mlir::dsp::createShapeInferencePass() {
  return std::make_unique<ShapeInferencePass>();
}


