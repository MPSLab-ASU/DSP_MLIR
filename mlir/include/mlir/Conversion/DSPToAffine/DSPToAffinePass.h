#ifndef MLIR_CONVERSION_DSPTOAFFINE_DSPTOAFFINEPASS_H
#define MLIR_CONVERSION_DSPTOAFFINE_DSPTOAFFINEPASS_H

#include "mlir/Pass/Pass.h"

namespace mlir {
class ModuleOp;

#define GEN_PASS_DECL_CONVERTDSPTOAFFINE
#include "mlir/Conversion/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createConvertDSPToAffinePass();

} // namespace mlir

#endif // MLIR_CONVERSION_DSPTOAFFINE_DSPTOAFFINEPASS_H
