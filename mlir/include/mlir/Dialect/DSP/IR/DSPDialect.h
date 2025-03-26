//===- Dialect.h - Dialect definition for the DSP IR ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the IR Dialect for the DSP language.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_DSP_IR_DSPDIALECT_H_
#define MLIR_DIALECT_DSP_IR_DSPDIALECT_H_

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/ShapeInferenceOpInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Dialect/DSP/IR/DSPDialect.h.inc"

// #include "mlir/Dialect/DSP/IR/DSPEnums.h.inc"

/// Include the auto-generated header file containing the declarations of the
/// DSP operations.
#define GET_OP_CLASSES
#include "mlir/Dialect/DSP/IR/DSP.h.inc"


#endif // MLIR_DIALECT_DSP_IR_DSPDIALECT_H_
