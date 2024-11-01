//====- LowerToAffineLoops.cpp - Partial lowering from Toy to Affine+Std --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a partial lowering of Toy operations to a combination of
// affine loops, memref operations and standard operations. This lowering
// expects that all calls have been inlined, and all shapes have been resolved.
//
//===----------------------------------------------------------------------===//

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"
#include "toy/DebugConfig.h"
#include "toy/Dialect.h"
#include "toy/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Casting.h"
#include <algorithm>
#include <cstdint>
#include <functional>
#include <memory>
#include <utility>

// For IntegerSet
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IntegerSet.h"
#include <iostream>
using namespace mlir;
using namespace std;
using namespace affine;
using namespace dsp;
//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns
//===----------------------------------------------------------------------===//

// #pragma warning(push, 0)
/// Convert the given RankedTensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

/// Insert an allocation and deallocation for the given MemRefType.
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(
      &parentBlock->front()); // Abhinav-- move allock->block->front before
                              // alloc operation??

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as dsp functions have no control flow.
  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(
      &parentBlock->back()); // move alloc->block->back before dealloc
  return alloc;
}

/// This defines the function type used to process an iteration of a lowered
/// loop. It takes as input an OpBuilder, an range of memRefOperands
/// corresponding to the operands of the input operation, and the range of loop
/// induction variables for the iteration. It returns a value to store at the
/// current index of the iteration.
using LoopIterationFn = function_ref<Value(
    OpBuilder &rewriter, ValueRange memRefOperands, ValueRange loopIvs)>;

static void lowerOpToLoops(Operation *op, ValueRange operands,
                           PatternRewriter &rewriter,
                           LoopIterationFn processIteration) {
  auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

  // for (auto i : tensorType.getShape())
  // {
  //    llvm::errs() << "tensortype =" << i << "\n" ;
  // }
  // llvm::errs() << "tensortype.getElementType =" <<
  // tensorType.getElementType() << "\n" ; llvm::errs() << "op->getLoc = " <<
  // op->getLoc() << "\n"; //getDialect llvm::errs() << "op->getDialect = " <<
  // op->getDialect() << "\n"; llvm::errs() << "op->getName = " << op->getName()
  // << "\n";
  // // llvm::errs() << "op->getType = " << op->getType() << "\n";
  // llvm::errs() << "op->getParentRegion = " << op->getParentRegion() << "\n";
  // llvm::errs() << "op->getParentOp = " << op->getParentOp()->getName() <<
  // "\n";

  // llvm::errs() << "op->getNumOperands = " << op->getNumOperands() << "\n";
  // for (auto i : op->getOperands())
  // {
  //   llvm::errs() << "op->Operand = " << i << "\n";
  // }

  // llvm::errs() << "op->getParentOp = " << op->getParentOp()->getName() <<
  // "\n"; llvm::errs() << "op->getParentOp = " << op->getParentOp()->getName()
  // << "\n"; llvm::errs() << "op->getParentOp = " <<
  // op->getParentOp()->getName() << "\n";

  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create a nest of affine loops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

  // llvm::errs() << "tensorType.getRank() " << tensorType.getRank() << "\n";
  // cout << "tensorType.getRank() .. " << tensorType.getRank() << "\n";
  // for (auto i : tensorType.getRank())
  // {
  //   llvm::errs() << "tensorType.getRank() = " << i << "\n";
  // }
  // for (auto i : tensorType.getShape())
  // {
  //   llvm::errs() << "tensorType.getShape() = " << i << "\n";
  // }
  // llvm::errs() << "tensorType.getShape() " << tensorType.getShape() << "\n";
  affine::buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](OpBuilder &nestedBuilder, Location loc, ValueRange ivs) {
        // Call the processing function with the rewriter, the memref operands,
        // and the loop induction variables. This function will return the value
        // to store at the current index.
        Value valueToStore = processIteration(nestedBuilder, operands, ivs);
        nestedBuilder.create<affine::AffineStoreOp>(loc, valueToStore, alloc,
                                                    ivs);
      });

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}

#define TryJustAffineLoop 0       // working
#define TryAffineForAndAffineIf 0 // working
#define TryAffineIf2 0
#define TryAffineMap 0    // working basic -- TO do --try with symbols
#define TrySumOfVector 0  // Working
#define TryMultiDimLoop 0 // Working
#define TryFIRFilter 1
#define TryMultiDimForAndIf 0         //
#define TryMultiDimLoopAndAffineMap 0 // Working
#define TryMultiDimLoopAndAffineSet 0 // Working
static void lowerOpToLoopsFIR(Operation *op, ValueRange operands,
                              PatternRewriter &rewriter,
                              LoopIterationFn processIteration) {
  auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create a nest of affine loops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

  // llvm::errs() << "tensorType.getRank() " << tensorType.getRank() << "\n";
  // cout << "tensorType.getRank() .. " << tensorType.getRank() << "\n";
  // for (auto i : tensorType.getRank())
  // {
  //   llvm::errs() << "tensorType.getRank() = " << i << "\n";
  // }
  // for (auto i : tensorType.getShape())
  // {
  //   llvm::errs() << "tensorType.getShape() = " << i << "\n";
  // }
  // llvm::errs() << "tensorType.getShape() " << tensorType.getShape() << "\n";

  // affine::AffineForOp forOp = rewriter.create<affine::AffineForOp>(
  //   loc, lowerBounds, tensorType.getShape() , steps, ValueRange());
  // mlir::IntegerSet set1 = mlir::IntegerSet::get(1, 0, map, {true});

  // create an affineFor
  //  affineFor It has one region containing its body & the region must contain
  //  a block terminating with affine.yield
  // block has argument of index type
  //

#if TryJustAffineLoop
  int64_t lb = 0;
  int64_t ub = tensorType.getShape()[0];
  int64_t step = 1;

  // create AffineMap and set
  //  %1 = affine.load
  //   if ( %arg0 >= 5)   ie, integerSet <(d0) : (d0 - 5 >= 0) >
  AffineExpr dimExpr =
      rewriter.getAffineDimExpr(0) - rewriter.getAffineConstantExpr(5);
  // AffineMap map = AffineMap::get(1, 0, dimExpr);
  // AffineMap map = AffineMap::get(1, 0 , rewriter.getAffineDimExpr(0) - 5);
  IntegerSet set1 = IntegerSet::get(1, 0, {dimExpr}, {false});
  affine::AffineForOp forOp1 =
      rewriter.create<affine::AffineForOp>(loc, lb, ub, step);

  // inside the forOp body --> create the operations & then close the body
  //  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(forOp1.getBody());

  // start adding operations like a arith::constant = 100.0 to the body of
  // forOp1
  //  Inside the loop body:

  Value constant15 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(15));

  llvm::errs() << "LINE = " << __LINE__ << "\n";
  auto storeOp = rewriter.create<affine::AffineStoreOp>(
      loc, constant15, alloc, forOp1.getInductionVar());

#endif

#if TryAffineForAndAffineIf
  int64_t lb = 0;
  int64_t ub = tensorType.getShape()[0];
  int64_t step = 1;

  // create AffineMap and set
  //  %1 = affine.load
  //   if ( %arg0 >= 5)   ie, integerSet <(d0) : (d0 - 5 >= 0) >
  AffineExpr dimExpr =
      rewriter.getAffineDimExpr(0) - rewriter.getAffineConstantExpr(5);
  // AffineExpr dimExpr2 = rewriter
  // AffineMap map = AffineMap::get(1, 0, dimExpr);
  // AffineMap map = AffineMap::get(1, 0 , rewriter.getAffineDimExpr(0) - 5);
  IntegerSet set1 = IntegerSet::get(1, 0, {dimExpr}, {false});

  // affine.if %arg1 >= 0 and %5 <= %1 - 1
  //  n-k >= 0 && n-k <= len -1 //n = %arg0 , k = %arg1
  //  %arg0 >= 0 and %arg0 - %arg1 - %sym1 + 1 <= 0

  affine::AffineForOp forOp1 =
      rewriter.create<affine::AffineForOp>(loc, lb, ub, step);

  // inside the forOp body --> create the operations & then close the body
  //  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(forOp1.getBody());
  auto iv = forOp1.getInductionVar();
  // start adding operations like a arith::constant = 100.0 to the body of
  // forOp1
  //  Inside the loop body:

  // #set affine_set<(d0) : (d0 - 5 <= 0)>
  // affine.for %arg0 = 0 to 10 {
  //   %3 = affine.if #set (%arg0) {
  //         %1 = arith.const 25
  //         affine.yield %1
  //     }
  // else{
  //       %2 = arith.const 15
  //       affine.yield %2
  //   }
  //     affine.store %3, alloc[%arg0]
  // }

  // auto ifOp = rewriter.create<affine::AffineIfOp>( loc, set1 , ValueRange{iv}
  // , false /*no else*/ ); auto ifOp = rewriter.create<affine::AffineIfOp>(
  // loc, set1 , ValueRange{iv} , true /*no else*/ );

  // use typeRange too:
  Type floatType = rewriter.getF64Type();
  auto ifOp = rewriter.create<affine::AffineIfOp>(
      loc, TypeRange{floatType}, set1, ValueRange{iv}, true /*no else*/);

  rewriter.setInsertionPointToStart(ifOp.getThenBlock());

  FIRFilterResponseAdaptor firFilterOperands(operands);

  // load from the input
  Value loadInput =
      rewriter.create<AffineLoadOp>(loc, firFilterOperands.getLhs(), iv);
  Value constant25 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(25));
  Value constsq25 = rewriter.create<arith::MulFOp>(loc, loadInput, constant25);

  rewriter.create<AffineStoreOp>(loc, constsq25, alloc, iv);
  rewriter.create<AffineYieldOp>(loc, ValueRange{constsq25});
  // rewriter.setInsertionPointToEnd(ifOp.getThenBlock());

  rewriter.setInsertionPointToStart(ifOp.getElseBlock());
  Value loadInput2 =
      rewriter.create<AffineLoadOp>(loc, firFilterOperands.getRhs(), iv);
  Value constant15 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(15));
  Value elseResult =
      rewriter.create<arith::MulFOp>(loc, loadInput2, constant15);
  rewriter.create<AffineStoreOp>(loc, elseResult, alloc, iv);
  rewriter.create<AffineYieldOp>(loc, ValueRange{elseResult});
  // rewriter.setInsertionPointToEnd(ifOp.getElseBlock());
  rewriter.setInsertionPointAfter(ifOp);
  ifOp->dump();
  // forOp1->dump();
  rewriter.create<AffineStoreOp>(loc, ifOp.getResult(0), alloc, iv);
  // getParentBlock then use
  //  rewriter.setInsertionPointToEnd(ifOp.getThenBlock()->getParentOp());
  //  rewriter.setInsertionPointToEnd(ifOp->getBlock());
  //  rewriter.setInsertionPoint(ifOp->getParentOp());
  //  rewriter.create<AffineYieldOp>(loc, ValueRange{constant25});
  //  rewriter.setInsertionPointToEnd(ifOp.getThenBlock());

  // rewriter.setInsertionPointAfter(ifOp);
  // rewriter.create<AffineStoreOp>(loc, ifOp.getResult(0) , alloc, iv);

  // try to add the affine.If condition
  // create affine.If ,
  //  use integer set to represent the condition
  // check the AffineArgs
  //  affine.if operation contains two regions for the “then” and “else” clauses
  // each region of affine.if must contain a single block with no args and
  // terminated by affine.yield op
  //  if affine.if defines no values --> no need for affine.yield

  // affineIf.setConditional(set1, forOp1.getInductionVar());
  // start then "block"
  // "then" block

  // Value constant15 = rewriter.create<arith::ConstantOp>(loc,
  // rewriter.getF64Type(),
  //                                                      rewriter.getF64FloatAttr(15));

  //  rewriter.create<affine::AffineYieldOp>(loc, ValueRange{constant15});
  // rewriter.setInsertionPointToEnd(ifOp.getThenBlock());
  // else block
  // rewriter.setInsertionPointToStart(ifOp.getElseBlock());

  // Set insertion point to the end of the "then" block
  // rewriter.setInsertionPointAfter(ifOp.getThenBlock()->getTerminator());

  // rewriter.create<affine::AffineYieldOp>(loc, constant25);
  llvm::errs() << "LINE = " << __LINE__ << "\n";
  // Back to parentOp -- ifOp stops here
  //  rewriter.setInsertionPointAfter(ifOp);

  // also use affine::AffineStore to store at the loop induction variable
  //  auto storeOp = rewriter.create<affine::AffineStoreOp>(loc,
  //  ifOp.getResult(0), alloc, forOp1.getInductionVar()); auto storeOp =
  //  rewriter.create<affine::AffineStoreOp>(loc, constant25, alloc,
  //  forOp1.getInductionVar()); Back to parentOp -- forOp1
  //  rewriter.setInsertionPointAfter(storeOp);

  llvm::errs() << "LINE = " << __LINE__ << "  xx\n";
  // create affine yield for the loop
  //  rewriter.create<affine::AffineYieldOp>(loc);

#endif

#if TryAffineIf2

  int64_t lb = 0;
  int64_t ub = tensorType.getShape()[0];
  int64_t step = 1;

  // create AffineMap and set
  //  %1 = affine.load
  //   if ( %arg0 >= 5)   ie, integerSet <(d0) : (d0 - 5 >= 0) >
  AffineExpr dimExpr =
      rewriter.getAffineDimExpr(0) - rewriter.getAffineConstantExpr(5);
  // AffineExpr dimExpr2 = rewriter
  // AffineMap map = AffineMap::get(1, 0, dimExpr);
  // AffineMap map = AffineMap::get(1, 0 , rewriter.getAffineDimExpr(0) - 5);
  IntegerSet set1 = IntegerSet::get(1, 0, {dimExpr}, {false});

  // affine.if %arg1 >= 0 and %5 <= %1 - 1
  //  n-k >= 0 && n-k <= len -1 //n = %arg0 , k = %arg1
  //  %arg0 >= 0 and %arg0 - %arg1 - %sym1 + 1 <= 0

  affine::AffineForOp forOp1 =
      rewriter.create<affine::AffineForOp>(loc, lb, ub, step);

  // inside the forOp body --> create the operations & then close the body
  //  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(forOp1.getBody());
  auto iv = forOp1.getInductionVar();
  // start adding operations like a arith::constant = 100.0 to the body of
  // forOp1
  //  Inside the loop body:

  // #set affine_set<(d0) : (d0 - 5 <= 0)>
  // affine.for %arg0 = 0 to 10 {
  //   %3 = affine.if #set (%arg0) {
  //         %1 = arith.const 25
  //         affine.yield %1
  //     }
  //     affine.store %3, alloc[%arg0]
  // }

  // auto ifOp = rewriter.create<affine::AffineIfOp>( loc, set1 , ValueRange{iv}
  // , false /*no else*/ );
  auto ifOp = rewriter.create<affine::AffineIfOp>(loc, set1, ValueRange{iv},
                                                  true /*no else*/);
  rewriter.setInsertionPointToStart(ifOp.getThenBlock());
  // rewriter.setInsertionPointToEnd(ifOp.getThenBlock());
  Value constant25 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(25));
  Value constsq25 = rewriter.create<arith::MulFOp>(loc, constant25, constant25);

  // ifOp.setR
  // rewriter.create<AffineStoreOp>(loc, constant25 , alloc, iv);
  // rewriter.setInsertionPointToStart(ifOp.getElseBlock());
  Value constant15 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(15));
  rewriter.create<AffineStoreOp>(loc, constsq25, alloc, iv);

  // getParentBlock then use
  //  rewriter.setInsertionPointToEnd(ifOp.getThenBlock()->getParentOp());
  //  rewriter.setInsertionPointToEnd(ifOp->getBlock());
  rewriter.setInsertionPoint(ifOp->getParentOp());
  // rewriter.create<AffineYieldOp>(loc, ValueRange{constant25});
  // rewriter.setInsertionPointToEnd(ifOp.getThenBlock());

  // rewriter.setInsertionPointAfter(ifOp);
  // rewriter.create<AffineStoreOp>(loc, ifOp.getResult(0) , alloc, iv);
  // rewriter.cre

#endif

#if TryAffineMap
  int64_t lb = 0;
  int64_t ub = tensorType.getShape()[0] - 2;
  int64_t step = 1;

  affine::AffineForOp forOp1 =
      rewriter.create<affine::AffineForOp>(loc, lb, ub, step);

  // inside the forOp body --> create the operations & then close the body
  //  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(forOp1.getBody());
  auto iv = forOp1.getInductionVar();
  // start adding operations like a arith::constant = 100.0 to the body of
  // forOp1
  //  Inside the loop body:
  // create affine for
  // use affine-map expression for dimension then symbol then combination
  // affine-map expression for dimension: affine_map<d0, d1)[s0] -> (d0 , d1 +
  // s0, d1 - s0) use affine map Define an affine map: #map2 = affine_map<(d0)
  // -> (d0 + 2)>
  auto symbol1 = tensorType.getShape()[0];
  AffineExpr indx = rewriter.getAffineDimExpr(0);
  AffineExpr constantExpr = rewriter.getAffineConstantExpr(2);
  AffineMap addMap = AffineMap::get(1, 0, symbol1 - indx);
  auto outputIndex = rewriter.create<affine::AffineApplyOp>(loc, addMap, iv);

  // Value constant15 = rewriter.create<arith::ConstantOp>(loc,
  // rewriter.getF64Type(), rewriter.getF64FloatAttr(15));

  // try replace constant15 ie, with input & filter
  FIRFilterResponseOpAdaptor firOpAdaptor(operands);

  Value inputForFilter =
      rewriter.create<affine::AffineLoadOp>(loc, firOpAdaptor.getLhs(), iv);
  // Value inputForFilterMapped = rewriter.create<affine::AffineLoadOp>(loc,
  // firOpAdaptor.getLhs() , addMap, iv);

  Value impulseFilter =
      rewriter.create<AffineLoadOp>(loc, firOpAdaptor.getRhs(), iv);

  auto storeOp = rewriter.create<affine::AffineStoreOp>(
      loc, inputForFilter, alloc, ValueRange{outputIndex});

  llvm::errs() << "LINE = " << __LINE__ << "\n";

#endif

#if TrySumOfVector
  // here, we have to use iter
  int64_t lb = 0;
  int64_t ub = tensorType.getShape()[0];
  int64_t step = 1;

  Value constant0 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

  affine::AffineForOp forOp1 = rewriter.create<affine::AffineForOp>(
      loc, lb, ub, step, ValueRange{constant0});

  rewriter.setInsertionPointToStart(forOp1.getBody());
  auto iv = forOp1.getInductionVar();

  // inside the forOp body --> create the operations & then close the body
  //  OpBuilder::InsertionGuard guard(rewriter);
  //  Initial sum set to 0.
  //  %sum_0 = arith.constant 0.0 : f32
  //  // iter_args binds initial values to the loop's region arguments.
  //  %sum = affine.for %i = 0 to 10 step 1
  //      iter_args(%sum_iter = %sum_0) -> (f32) {
  //    %t = affine.load %buffer[%i] : memref<10xf32>
  //    %sum_next = arith.addf %sum_iter, %t : f32
  //    // Yield current iteration sum to next iteration %sum_iter or to %sum
  //    // if final iteration.
  //    affine.yield %sum_next : f32
  //  }
  //  return %sum : f32
  //  }

  // Inside the loop body:

  // try replace constant15 ie, with input & filter
  FIRFilterResponseOpAdaptor firOpAdaptor(operands);

  Value inputForFilter =
      rewriter.create<affine::AffineLoadOp>(loc, firOpAdaptor.getLhs(), iv);

  // Get iter_arg
  auto getIterArg =
      forOp1.getBody()->getArgument(1); // forOp1.getIterOperands();
  Value sumNext =
      rewriter.create<arith::AddFOp>(loc, inputForFilter, getIterArg);
  // Value sumNext = rewriter.create<arith::AddFOp>(loc, inputForFilter,
  // constant0);

  // here, at indx 0 , o/p = in[0]
  //  at indx 1 , o/p = in[0] + in[1] & so on
  // at indx last o/p[9] = sum of all input elements
  auto storeOp = rewriter.create<affine::AffineStoreOp>(loc, sumNext, alloc,
                                                        ValueRange{iv});
  rewriter.create<AffineYieldOp>(loc, ValueRange{sumNext});
  // rewriter.create<AffineYieldOp>(loc);
  // auto result = forOp1.getResult(0);
  llvm::errs() << "LINE = " << __LINE__ << "\n";

#endif

#if TryMultiDimLoop
  // here, we have to use iter
  int64_t lb = 0;
  int64_t ub = tensorType.getShape()[0];
  int64_t step = 1;

  Value constant0 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

  affine::AffineForOp forOp1 =
      rewriter.create<affine::AffineForOp>(loc, lb, ub, step);

  rewriter.setInsertionPointToStart(forOp1.getBody());
  auto iv = forOp1.getInductionVar();

  // create loadOp
  FIRFilterResponseOpAdaptor firOpAdaptor(operands);

  Value loadInput =
      rewriter.create<affine::AffineLoadOp>(loc, firOpAdaptor.getLhs(), iv);

  // create another loop --
  affine::AffineForOp forOp2 = rewriter.create<affine::AffineForOp>(
      loc, lb, ub, step, ValueRange{loadInput});

  rewriter.setInsertionPointToStart(forOp2.getBody());
  auto iv2 = forOp2.getInductionVar();
  Value loadFilter =
      rewriter.create<AffineLoadOp>(loc, firOpAdaptor.getRhs(), iv2);

  // get iterArg
  auto getIterArg = forOp2.getBody()->getArgument(1);
  auto sumNext = rewriter.create<arith::AddFOp>(loc, loadInput, loadFilter);

  // store the result to output
  //  rewriter.create<AffineStoreOp>(loc, sumNext, alloc, iv );
  rewriter.create<AffineYieldOp>(loc, ValueRange{sumNext});
  rewriter.setInsertionPointAfter(forOp2);
  rewriter.create<AffineStoreOp>(loc, forOp2.getResult(0), alloc, iv);
  //
  // yield the
  // inside the forOp body --> create the operations & then close the body
  // OpBuilder::InsertionGuard guard(rewriter);
  // Initial sum set to 0.
  // affine.for %arg0 = 0 to 10 {
  //   %1 = affine.load input[%arg0]
  //   %4 = affine.for %arg1 = 0 to 10 step 1
  //     iter_args(%sum_iter = %1) {
  //       %2 = affine.load filter[%arg1]
  //       %3 = arith.add sum_iter , %2
  //         affine.yield %3 : f64
  //   }
  //   affine.store %4, output[%arg0]
  // }

  // Inside the loop body:

  llvm::errs() << "LINE = " << __LINE__ << "\n";

#endif

#if TryMultiDimForAndIf
  int64_t lb = 0;
  int64_t ub = tensorType.getShape()[0];
  int64_t step = 1;

  // create AffineMap and set
  //  %1 = affine.load
  //   if ( %arg0 >= 5)   ie, integerSet <(d0) : (d0 - 5 >= 0) >

  // affine.if %arg1 >= 0 and %5 <= %1 - 1
  //  n-k >= 0 && n-k <= len -1 //n = %arg0 , k = %arg1
  //  %arg0 >= 0 and %arg0 - %arg1 - %sym1 + 1 <= 0

  affine::AffineForOp forOp1 =
      rewriter.create<affine::AffineForOp>(loc, lb, ub, step);

  // inside the forOp body --> create the operations & then close the body
  //  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPointToStart(forOp1.getBody());
  auto iv = forOp1.getInductionVar();
  // start adding operations like a arith::constant = 100.0 to the body of
  // forOp1
  //  Inside the loop body:

  AffineExpr dimExpr =
      rewriter.getAffineDimExpr(0) - rewriter.getAffineConstantExpr(5);
  IntegerSet set1 = IntegerSet::get(1, 0, {dimExpr}, {false});

  // create 2nd loop
  // use loop inductn variable for 2nd loop
  // use if condition on 2nd loop inductn variable
  // get the result of inner for loop and store at output

  affine::AffineForOp forOp2 =
      rewriter.create<affine::AffineForOp>(loc, lb, ub, step);
  rewriter.setInsertionPointToStart(forOp2.getBody());
  auto iv2 = forOp2.getInductionVar();
  AffineExpr dimExpr2 =
      rewriter.getAffineDimExpr(1) - rewriter.getAffineConstantExpr(6);
  IntegerSet set2 = IntegerSet::get(1, 0, {dimExpr, dimExpr2}, {false});

  auto ifOp = rewriter.create<affine::AffineIfOp>(loc, set2, ValueRange{iv},
                                                  false /*no else*/);
  rewriter.setInsertionPointToStart(ifOp.getThenBlock());
  Value constant25 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(25));
  Value resultFromInnerLoop =
      rewriter.create<arith::MulFOp>(loc, constant25, constant25);

  // rewriter.setInsertionPointAfter(forOp2);
  // rewriter.setInsertionPointToEnd(forOp2->getBlock());
  // rewriter.create<AffineStoreOp>(loc, constant25 , alloc, iv2);
  // rewriter.create<AffineYieldOp>(loc, ValueRange{resultFromInnerLoop});
  // rewriter.setInsertionPointAfter(ifOp);
  // rewriter.create<AffineYieldOp>(loc, ValueRange{resultFromInnerLoop});
  // rewriter.setInsertionPointAfter(forOp2);
  rewriter.create<AffineStoreOp>(loc, constant25, alloc, iv);
  // #set2 = affine_set<(d0, d1)[]: (d0 - 5 >= 0, d1- 5 >= 0 ) >
  // affine.for %arg0 = 0 to 10 {
  //     %N = len(output)
  //   %4 =  affine.for %arg1 = 0 to 10 {
  //         affine.if #set2(%arg0 , %arg1 )[%N] {
  //             %1 = const 5
  //             %2 = const 3
  //             %3 = arith.mulf %1 , %2
  //             affine.yield %3
  //         }
  //     }
  //   affine.store %4, alloc[%arg0]
  // }

  // rewriter.create<AffineYieldOp>(loc, ValueRange{constant25});
  // rewriter.setInsertionPointAfter(ifOp);
  // rewriter.create<AffineStoreOp>(loc, ifOp.getResult(0) , alloc, iv);

  // try to add the affine.If condition
  // create affine.If ,
  //  use integer set to represent the condition
  // check the AffineArgs
  //  affine.if operation contains two regions for the “then” and “else” clauses
  // each region of affine.if must contain a single block with no args and
  // terminated by affine.yield op
  //  if affine.if defines no values --> no need for affine.yield

  // affineIf.setConditional(set1, forOp1.getInductionVar());
  // start then "block"
  // "then" block

  // rewriter.create<affine::AffineYieldOp>(loc, constant25);
  llvm::errs() << "LINE = " << __LINE__ << "\n";
  // Back to parentOp -- ifOp stops here
  //  rewriter.setInsertionPointAfter(ifOp);

  llvm::errs() << "LINE = " << __LINE__ << "  xx\n";

#endif

#if TryMultiDimLoopAndAffineMap
  // here, we have to use iter
  int64_t lb = 0;
  int64_t ub = tensorType.getShape()[0];
  int64_t step = 1;

  Value constant0 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

  affine::AffineForOp forOp1 =
      rewriter.create<affine::AffineForOp>(loc, lb, ub, step);

  rewriter.setInsertionPointToStart(forOp1.getBody());
  auto iv = forOp1.getInductionVar();

  // create loadOp
  FIRFilterResponseOpAdaptor firOpAdaptor(operands);

  Value loadInput =
      rewriter.create<affine::AffineLoadOp>(loc, firOpAdaptor.getLhs(), iv);

  // create another loop --
  affine::AffineForOp forOp2 = rewriter.create<affine::AffineForOp>(
      loc, lb, ub, step, ValueRange{loadInput});

  rewriter.setInsertionPointToStart(forOp2.getBody());
  auto iv2 = forOp2.getInductionVar();

  // Use AffineMap for affine.load alloc_9[%arg0 - %arg1]
  AffineExpr OuterIndx = rewriter.getAffineDimExpr(0);
  AffineExpr InnerIndx = rewriter.getAffineDimExpr(1);
  AffineMap addMap = AffineMap::get(2, 0, OuterIndx - InnerIndx);
  // auto outputIndex = rewriter.create<affine::AffineApplyOp>(loc, addMap ,
  // ValueRange{iv,iv2});

  // Value constant15 = rewriter.create<arith::ConstantOp>(loc,
  // rewriter.getF64Type(), rewriter.getF64FloatAttr(15));

  // Value loadFilter = rewriter.create<AffineLoadOp>(loc, firOpAdaptor.getRhs()
  // , addMap, ValueRange{iv2,iv});
  Value loadFilter = rewriter.create<AffineLoadOp>(loc, firOpAdaptor.getRhs(),
                                                   addMap, ValueRange{iv, iv2});
  // Value loadFilter = rewriter.create<AffineLoadOp>(loc, firOpAdaptor.getRhs()
  // , outputIndex); get iterArg
  auto getIterArg = forOp2.getBody()->getArgument(1);
  auto sumNext = rewriter.create<arith::AddFOp>(loc, getIterArg, loadFilter);

  // store the result to output
  //  rewriter.create<AffineStoreOp>(loc, sumNext, alloc, iv );
  rewriter.create<AffineYieldOp>(loc, ValueRange{sumNext});
  rewriter.setInsertionPointAfter(forOp2);
  rewriter.create<AffineStoreOp>(loc, forOp2.getResult(0), alloc, iv);
  //
  // yield the
  // inside the forOp body --> create the operations & then close the body
  // OpBuilder::InsertionGuard guard(rewriter);
  // Initial sum set to 0.
  // affine.for %arg0 = 0 to 10 {
  //   %1 = affine.load input[%arg0]
  //   %4 = affine.for %arg1 = 0 to 10 step 1
  //     iter_args(%sum_iter = %1) {
  //       %2 = affine.load filter[%arg1]
  //       %3 = arith.add sum_iter , %2
  //         affine.yield %3 : f64
  //   }
  //   affine.store %4, output[%arg0]
  // }

  // Inside the loop body:

  llvm::errs() << "LINE = " << __LINE__ << "\n";

#endif

#if TryMultiDimLoopAndAffineSet
  // here, we have to use iter
  int64_t lb = 0;
  int64_t ub = tensorType.getShape()[0];
  int64_t step = 1;

  Value constant0 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

  affine::AffineForOp forOp1 =
      rewriter.create<affine::AffineForOp>(loc, lb, ub, step);

  rewriter.setInsertionPointToStart(forOp1.getBody());
  auto iv = forOp1.getInductionVar();

  // create loadOp
  FIRFilterResponseOpAdaptor firOpAdaptor(operands);

  Value loadInput =
      rewriter.create<affine::AffineLoadOp>(loc, firOpAdaptor.getLhs(), iv);

  // create another loop --
  affine::AffineForOp forOp2 = rewriter.create<affine::AffineForOp>(
      loc, lb, ub, step, ValueRange{loadInput});

  rewriter.setInsertionPointToStart(forOp2.getBody());
  auto iv2 = forOp2.getInductionVar();

  // Use AffineMap for affine.load alloc_9[%arg0 - %arg1]
  AffineExpr OuterIndx = rewriter.getAffineDimExpr(0);
  AffineExpr InnerIndx = rewriter.getAffineDimExpr(1);
  AffineMap addMap = AffineMap::get(2, 0, OuterIndx - InnerIndx);
  auto outputIndex =
      rewriter.create<affine::AffineApplyOp>(loc, addMap, ValueRange{iv, iv2});

  // Value constant15 = rewriter.create<arith::ConstantOp>(loc,
  // rewriter.getF64Type(), rewriter.getF64FloatAttr(15));
  AffineExpr dimExpr = OuterIndx - InnerIndx;
  IntegerSet set1 = IntegerSet::get(2, 0, {dimExpr}, {false});

  auto ifOp = rewriter.create<affine::AffineIfOp>(
      loc, set1, ValueRange{iv, iv2}, false /*no else*/);
  rewriter.setInsertionPointToStart(ifOp.getThenBlock());
  // Value loadFilter = rewriter.create<AffineLoadOp>(loc, firOpAdaptor.getRhs()
  // , addMap, ValueRange{iv2,iv});
  Value loadFilter = rewriter.create<AffineLoadOp>(loc, firOpAdaptor.getRhs(),
                                                   addMap, ValueRange{iv, iv2});
  // get iterArg
  auto getIterArg = forOp2.getBody()->getArgument(1);
  auto sumNext = rewriter.create<arith::AddFOp>(loc, loadFilter, loadFilter);
  // rewriter.create<AffineStoreOp>(loc, sumNext, alloc, iv );
  rewriter.create<AffineYieldOp>(loc, ValueRange{sumNext});

  // store the result to output
  //  rewriter.create<AffineStoreOp>(loc, sumNext, alloc, iv );
  rewriter.setInsertionPointAfter(ifOp);
  rewriter.create<AffineYieldOp>(loc, ValueRange{sumNext});
  rewriter.setInsertionPointAfter(forOp2);
  rewriter.create<AffineStoreOp>(loc, forOp2.getResult(0), alloc, iv);
  //
  // yield the
  // inside the forOp body --> create the operations & then close the body
  // OpBuilder::InsertionGuard guard(rewriter);
  // Initial sum set to 0.
  // affine.for %arg0 = 0 to 10 {
  //   %1 = affine.load input[%arg0]
  //   %4 = affine.for %arg1 = 0 to 10 step 1
  //     iter_args(%sum_iter = %1) {
  //       %2 = affine.load filter[%arg1]
  //       %3 = arith.add sum_iter , %2
  //         affine.yield %3 : f64
  //   }
  //   affine.store %4, output[%arg0]
  // }

  // Inside the loop body:

  llvm::errs() << "LINE = " << __LINE__ << "\n";

#endif

#if TryFIRFilter

  int64_t lb = 0;
  int64_t ub = tensorType.getShape()[0];
  int64_t step = 1;

  affine::AffineForOp forOp1 =
      rewriter.create<affine::AffineForOp>(loc, lb, ub, step);
  rewriter.setInsertionPointToStart(forOp1.getBody());
  auto iv = forOp1.getInductionVar();

  // Value sum0 = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(),
  //                                             rewriter.getF64FloatAttr(0));
  // get filter len
  // auto tensorTypeFilter = llvm::cast<RankedTensorType>((*op->getOperand(1)));
  // //operand_type_end auto tensorTypeFilter =
  // llvm::cast<RankedTensorType>((*op->operand_type_begin()));
  auto operandIt = op->operand_type_begin();
  auto tensorTypeInput = llvm::cast<RankedTensorType>(*operandIt);
  int64_t ubForInput = tensorTypeInput.getShape()[0];
  // get second operand
  operandIt = operandIt + 1;

  // auto tensorTypeFilter =
  // llvm::cast<RankedTensorType>((*op->operand_type_begin())); //operandIt
  auto tensorTypeFilter = llvm::cast<RankedTensorType>(*operandIt);
  int64_t ubForFilter = tensorTypeFilter.getShape()[0];

  // llvm::errs() << "ubForFilter= " << ubForFilter << "\n";
  // create a constant for sum
  Value constant0 = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
  affine::AffineForOp forOp2 = rewriter.create<affine::AffineForOp>(
      loc, lb, ubForFilter, step, ValueRange{constant0});
  rewriter.setInsertionPointToStart(forOp2.getBody());
  auto iv2 = forOp2.getInductionVar();

  auto getIterArg =
      forOp2.getBody()->getArgument(1); // forOp1.getIterOperands();

  // AffineExpr dimExpr = rewriter.getAffineDimExpr(0);
  AffineExpr dimExpr2 =
      rewriter.getAffineDimExpr(0) - rewriter.getAffineDimExpr(1);
  // n-k <= inputLen -1 or, k-n >= 1 - inputLen ie, k - n + inputLen - 1 >= 0
  AffineExpr ExprForUpperBoundCheck =
      rewriter.getAffineConstantExpr(ubForInput) +
      rewriter.getAffineDimExpr(1) - rewriter.getAffineDimExpr(0) -
      rewriter.getAffineConstantExpr(1);
  IntegerSet set2 =
      IntegerSet::get(2, 0, {dimExpr2, ExprForUpperBoundCheck}, {false, false});

  // use typeRange too:
  Type floatType = rewriter.getF64Type();
  //  if n-k >= 0
  auto ifOp = rewriter.create<affine::AffineIfOp>(
      loc, TypeRange{floatType}, set2, ValueRange{iv, iv2}, true /*else*/);
  rewriter.setInsertionPointToStart(ifOp.getThenBlock());

  AffineMap addMap = AffineMap::get(2, 0, dimExpr2);
  // auto inputIndex = rewriter.create<affine::AffineApplyOp>(loc, addMap ,
  // ValueRange{iv,iv2});

  FIRFilterResponseOpAdaptor firOpAdaptor(operands);
  Value loadInput = rewriter.create<AffineLoadOp>(loc, firOpAdaptor.getLhs(),
                                                  addMap, ValueRange{iv, iv2});

  rewriter.create<AffineYieldOp>(loc, ValueRange{loadInput});
  // else block
  rewriter.setInsertionPointToStart(ifOp.getElseBlock());
  Value const0ForElse = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
  rewriter.create<AffineYieldOp>(loc, ValueRange{const0ForElse});
  rewriter.setInsertionPointAfter(ifOp);

  // load filter and then mult and then sum
  Value loadFilter =
      rewriter.create<affine::AffineLoadOp>(loc, firOpAdaptor.getRhs(), iv2);
  // Value constant25 = rewriter.create<arith::ConstantOp>(loc,
  // rewriter.getF64Type(),
  //                                                      rewriter.getF64FloatAttr(25));
  Value filterMulInput =
      rewriter.create<arith::MulFOp>(loc, ifOp.getResult(0), loadFilter);
  Value sumNext =
      rewriter.create<arith::AddFOp>(loc, filterMulInput, getIterArg);
  rewriter.create<AffineYieldOp>(loc, ValueRange{sumNext});
  // rewriter.setInsertionPointToEnd(forOp2->getBlock());
  rewriter.setInsertionPointAfter(forOp2);
  rewriter.create<AffineStoreOp>(loc, forOp2.getResult(0), alloc, iv);
  rewriter.setInsertionPointAfter(forOp1);

  // ifOp->dump();

  // FIRFilterResponse code -- x[n] , h[n]

  // iterate for output
  // start with sum=0
  // iterate for filter len
  // check for input_indx must be within bounds
  // load filter and input[indx]
  // multiply them
  // add this to sum
  // update output with sum

  // inside the forOp body --> create the operations & then close the body
  //  OpBuilder::InsertionGuard guard(rewriter);

  // start adding operations like a arith::constant = 100.0 to the body of
  // forOp1
  //  Inside the loop body:

  // #set2 = affine_set<(d0, d1)[]: (d0 - 5 >= 0, d1- 5 >= 0 ) >
  // affine.for %arg0 = 0 to 10 {
  //     %N = len(output)
  //   %4 =  affine.for %arg1 = 0 to 10 {
  //         affine.if #set2(%arg0 , %arg1 )[%N] {
  //             %1 = const 5
  //             %2 = const 3
  //             %3 = arith.mulf %1 , %2
  //             affine.yield %3
  //         }
  //     }
  //   affine.store %4, alloc[%arg0]
  // }

  // rewriter.create<AffineYieldOp>(loc, ValueRange{constant25});
  // rewriter.setInsertionPointAfter(ifOp);
  // rewriter.create<AffineStoreOp>(loc, ifOp.getResult(0) , alloc, iv);

  // try to add the affine.If condition
  // create affine.If ,
  //  use integer set to represent the condition
  // check the AffineArgs
  //  affine.if operation contains two regions for the “then” and “else” clauses
  // each region of affine.if must contain a single block with no args and
  // terminated by affine.yield op
  //  if affine.if defines no values --> no need for affine.yield

  // affineIf.setConditional(set1, forOp1.getInductionVar());
  // start then "block"
  // "then" block

  // rewriter.create<affine::AffineYieldOp>(loc, constant25);
  // llvm::errs() << "LINE = " << __LINE__ << "\n";
  // Back to parentOp -- ifOp stops here
  // rewriter.setInsertionPointAfter(ifOp);

  // llvm::errs() << "LINE = " << __LINE__ << "  xx\n";

#endif
  // Terminate the loop body with affine.yield.
  // rewriter.create<affine::AffineYieldOp>(loc);

  // Replace this operation with the generated alloc.
  rewriter.replaceOp(op, alloc);
}

namespace {

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: FFT1DImg operations
//===----------------------------------------------------------------------===//

struct FFT1DImgConjSymmOpLowering : public ConversionPattern {
  FFT1DImgConjSymmOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::FFT1DImgConjSymmOp::getOperationName(), 1, ctx) {
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //   y[k] = y_real[k] + j *y_img[k]
    //  y_img = sumOver_n(x[n]*sin[2*pi * k *n/N ] * -1

    // For k=0:
    // y[0] = 0

    // for k=1 to (N+1)/2
    // sum = 0
    // for n=0 to N
    // sum = sum + x[n] * sin(2*pi*k*n/N)
    // y[k] = -1 * sum
    // y[N-k] = sum
    // init  output mem for y_real & y_img as 0
    // iterate for output from k=0 to last
    // iterate for all x from n=0 to last
    // perform the calculations : ie x[n] * cos[2*pi * k *n/N ] and sum and
    // store them at y[k]
    //
    // replace this upsampling op with the output_mem_allocation op

    DEBUG_PRINT_NO_ARGS();

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    // auto memRefType2 = convertTensorToMemRef(tensorType1);
    auto alloc_img = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // affine.for %y = 0 to 4 {
    //     affine.store %cst_3, %alloc_real[%y] : memref<4xf64>
    //     affine.store %cst_3, %alloc_img[%y] : memref<4xf64>
    // }
    Value constant0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

    // For loop -- iterate from 1 to last
    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t ubBy2 = (ub + 1) / 2;
    int64_t step = 1;

    // affine::AffineForOp forOp1 = rewriter.create<AffineForOp>(loc, lb, ub,
    // step); auto iv = forOp1.getInductionVar();
    // rewriter.setInsertionPointToStart(forOp1.getBody());
    // rewriter.create<AffineStoreOp>(loc, constant0, alloc_img,
    // ValueRange{iv}); rewriter.setInsertionPointAfter(forOp1);
    DEBUG_PRINT_NO_ARGS();
    // for k=0
    Value Indx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    rewriter.create<AffineStoreOp>(loc, constant0, alloc_img,
                                   ValueRange{Indx0});

    // loop for Y
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb + 1, ubBy2, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());

    // loop for X
    affine::AffineForOp forOpX =
        rewriter.create<AffineForOp>(loc, lb, ub, step, ValueRange{constant0});
    auto ivX = forOpX.getInductionVar();
    auto getIterArg = forOpX.getBody()->getArgument(1);
    rewriter.setInsertionPointToStart(forOpX.getBody());

    // load from X, & y1 & y2
    FFT1DImgConjSymmOpAdaptor fft1DImgConjSymmAdaptor(operands);
    Value inputX = rewriter.create<AffineLoadOp>(
        loc, fft1DImgConjSymmAdaptor.getInput(), ValueRange{ivX});
    // Value loadYImg = rewriter.create<AffineLoadOp>(loc, alloc_img,
    // ValueRange{ivY});

    // convert index to f64
    Value IndxY = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivY);
    Value k =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxY);

    Value IndxX = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivX);
    Value i =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxX);

    // get 2*pi * k * i / N
    Value muli_k = rewriter.create<arith::MulFOp>(loc, k, i);

    Value const2pi = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(6.28318530718));
    Value mul2piKI = rewriter.create<arith::MulFOp>(loc, const2pi, muli_k);

    // getOperand().getType()
    // auto inputTensorType =
    // llvm::cast<RankedTensorType>(op->getOperand(0).getType());
    float LengthOfInput = (float)ub;
    Value N = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(LengthOfInput));
    // Value N = inputTensorType.getShape()[0];

    Value divIndxByN = rewriter.create<arith::DivFOp>(loc, mul2piKI, N);

    // Img part = -1 * Sum(x[i] * sin(div) )
    Value GetSin = rewriter.create<math::SinOp>(loc, divIndxByN);
    Value xMulSin = rewriter.create<arith::MulFOp>(loc, inputX, GetSin);
    Value imgSum = rewriter.create<arith::SubFOp>(loc, getIterArg, xMulSin);

    rewriter.create<AffineYieldOp>(loc, ValueRange{imgSum});
    rewriter.setInsertionPointAfter(forOpX);

    // store imgSum at y[k]
    rewriter.create<AffineStoreOp>(loc, forOpX.getResult(0), alloc_img,
                                   ValueRange{ivY});

    // store -1 * imgSum at y[N-k]
    AffineExpr ExprNminusK =
        rewriter.getAffineConstantExpr(ub) - rewriter.getAffineDimExpr(0);
    AffineMap mapNminusK = AffineMap::get(1, 0, ExprNminusK);
    Value constMinus1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(-1));
    Value NegImgSum =
        rewriter.create<arith::MulFOp>(loc, constMinus1, forOpX.getResult(0));

    rewriter.create<AffineStoreOp>(loc, NegImgSum, alloc_img, mapNminusK,
                                   ValueRange{ivY});

    rewriter.setInsertionPointAfter(forOpY);
    // debug
    //  forOpX->dump();
    //  forOpY->dump();
    //  affine.for %y = 0 to 4 {
    //      affine.store %cst_3, %alloc_real[%y] : memref<4xf64>
    //      affine.store %cst_3, %alloc_img[%y] : memref<4xf64>
    //  }

    // affine.for %y = 0 to 4 {
    // //   %0 = affine.load %alloc_3[%arg0] : memref<4xf64>
    // //   affine.store %0, %alloc_real[%arg0] : memref<4xf64>
    // affine.for %x = 0 to 4 {
    //     // CAcluations
    //           %1 = affine.load %alloc_3[%x] : memref<4xf64>
    //           %2 = affine.load %alloc_real[%y] : memref<4xf64>
    //           %3 = affine.load %alloc_img[%y] : memref<4xf64>
    //           // index cast for multiply
    //           %4 = arith.index_castui %y : index to i32
    //           %k = arith.uitofp %4 : i32 to f64
    //           %6 = arith.index_castui %x : index to i32
    //           %i = arith.uitofp %6 : i32 to f64
    //         //   %8 = arith.index_castui %arg3 : index to i32
    //         //   %9 = arith.uitofp %8 : i32 to f64
    //         //   %10 = arith.index_castui %arg4 : index to i32
    //         //   %11 = arith.uitofp %10 : i32 to f64

    //           %mul_1 = arith.mulf %i, %k : f64
    //           %mul = arith.mulf %mul_1, %cst_2pi : f64
    //         //  ixk / N
    //           %div = arith.divf %mul, %N : f64
    //         //   cos of the above
    //           %res_cos = math.cos %div : f64
    //         //   %16 = arith.addf %14, %15 : f64
    //         //   %res_sin = arith.mulf %16, %cst_0 : f64

    //           %res_sin = math.sin %div : f64
    //           %real_prod = arith.mulf %1, %res_cos : f64
    //           %img_prod_1 = arith.mulf %1, %res_sin : f64
    //           %img_prod = arith.mulf %cst_5, %img_prod_1 : f64

    //           %real = arith.addf %2, %real_prod : f64
    //           %img = arith.addf %3, %img_prod : f64
    //           affine.store %real, %alloc_real[%y] : memref<4xf64>
    //         //    dsp.print %alloc_real : memref<4xf64>
    //           affine.store %img, %alloc_img[%y] : memref<4xf64>

    // }
    // }
    // rewriter.replaceOp(op, alloc_real);
    rewriter.replaceOp(op, alloc_img);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: FFT1DRealSymmOp operations
//===----------------------------------------------------------------------===//

struct FFT1DRealSymmOpLowering : public ConversionPattern {
  FFT1DRealSymmOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::FFT1DRealSymmOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //  y[k] = sumOver_n(x[n]*cos[2*pi * k *n/N ] , 0<=k < (N+1)/2
    //         & y[N-k] = y[k]  (N+1)/2<= k< N
    //  For k=0:
    // sum=0
    //  for n= 0 to N
    // sum = sum + x[n]
    // y[0] = sum

    // for k=1 to (N+1)/2
    // sum = 0
    // for n=0 to N
    // sum = sum + x[n] * cos(2*pi*k*n/N)
    // y[k] = sum
    // y[N-k] = sum

    // Actual definition
    //   y[k] = y_real[k] + j *y_img[k]
    //  y_real = sumOver_n(x[n]*cos[2*pi * k *n/N ]
    //  y_img = sumOver_n(x[n]*sin[2*pi * k *n/N ] * -1
    // init  output mem for y_real & y_img as 0
    //  replace this upsampling op with the output_mem_allocation op

    // DEBUG_PRINT_NO_ARGS() ;

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    // iterate to result1 --not needed for now but for future reference
    //  auto tensorType1 =
    //  llvm::cast<RankedTensorType>(*std::next(op->result_type_begin(), 1));

    // DEBUG_PRINT_NO_ARGS() ;
    // tensorType.getShape()[0]
    // llvm::errs() << "tensorType1.getShape()[0] " << tensorType1.getShape()[0]
    // << " func= " << __func__ << "\n";

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    // auto memRefType2 = convertTensorToMemRef(tensorType1);
    auto alloc_real = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // affine.for %y = 0 to 4 {
    //     affine.store %cst_3, %alloc_real[%y] : memref<4xf64>
    //     affine.store %cst_3, %alloc_img[%y] : memref<4xf64>
    // }
    Value constant0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

    // For loop -- iterate from 1 to last
    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t ubBy2 = (ub + 1) / 2;
    int64_t step = 1;

    // load from X, & y1 & y2
    FFT1DRealSymmOpAdaptor fft1DRealSymmAdaptor(operands);

    // affine::AffineForOp forOp1 = rewriter.create<AffineForOp>(loc, lb, ub,
    // step); auto iv = forOp1.getInductionVar();
    // rewriter.setInsertionPointToStart(forOp1.getBody());
    // rewriter.create<AffineStoreOp>(loc, constant0, alloc_real,
    // ValueRange{iv}); rewriter.setInsertionPointAfter(forOp1);

    // k=0
    // sum=0
    //  for n= 0 to N
    // sum = sum + x[n]
    // y[0] = sum
    affine::AffineForOp forOp2 = rewriter.create<affine::AffineForOp>(
        loc, lb, ub, step, ValueRange{constant0});
    auto iv2 = forOp2.getInductionVar();
    rewriter.setInsertionPointToStart(forOp2.getBody());
    // get previous sum
    auto getIterArg1 = forOp2.getBody()->getArgument(1);
    Value loadX = rewriter.create<AffineLoadOp>(
        loc, fft1DRealSymmAdaptor.getInput(), ValueRange{iv2});
    Value sumNext1 = rewriter.create<arith::AddFOp>(loc, loadX, getIterArg1);
    rewriter.create<AffineYieldOp>(loc, ValueRange{sumNext1});
    rewriter.setInsertionPointAfter(forOp2);

    // store result for k=0
    Value Indx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    rewriter.create<AffineStoreOp>(loc, forOp2.getResult(0), alloc_real,
                                   ValueRange{Indx0});

    // for k=1 to (N+1)/2
    // sum = 0
    // for n=0 to N
    // sum = sum + x[n] * cos(2*pi*k*n/N)
    // y[k] = sum
    // y[N-k] = sum
    // loop for Y ie, k
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb + 1, ubBy2, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());

    // loop for X
    affine::AffineForOp forOpX =
        rewriter.create<AffineForOp>(loc, lb, ub, step, ValueRange{constant0});
    auto ivX = forOpX.getInductionVar();
    // get sum
    auto getIterArg = forOpX.getBody()->getArgument(1);
    rewriter.setInsertionPointToStart(forOpX.getBody());

    // load from X, & y1 & y2
    Value inputX = rewriter.create<AffineLoadOp>(
        loc, fft1DRealSymmAdaptor.getInput(), ValueRange{ivX});
    // Value loadYReal = rewriter.create<AffineLoadOp>(loc, alloc_real,
    // ValueRange{ivY});

    // convert index to f64
    Value IndxY = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivY);
    Value k =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxY);

    Value IndxX = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivX);
    Value i =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxX);

    // get 2*pi * k * i / N
    Value muli_k = rewriter.create<arith::MulFOp>(loc, k, i);

    Value const2pi = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(6.28318530718));
    Value mul2piKI = rewriter.create<arith::MulFOp>(loc, const2pi, muli_k);

    float LengthOfInput = (float)ub;
    Value N = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(LengthOfInput));
    // Value N = inputTensorType.getShape()[0];

    Value divIndxByN = rewriter.create<arith::DivFOp>(loc, mul2piKI, N);

    // Real part = Sum(x[i] * cos(div) )
    Value GetCos = rewriter.create<math::CosOp>(loc, divIndxByN);
    Value xMulCos = rewriter.create<arith::MulFOp>(loc, inputX, GetCos);

    // realSu
    Value sumNext = rewriter.create<arith::AddFOp>(loc, getIterArg, xMulCos);
    // rewriter.create<AffineStoreOp>(loc, sumNext, alloc_real,
    // ValueRange{ivX});

    // DEBUG_PRINT_NO_ARGS() ;
    rewriter.create<AffineYieldOp>(loc, ValueRange{sumNext});
    rewriter.setInsertionPointAfter(forOpX);
    // forOpX->dump();
    // store realSum at y[k]
    rewriter.create<AffineStoreOp>(loc, forOpX.getResult(0), alloc_real,
                                   ValueRange{ivY});

    // store realSum at y[N-k]
    AffineExpr ExprNminusK =
        rewriter.getAffineConstantExpr(ub) - rewriter.getAffineDimExpr(0);
    AffineMap mapNminusK = AffineMap::get(1, 0, ExprNminusK);
    rewriter.create<AffineStoreOp>(loc, forOpX.getResult(0), alloc_real,
                                   mapNminusK, ValueRange{ivY});

    rewriter.setInsertionPointAfter(forOpY);
    // debug
    //  forOpX->dump();
    //  forOpY->dump();
    rewriter.replaceOp(op, alloc_real);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: FIRFilterYSymmOptimizedOp operations
//===----------------------------------------------------------------------===//
struct FIRFilterYSymmOptimizedOpLowering : public ConversionPattern {
  FIRFilterYSymmOptimizedOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::FIRFilterYSymmOptimizedOp::getOperationName(), 1,
                          ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // dsp.FIRFilterYSymmOptimizedOp has 2 operands -- both of type tensor f64

    // Get the location of FIRFilterYSymmOptimizedOp
    auto loc = op->getLoc();

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
    // Pseudo-code:
    // N=lenY , M=lenX here, output is symm ie, y[n] = y[N-1-n]
    // y[n] = x[n] conv x[-n] ie, x[M-1-n] ie, x2[n]
    // y[n] = SumOverAllk x[k] * x2[n-k]  , 0<=k<M  , 0<=n<N
    //      = SumOverAllk x[k] * x[M-1-(n-k)] , check for 0<=M+k-1-n<M

    // code:
    // for n=0 to (N+1)/2
    //  sum =0
    //  for k=0 to M
    //  if( 0<= M+k-n-1 <M)
    //  sum = sum + x[k] * x[M+k-n-1]
    // return sum
    // y[n]= sum
    // y[N-1-n] = sum

    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int ubBy2 = (ub + 1) / 2;
    int64_t step = 1;
    DEBUG_PRINT_NO_ARGS();
    affine::AffineForOp forOp1 =
        rewriter.create<affine::AffineForOp>(loc, lb, ubBy2, step);
    rewriter.setInsertionPointToStart(forOp1.getBody());
    auto iv = forOp1.getInductionVar();

    // for n=0 to N
    //  sum = 0, temp =0
    // for n=0 to (N+1)/2
    //  sum =0
    // get filter len
    auto operandIt = op->operand_type_begin();
    auto tensorTypeInput = llvm::cast<RankedTensorType>(*operandIt);
    int64_t ubForInput = tensorTypeInput.getShape()[0];
    DEBUG_PRINT_NO_ARGS();
    DEBUG_PRINT_WITH_ARGS("ubForInput=", ubForInput);

    // create a constant for sum
    Value constant0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
    affine::AffineForOp forOp2 = rewriter.create<affine::AffineForOp>(
        loc, lb, ubForInput, step, ValueRange{constant0});
    rewriter.setInsertionPointToStart(forOp2.getBody());
    auto iv2 = forOp2.getInductionVar();
    // get sum
    auto getIterArg = forOp2.getBody()->getArgument(1);
    DEBUG_PRINT_NO_ARGS();
    FIRFilterYSymmOptimizedOpAdaptor firFilterYSymmOpAdaptor(operands);

    // if( 0<= M+k-n-1 <M)
    // sum = sum + x[k] * x[M+k-n-1]
    // For M+k-n-1
    // LowerBoundSet: M+k-n-1 >=0  ie, 2 dimensions =n & k
    // UpperBoundSet: M+k-n-1 <= M-1 ie, n-k>=0

    // LowerBound Expr: M+k-n-1 >=0 ie, M-1 + k -n >= 0
    AffineExpr ExprLowerBound = rewriter.getAffineConstantExpr(ubForInput - 1) +
                                rewriter.getAffineDimExpr(1) -
                                rewriter.getAffineDimExpr(0);
    // UpperBoundSet: M+k-n-1 <= M-1 ie, n-k>=0
    AffineExpr ExprUpperBound =
        rewriter.getAffineDimExpr(0) - rewriter.getAffineDimExpr(1);
    IntegerSet setForIf =
        IntegerSet::get(2, 0, {ExprLowerBound, ExprUpperBound}, {false, false});
    DEBUG_PRINT_NO_ARGS();

    // if( 0<= M+k-n-1 <M)
    Type floatType = rewriter.getF64Type();
    auto ifOp =
        rewriter.create<affine::AffineIfOp>(loc, TypeRange{floatType}, setForIf,
                                            ValueRange{iv, iv2}, true /*else*/);
    rewriter.setInsertionPointToStart(ifOp.getThenBlock());
    DEBUG_PRINT_NO_ARGS();

    // sum = sum + x[k] * x[M+k-n-1]
    // load x[M+k-n-1]
    AffineMap mapMPlusKMinusNmin1 = AffineMap::get(2, 0, ExprLowerBound);
    Value loadInputIndx2 =
        rewriter.create<AffineLoadOp>(loc, firFilterYSymmOpAdaptor.getLhs(),
                                      mapMPlusKMinusNmin1, ValueRange{iv, iv2});
    rewriter.create<AffineYieldOp>(loc, ValueRange{loadInputIndx2});

    // else return 0
    rewriter.setInsertionPointToStart(ifOp.getElseBlock());
    Value const0ForElse = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
    rewriter.create<AffineYieldOp>(loc, ValueRange{const0ForElse});
    rewriter.setInsertionPointAfter(ifOp);

    // outside if
    // Now, sum = sum + val2 * x[k]
    Value loadX = rewriter.create<AffineLoadOp>(
        loc, firFilterYSymmOpAdaptor.getLhs(), ValueRange{iv2});
    DEBUG_PRINT_NO_ARGS();

    // x[k] * x[M+k-n-1]   here, val2 = x[M+k-n-1]
    Value XMulReverseXIndx =
        rewriter.create<arith::MulFOp>(loc, loadX, ifOp.getResult(0));
    // sum = sum + x[k] * x[M+k-n-1]
    Value sumNext =
        rewriter.create<arith::AddFOp>(loc, XMulReverseXIndx, getIterArg);
    rewriter.create<AffineYieldOp>(loc, ValueRange{sumNext});

    DEBUG_PRINT_NO_ARGS();
    rewriter.setInsertionPointAfter(forOp2);
    // forOp2->dump();
    DEBUG_PRINT_NO_ARGS();

    // y[n] = sum ie, y[n] = sumNext
    rewriter.create<AffineStoreOp>(loc, forOp2.getResult(0), alloc, iv);
    // y[N-1-n] = sum
    AffineExpr ExprNminus1minYn =
        rewriter.getAffineConstantExpr(ub - 1) - rewriter.getAffineDimExpr(0);
    AffineMap mapNminus1minYn = AffineMap::get(1, 0, ExprNminus1minYn);

    rewriter.create<AffineStoreOp>(loc, forOp2.getResult(0), alloc,
                                   mapNminus1minYn, ValueRange{iv});
    rewriter.setInsertionPointAfter(forOp1);
    DEBUG_PRINT_NO_ARGS();

    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: PaddingOp operations
//===----------------------------------------------------------------------===//

struct PaddingOpLowering : public ConversionPattern {
  PaddingOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::PaddingOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //   y[n] = x[n]  0<=n<N
    //   y[n] = val  N<=n < N+len
    // ie,
    // for i=0 to N --inputLen
    // y[n] = x[n]
    // for i=N to N+len
    // y[n] = val

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    DEBUG_PRINT_NO_ARGS();
    // construct affine loops for the input
    PaddingOpAdaptor paddingOpAdaptor(operands);
    Value GetPadLenOperand = op->getOperand(2);
    dsp::ConstantOp constantOp3rdArg =
        GetPadLenOperand.getDefiningOp<dsp::ConstantOp>();

    if (!constantOp3rdArg) {
      llvm::errs() << "Fail:padding op 3rd operand is not constant\n";
      return failure();
    }
    DenseElementsAttr constant3rdValue = constantOp3rdArg.getValue();
    ;
    auto elements1 = constant3rdValue.getValues<FloatAttr>();
    float Padlen = elements1[0].getValueAsDouble();
    DEBUG_PRINT_WITH_ARGS("Padlen is", Padlen);
    // first from 0 <= i < N
    auto inputType =
        llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    int64_t lb = 0;
    int64_t ub = inputType.getShape()[0];
    int64_t step = 1;

    DEBUG_PRINT_NO_ARGS();

    // loop from 0 <= i < N
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());
    Value InputX =
        rewriter.create<AffineLoadOp>(loc, paddingOpAdaptor.getInput(), ivY);
    rewriter.create<AffineStoreOp>(loc, InputX, alloc, ivY);
    rewriter.setInsertionPointAfter(forOpY);

    // loop from N to N+PadLen
    int64_t lb2 = ub;
    int64_t ub2 = ub + (int64_t)Padlen;

    affine::AffineForOp forOp2 =
        rewriter.create<AffineForOp>(loc, lb2, ub2, step);
    auto iv2 = forOp2.getInductionVar();
    rewriter.setInsertionPointToStart(forOp2.getBody());
    Value PaddingValue = rewriter.create<AffineLoadOp>(
        loc, paddingOpAdaptor.getPadValue(), ValueRange{}); // getPadValue
    rewriter.create<AffineStoreOp>(loc, PaddingValue, alloc, iv2);
    rewriter.setInsertionPointAfter(forOp2);

    // debug
    //  forOpX->dump();
    //  forOpY->dump();

    // %cst = arith.constant 6.2831853071800001 : f64
    // %cst_0 = arith.constant 4.600000e-01 : f64
    // %cst_1 = arith.constant 5.400000e-01 : f64
    // %cst_2 = arith.constant 4.000000e+00 : f64
    // %alloc = memref.alloc() : memref<4xf64>
    // %alloc_3 = memref.alloc() : memref<f64>
    // affine.store %cst_2, %alloc_3[] : memref<f64>
    // affine.for %arg0 = 0 to 4 {
    //   %0 = arith.index_castui %arg0 : index to i32
    //   %1 = arith.uitofp %0 : i32 to f64
    //   %2 = arith.mulf %1, %cst : f64
    //   %3 = arith.divf %2, %cst_2 : f64
    //   %4 = math.cos %3 : f64
    //   %5 = arith.mulf %4, %cst_0 : f64
    //   %6 = arith.subf %cst_1, %5 : f64
    //   affine.store %6, %alloc[%arg0] : memref<4xf64>
    // }

    // }
    // }
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: ReverseInputOp operations
//===----------------------------------------------------------------------===//

struct ReverseInputOpLowering : public ConversionPattern {
  ReverseInputOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::ReverseInputOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    // output = 0
    // iterate for len = 0 to N
    //   output[i] = a[N-1-i]

    DEBUG_PRINT_NO_ARGS();

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // For loop
    ReverseInputOpAdaptor reverseInputOpAdaptor(operands);
    // DEBUG_PRINT_NO_ARGS() ;

    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    // for loop
    affine::AffineForOp forOp1 =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto iv = forOp1.getInductionVar();

    rewriter.setInsertionPointToStart(forOp1.getBody());

    // DEBUG_PRINT_NO_ARGS() ;
    //: N-1 - i
    AffineExpr reverseIndxExpr =
        rewriter.getAffineConstantExpr(ub - 1) - rewriter.getAffineDimExpr(0);

    AffineMap addMap2 = AffineMap::get(1, 0, reverseIndxExpr);
    // load x[N-1-i]
    DEBUG_PRINT_NO_ARGS();
    Value loadInputFrmReverseIndx = rewriter.create<AffineLoadOp>(
        loc, reverseInputOpAdaptor.getInput(), addMap2, ValueRange{iv});

    // store the result at indx i
    rewriter.create<AffineStoreOp>(loc, loadInputFrmReverseIndx, alloc, iv);

    rewriter.setInsertionPointAfter(forOp1);
    // debug
    //  forOp1->dump();
    //    affine.for %arg0 = 0 to 5 {
    //    %0 = affine.load %alloc_6[%arg0] : memref<5xf64>
    //    %1 = arith.mulf %0, %0 : f64
    //    affine.store %1, %alloc_5[%arg0] : memref<5xf64>
    //  }
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: LengthOp operations
//===----------------------------------------------------------------------===//
struct LengthOpLowering : public ConversionPattern {
  LengthOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::LengthOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //   output = len(input)

    DEBUG_PRINT_NO_ARGS();

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    // iterate to result1 --not needed for now but for future reference

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);

    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    auto inputType = llvm::dyn_cast<RankedTensorType>(
        op->getOperand(0).getType()); // op->getOperand(

    int64_t ub = inputType.getShape()[0];
    Value constantUb = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(ub));

    DEBUG_PRINT_WITH_ARGS("\nCheck for index --here");
    // load from X, using 2nd operand as index
    //  DEBUG_PRINT_WITH_ARGS("Indx is" , SecondValueInt);
    Value constantIndx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    rewriter.create<AffineStoreOp>(loc, constantUb, alloc,
                                   ValueRange{constantIndx0});

    // debug
    //  forOpX->dump();
    //  forOpY->dump();
    //  affine.store %cst, %alloc_10[] : memref<f64>
    //  %0 = affine.load %alloc_11[4] : memref<10xf64>
    //  affine.store %0, %alloc[0] : memref<1xf64>

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: FFTRealOp operations
//===----------------------------------------------------------------------===//

struct FFTRealOpLowering : public ConversionPattern {
  FFTRealOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::FFTRealOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    auto memrefType = convertTensorToMemRef(tensorType);

    auto alloc_temp_real = insertAllocAndDealloc(memrefType, loc, rewriter);
    auto alloc_temp_imag = insertAllocAndDealloc(memrefType, loc, rewriter);

    FFTRealOpAdaptor fftRealOpAdaptor(operands);
    auto input = fftRealOpAdaptor.getLhs();
    auto lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto ub =
        rewriter.create<arith::ConstantIndexOp>(loc, tensorType.getShape()[0]);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // load real and imag
    auto load_temp = rewriter.create<scf::ForOp>(loc, lb, ub, step);
    rewriter.setInsertionPointToStart(load_temp.getBody());
    auto iv = load_temp.getInductionVar();
    auto inputValue =
        rewriter.create<memref::LoadOp>(loc, input, ValueRange{iv});
    auto constantZero = rewriter.create<arith::ConstantFloatOp>(
        loc, llvm::APFloat(0.0), rewriter.getF64Type());
    rewriter.create<memref::StoreOp>(loc, inputValue, alloc_temp_real,
                                     ValueRange{iv});
    rewriter.create<memref::StoreOp>(loc, constantZero, alloc_temp_imag,
                                     ValueRange{iv});
    rewriter.setInsertionPointAfter(load_temp);

    // alloc memory for reversed and dealloc when not required
    auto alloc_reversed_real = insertAllocAndDealloc(memrefType, loc, rewriter);
    auto alloc_reversed_imag = insertAllocAndDealloc(memrefType, loc, rewriter);

    // bits needed for bit  reversal
    auto ubInt =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), ub);
    auto ubFloat =
        rewriter.create<arith::SIToFPOp>(loc, rewriter.getF64Type(), ubInt);
    auto bitsNeededFloat = rewriter.create<math::Log2Op>(loc, ubFloat);
    auto bitsNeededInt = rewriter.create<arith::FPToSIOp>(
        loc, rewriter.getI64Type(), bitsNeededFloat);
    auto bitsNeeded = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), bitsNeededInt);

    // bit reversal
    auto bitReversalLoop = rewriter.create<scf::ForOp>(loc, lb, ub, step);
    rewriter.setInsertionPointToStart(bitReversalLoop.getBody());
    auto i = bitReversalLoop.getInductionVar();
    auto iInt = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(),
                                                    i); // check here

    // Calculate reversed index
    // auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto initialRevIndex = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);

    auto innerLoop = rewriter.create<scf::ForOp>(loc, lb, bitsNeeded, step,
                                                 ValueRange{initialRevIndex});
    rewriter.setInsertionPointToStart(innerLoop.getBody());
    auto j = innerLoop.getInductionVar();
    auto jInt =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), j);
    auto carriedRevIndex = innerLoop.getRegionIterArgs()[0];

    auto bitMask = rewriter.create<arith::ShLIOp>(
        loc, rewriter.create<arith::ConstantIntOp>(loc, 1, 64), jInt);
    auto iAndMask = rewriter.create<arith::AndIOp>(loc, iInt, bitMask);
    auto isNonZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, iAndMask,
        rewriter.create<arith::ConstantIntOp>(loc, 0, 64));
    auto shiftAmount = rewriter.create<arith::SubIOp>(
        loc, rewriter.create<arith::SubIOp>(loc, bitsNeeded, j),
        rewriter.create<arith::ConstantIndexOp>(loc, 1));
    auto shiftAmountI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), shiftAmount);
    auto bitToSet = rewriter.create<arith::ShLIOp>(
        loc, rewriter.create<arith::ConstantIntOp>(loc, 1, 64), shiftAmountI64);

    // Update newRevIndex using a select operation
    auto updatedRevIndex = rewriter.create<arith::OrIOp>(
        loc, carriedRevIndex,
        rewriter.create<arith::SelectOp>(
            loc, isNonZero, bitToSet,
            rewriter.create<arith::ConstantIntOp>(loc, 0, 64)));

    // Yield the updated value to carry it forward
    rewriter.create<scf::YieldOp>(loc, ValueRange{updatedRevIndex});

    // auto revIndex = rewriter.create<arith::IndexCastOp>(loc,
    // rewriter.getIndexType(), newRevIndex);

    rewriter.setInsertionPointAfter(innerLoop);

    auto finalRevIndex = innerLoop.getResult(0);
    auto revIndex = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), finalRevIndex);

    // Load from alloc_temp and store in alloc_reversed
    auto realValue =
        rewriter.create<memref::LoadOp>(loc, alloc_temp_real, ValueRange{i});
    auto imagValue =
        rewriter.create<memref::LoadOp>(loc, alloc_temp_imag, ValueRange{i});
    rewriter.create<memref::StoreOp>(loc, realValue, alloc_reversed_real,
                                     ValueRange{revIndex});
    rewriter.create<memref::StoreOp>(loc, imagValue, alloc_reversed_imag,
                                     ValueRange{revIndex});

    rewriter.setInsertionPointAfter(bitReversalLoop);

    // Cooley-Tukey FFT implementation
    auto N = tensorType.getShape()[0];
    auto stages = static_cast<int64_t>(std::log2(N));
    auto stagesValue = rewriter.create<arith::ConstantIndexOp>(loc, stages);

    // Constants for complex arithmetic
    auto pi = rewriter.create<arith::ConstantFloatOp>(loc, llvm::APFloat(M_PI),
                                                      rewriter.getF64Type());
    auto neg2 = rewriter.create<arith::ConstantFloatOp>(
        loc, llvm::APFloat(-2.0), rewriter.getF64Type());

    auto fftLoop = rewriter.create<scf::ForOp>(loc, lb, stagesValue, step);
    rewriter.setInsertionPointToStart(fftLoop.getBody());
    auto stage = fftLoop.getInductionVar();
    auto half_size = rewriter.create<arith::ShLIOp>(
        loc, rewriter.create<arith::ConstantIndexOp>(loc, 1), stage);
    auto full_size = rewriter.create<arith::ShLIOp>(
        loc, half_size, rewriter.create<arith::ConstantIndexOp>(loc, 1));

    auto outerLoop = rewriter.create<scf::ForOp>(loc, lb, ub, full_size);
    rewriter.setInsertionPointToStart(outerLoop.getBody());
    auto start = outerLoop.getInductionVar();

    auto butterflyLoop = rewriter.create<scf::ForOp>(loc, lb, half_size, step);
    rewriter.setInsertionPointToStart(butterflyLoop.getBody());
    auto k = butterflyLoop.getInductionVar();

    // Calculate indices for even and odd elements
    auto even_index = rewriter.create<arith::AddIOp>(loc, start, k);
    auto odd_index = rewriter.create<arith::AddIOp>(loc, even_index, half_size);

    // Calculate twiddle factor
    auto k_i64 =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), k);
    auto k_f64 =
        rewriter.create<arith::SIToFPOp>(loc, rewriter.getF64Type(), k_i64);
    auto full_size_i64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), full_size);
    auto full_size_f64 = rewriter.create<arith::SIToFPOp>(
        loc, rewriter.getF64Type(), full_size_i64);
    auto angle_div = rewriter.create<arith::DivFOp>(loc, k_f64, full_size_f64);
    auto angle_mul = rewriter.create<arith::MulFOp>(loc, neg2, angle_div);
    auto angle_final = rewriter.create<arith::MulFOp>(loc, pi, angle_mul);
    auto cos = rewriter.create<math::CosOp>(loc, angle_final);
    auto sin = rewriter.create<math::SinOp>(loc, angle_final);

    // Load odd value
    auto odd_real = rewriter.create<memref::LoadOp>(loc, alloc_reversed_real,
                                                    ValueRange{odd_index});
    auto odd_imag = rewriter.create<memref::LoadOp>(loc, alloc_reversed_imag,
                                                    ValueRange{odd_index});

    // Multiply by twiddle factor
    auto odd_real_cos = rewriter.create<arith::MulFOp>(loc, odd_real, cos);
    auto odd_imag_sin = rewriter.create<arith::MulFOp>(loc, odd_imag, sin);
    auto t_real =
        rewriter.create<arith::SubFOp>(loc, odd_real_cos, odd_imag_sin);

    auto odd_real_sin = rewriter.create<arith::MulFOp>(loc, odd_real, sin);
    auto odd_imag_cos = rewriter.create<arith::MulFOp>(loc, odd_imag, cos);
    auto t_imag =
        rewriter.create<arith::AddFOp>(loc, odd_real_sin, odd_imag_cos);

    // Load even value
    auto even_real = rewriter.create<memref::LoadOp>(loc, alloc_reversed_real,
                                                     ValueRange{even_index});
    auto even_imag = rewriter.create<memref::LoadOp>(loc, alloc_reversed_imag,
                                                     ValueRange{even_index});
    // Butterfly operation
    auto new_even_real = rewriter.create<arith::AddFOp>(loc, even_real, t_real);
    auto new_even_imag = rewriter.create<arith::AddFOp>(loc, even_imag, t_imag);
    auto new_odd_real = rewriter.create<arith::SubFOp>(loc, even_real, t_real);
    auto new_odd_imag = rewriter.create<arith::SubFOp>(loc, even_imag, t_imag);

    // Store results
    rewriter.create<memref::StoreOp>(loc, new_even_real, alloc_reversed_real,
                                     ValueRange{even_index});
    rewriter.create<memref::StoreOp>(loc, new_even_imag, alloc_reversed_imag,
                                     ValueRange{even_index});
    rewriter.create<memref::StoreOp>(loc, new_odd_real, alloc_reversed_real,
                                     ValueRange{odd_index});
    rewriter.create<memref::StoreOp>(loc, new_odd_imag, alloc_reversed_imag,
                                     ValueRange{odd_index});

    // replace the operation with the final value
    rewriter.replaceOp(op, alloc_reversed_real);
    return success();
  }
};
//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: FFTImagOp operations
//===----------------------------------------------------------------------===//

struct FFTImagOpLowering : public ConversionPattern {
  // constructor takes the mlir context and the operation as inputs
  FFTImagOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::FFTImagOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    auto memrefType = convertTensorToMemRef(tensorType);

    auto alloc_temp_real = insertAllocAndDealloc(memrefType, loc, rewriter);
    auto alloc_temp_imag = insertAllocAndDealloc(memrefType, loc, rewriter);

    FFTRealOpAdaptor fftRealOpAdaptor(operands);
    auto input = fftRealOpAdaptor.getLhs();
    auto lb = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto ub =
        rewriter.create<arith::ConstantIndexOp>(loc, tensorType.getShape()[0]);
    auto step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // load real and imag
    auto load_temp = rewriter.create<scf::ForOp>(loc, lb, ub, step);
    rewriter.setInsertionPointToStart(load_temp.getBody());
    auto iv = load_temp.getInductionVar();
    auto inputValue =
        rewriter.create<memref::LoadOp>(loc, input, ValueRange{iv});
    auto constantZero = rewriter.create<arith::ConstantFloatOp>(
        loc, llvm::APFloat(0.0), rewriter.getF64Type());
    rewriter.create<memref::StoreOp>(loc, inputValue, alloc_temp_real,
                                     ValueRange{iv});
    rewriter.create<memref::StoreOp>(loc, constantZero, alloc_temp_imag,
                                     ValueRange{iv});
    rewriter.setInsertionPointAfter(load_temp);

    // alloc memory for reversed and dealloc when not required
    auto alloc_reversed_real = insertAllocAndDealloc(memrefType, loc, rewriter);
    auto alloc_reversed_imag = insertAllocAndDealloc(memrefType, loc, rewriter);

    // bits needed for bit  reversal
    auto ubInt =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), ub);
    auto ubFloat =
        rewriter.create<arith::SIToFPOp>(loc, rewriter.getF64Type(), ubInt);
    auto bitsNeededFloat = rewriter.create<math::Log2Op>(loc, ubFloat);
    auto bitsNeededInt = rewriter.create<arith::FPToSIOp>(
        loc, rewriter.getI64Type(), bitsNeededFloat);
    auto bitsNeeded = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), bitsNeededInt);

    // bit reversal
    auto bitReversalLoop = rewriter.create<scf::ForOp>(loc, lb, ub, step);
    rewriter.setInsertionPointToStart(bitReversalLoop.getBody());
    auto i = bitReversalLoop.getInductionVar();
    auto iInt = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(),
                                                    i); // check here

    // Calculate reversed index
    // auto zero = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    auto initialRevIndex = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);

    auto innerLoop = rewriter.create<scf::ForOp>(loc, lb, bitsNeeded, step,
                                                 ValueRange{initialRevIndex});
    rewriter.setInsertionPointToStart(innerLoop.getBody());
    auto j = innerLoop.getInductionVar();
    auto jInt =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), j);
    auto carriedRevIndex = innerLoop.getRegionIterArgs()[0];

    auto bitMask = rewriter.create<arith::ShLIOp>(
        loc, rewriter.create<arith::ConstantIntOp>(loc, 1, 64), jInt);
    auto iAndMask = rewriter.create<arith::AndIOp>(loc, iInt, bitMask);
    auto isNonZero = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::ne, iAndMask,
        rewriter.create<arith::ConstantIntOp>(loc, 0, 64));
    auto shiftAmount = rewriter.create<arith::SubIOp>(
        loc, rewriter.create<arith::SubIOp>(loc, bitsNeeded, j),
        rewriter.create<arith::ConstantIndexOp>(loc, 1));
    auto shiftAmountI64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), shiftAmount);
    auto bitToSet = rewriter.create<arith::ShLIOp>(
        loc, rewriter.create<arith::ConstantIntOp>(loc, 1, 64), shiftAmountI64);

    // Update newRevIndex using a select operation
    auto updatedRevIndex = rewriter.create<arith::OrIOp>(
        loc, carriedRevIndex,
        rewriter.create<arith::SelectOp>(
            loc, isNonZero, bitToSet,
            rewriter.create<arith::ConstantIntOp>(loc, 0, 64)));

    // Yield the updated value to carry it forward
    rewriter.create<scf::YieldOp>(loc, ValueRange{updatedRevIndex});

    // auto revIndex = rewriter.create<arith::IndexCastOp>(loc,
    // rewriter.getIndexType(), newRevIndex);

    rewriter.setInsertionPointAfter(innerLoop);

    auto finalRevIndex = innerLoop.getResult(0);
    auto revIndex = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), finalRevIndex);

    // Load from alloc_temp and store in alloc_reversed
    auto realValue =
        rewriter.create<memref::LoadOp>(loc, alloc_temp_real, ValueRange{i});
    auto imagValue =
        rewriter.create<memref::LoadOp>(loc, alloc_temp_imag, ValueRange{i});
    rewriter.create<memref::StoreOp>(loc, realValue, alloc_reversed_real,
                                     ValueRange{revIndex});
    rewriter.create<memref::StoreOp>(loc, imagValue, alloc_reversed_imag,
                                     ValueRange{revIndex});

    rewriter.setInsertionPointAfter(bitReversalLoop);

    // Cooley-Tukey FFT implementation
    auto N = tensorType.getShape()[0];
    auto stages = static_cast<int64_t>(std::log2(N));
    auto stagesValue = rewriter.create<arith::ConstantIndexOp>(loc, stages);

    // Constants for complex arithmetic
    auto pi = rewriter.create<arith::ConstantFloatOp>(loc, llvm::APFloat(M_PI),
                                                      rewriter.getF64Type());
    auto neg2 = rewriter.create<arith::ConstantFloatOp>(
        loc, llvm::APFloat(-2.0), rewriter.getF64Type());

    auto fftLoop = rewriter.create<scf::ForOp>(loc, lb, stagesValue, step);
    rewriter.setInsertionPointToStart(fftLoop.getBody());
    auto stage = fftLoop.getInductionVar();
    auto half_size = rewriter.create<arith::ShLIOp>(
        loc, rewriter.create<arith::ConstantIndexOp>(loc, 1), stage);
    auto full_size = rewriter.create<arith::ShLIOp>(
        loc, half_size, rewriter.create<arith::ConstantIndexOp>(loc, 1));

    auto outerLoop = rewriter.create<scf::ForOp>(loc, lb, ub, full_size);
    rewriter.setInsertionPointToStart(outerLoop.getBody());
    auto start = outerLoop.getInductionVar();

    auto butterflyLoop = rewriter.create<scf::ForOp>(loc, lb, half_size, step);
    rewriter.setInsertionPointToStart(butterflyLoop.getBody());
    auto k = butterflyLoop.getInductionVar();

    // Calculate indices for even and odd elements
    auto even_index = rewriter.create<arith::AddIOp>(loc, start, k);
    auto odd_index = rewriter.create<arith::AddIOp>(loc, even_index, half_size);

    // Calculate twiddle factor
    auto k_i64 =
        rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), k);
    auto k_f64 =
        rewriter.create<arith::SIToFPOp>(loc, rewriter.getF64Type(), k_i64);
    auto full_size_i64 = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getI64Type(), full_size);
    auto full_size_f64 = rewriter.create<arith::SIToFPOp>(
        loc, rewriter.getF64Type(), full_size_i64);
    auto angle_div = rewriter.create<arith::DivFOp>(loc, k_f64, full_size_f64);
    auto angle_mul = rewriter.create<arith::MulFOp>(loc, neg2, angle_div);
    auto angle_final = rewriter.create<arith::MulFOp>(loc, pi, angle_mul);
    auto cos = rewriter.create<math::CosOp>(loc, angle_final);
    auto sin = rewriter.create<math::SinOp>(loc, angle_final);

    // Load odd value
    auto odd_real = rewriter.create<memref::LoadOp>(loc, alloc_reversed_real,
                                                    ValueRange{odd_index});
    auto odd_imag = rewriter.create<memref::LoadOp>(loc, alloc_reversed_imag,
                                                    ValueRange{odd_index});

    // Multiply by twiddle factor
    auto odd_real_cos = rewriter.create<arith::MulFOp>(loc, odd_real, cos);
    auto odd_imag_sin = rewriter.create<arith::MulFOp>(loc, odd_imag, sin);
    auto t_real =
        rewriter.create<arith::SubFOp>(loc, odd_real_cos, odd_imag_sin);

    auto odd_real_sin = rewriter.create<arith::MulFOp>(loc, odd_real, sin);
    auto odd_imag_cos = rewriter.create<arith::MulFOp>(loc, odd_imag, cos);
    auto t_imag =
        rewriter.create<arith::AddFOp>(loc, odd_real_sin, odd_imag_cos);

    // Load even value
    auto even_real = rewriter.create<memref::LoadOp>(loc, alloc_reversed_real,
                                                     ValueRange{even_index});
    auto even_imag = rewriter.create<memref::LoadOp>(loc, alloc_reversed_imag,
                                                     ValueRange{even_index});
    // Butterfly operation
    auto new_even_real = rewriter.create<arith::AddFOp>(loc, even_real, t_real);
    auto new_even_imag = rewriter.create<arith::AddFOp>(loc, even_imag, t_imag);
    auto new_odd_real = rewriter.create<arith::SubFOp>(loc, even_real, t_real);
    auto new_odd_imag = rewriter.create<arith::SubFOp>(loc, even_imag, t_imag);

    // Store results
    rewriter.create<memref::StoreOp>(loc, new_even_real, alloc_reversed_real,
                                     ValueRange{even_index});
    rewriter.create<memref::StoreOp>(loc, new_even_imag, alloc_reversed_imag,
                                     ValueRange{even_index});
    rewriter.create<memref::StoreOp>(loc, new_odd_real, alloc_reversed_real,
                                     ValueRange{odd_index});
    rewriter.create<memref::StoreOp>(loc, new_odd_imag, alloc_reversed_imag,
                                     ValueRange{odd_index});

    // replace the operation with the final value
    rewriter.replaceOp(op, alloc_reversed_imag);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: FIRFilterResSymmOptimizedOp operations
//===----------------------------------------------------------------------===//
struct FIRFilterResSymmOptimizedOpLowering : public ConversionPattern {
  FIRFilterResSymmOptimizedOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::FIRFilterResSymmOptimizedOp::getOperationName(),
                          1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // dsp.FIRFilterResSymmOptimizedOp has 2 operands -- both of type tensor f64

    // Get the location of FIRFilterResSymmOptimizedOp
    auto loc = op->getLoc();

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
    // Pseudo-Code
    // y[n] = sum(h[k] .{ x[n-k] + x[n-(L-1-k)]}) + h[L-1/2].x[n-(L-1)/2] , k=0
    // to L-1/2
    //  N = lenY , M = lenX ,  L = lenH
    // for n=0 to N
    //  sum = 0, temp =0
    //  for k = 0 to L-1/2
    // if 0 <= n-k < M
    // val1 = x[n-k] else, val1 = 0
    // if 0 <= n+k - (L-1) < M
    // val2 = x[n+k-(L-1)] else, val2 = 0
    // temp = val1 + val2
    //  sum = sum + h[k] . temp

    // middle-one
    //  if 0 <= n - (L-1)/2 < M
    //  sum2 = sum + h[L-1/2] . x[n-(n - (L-1)/2)]
    // y[n] = sum2

    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;
    DEBUG_PRINT_NO_ARGS();
    affine::AffineForOp forOp1 =
        rewriter.create<affine::AffineForOp>(loc, lb, ub, step);
    rewriter.setInsertionPointToStart(forOp1.getBody());
    auto iv = forOp1.getInductionVar();

    // for n=0 to N
    //  sum = 0, temp =0
    // get filter len
    //  auto tensorTypeFilter =
    //  llvm::cast<RankedTensorType>((*op->getOperand(1))); //operand_type_end
    //  auto tensorTypeFilter =
    //  llvm::cast<RankedTensorType>((*op->operand_type_begin()));
    auto operandIt = op->operand_type_begin();
    auto tensorTypeInput = llvm::cast<RankedTensorType>(*operandIt);
    int64_t ubForInput = tensorTypeInput.getShape()[0];
    // get second operand
    operandIt = operandIt + 1;

    // auto tensorTypeFilter =
    // llvm::cast<RankedTensorType>((*op->operand_type_begin())); //operandIt
    auto tensorTypeFilter = llvm::cast<RankedTensorType>(*operandIt);
    int64_t ubForFilter = tensorTypeFilter.getShape()[0];
    DEBUG_PRINT_NO_ARGS();
    // llvm::errs() << "ubForFilter= " << ubForFilter << "\n";
    // create a constant for sum
    Value constant0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
    affine::AffineForOp forOp2 = rewriter.create<affine::AffineForOp>(
        loc, lb, ubForFilter / 2, step, ValueRange{constant0});
    rewriter.setInsertionPointToStart(forOp2.getBody());
    auto iv2 = forOp2.getInductionVar();

    auto getIterArg =
        forOp2.getBody()->getArgument(1); // forOp1.getIterOperands();
    DEBUG_PRINT_NO_ARGS();
    FIRFilterResSymmOptimizedOpAdaptor firFilterResSymmOpAdaptor(operands);

    // if 0 <= n-k < M
    // val1 = x[n-k] else, val1 = 0
    // For n-k
    // if 0 <= n-k < M or, 0 <= n-k <= M -1
    AffineExpr d0, d1, s0, s1;
    bindDims(rewriter.getContext(), d0, d1);
    AffineExpr ExprNMinusK = d0 - d1;
    AffineMap mapNMinusK = AffineMap::get(2, 0, ExprNMinusK);
    // n-k <= M -1 or, n-k-(M-1) <= 0
    bindSymbols(rewriter.getContext(), s0, s1);
    Value constantMMinus1Indx =
        rewriter.create<arith::ConstantIndexOp>(loc, ubForInput - 1);

    AffineExpr ExprNMinusKMinusMPlus1 = s0 - d0 + d1;
    IntegerSet setForIf = IntegerSet::get(
        2, 1, {ExprNMinusK, ExprNMinusKMinusMPlus1}, {false, false});
    DEBUG_PRINT_NO_ARGS();

    // if 0 <= n-k <= M -1
    // use typeRange too:
    Type floatType = rewriter.getF64Type();
    //  if n-k >= 0 && n-k <= M -1 or, M-1 -n + k >= 0
    auto ifOp = rewriter.create<affine::AffineIfOp>(
        loc, TypeRange{floatType}, setForIf,
        ValueRange{iv, iv2, constantMMinus1Indx}, true /*else*/);
    rewriter.setInsertionPointToStart(ifOp.getThenBlock());

    // val1 = x[n-k] else, val1 = 0
    // load x[n-k]
    DEBUG_PRINT_NO_ARGS();
    Value loadInput =
        rewriter.create<AffineLoadOp>(loc, firFilterResSymmOpAdaptor.getLhs(),
                                      mapNMinusK, ValueRange{iv, iv2});
    rewriter.create<AffineYieldOp>(loc, ValueRange{loadInput});
    // else block
    rewriter.setInsertionPointToStart(ifOp.getElseBlock());
    Value const0ForElse = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
    rewriter.create<AffineYieldOp>(loc, ValueRange{const0ForElse});
    rewriter.setInsertionPointAfter(ifOp);

    // if 0 <= n+k - (L-1) < M
    // val2 = x[n+k-(L-1)] else, val2 = 0
    // val2 lower bound
    //  AffineExpr ExprNMinKMinLPlus1 = d0 - d1 - s0; //s0 = (L-1) => -s0 = -L+1
    //  AffineExpr ExprLowerBoundVal2 = d0 - d1 - s0; //s0 = (L-1) => -s0 = -L+1
    // Val2 LowerBound: n+k - (L-1) >= 0
    AffineExpr ExprLowerBoundVal2 =
        rewriter.getAffineDimExpr(0) + rewriter.getAffineDimExpr(1) -
        rewriter.getAffineConstantExpr(ubForFilter - 1);
    // Val2 UpperBound: n+k - (L-1) <= M -1 ie, M - 1 + L -1 -k -n >= 0 ie,
    // (M+L-2) - k -n >= 0
    //  AffineExpr ExprUpperBoundVal2 = s0 + s1 + d1 - d0; //s1 = M+L-2 = L-1 +
    //  M -1
    AffineExpr ExprUpperBoundVal2 =
        rewriter.getAffineConstantExpr(ubForInput + ubForFilter - 2) -
        rewriter.getAffineDimExpr(1) - rewriter.getAffineDimExpr(0);
    // s0 = L -1
    //  Value s0LMin1Indx = rewriter.create<arith::ConstantIndexOp>(loc,
    //  ubForFilter - 1); s1 = M + L -2 for val2 upperBound Value
    //  s1MPlusLPlus2Indx = rewriter.create<arith::ConstantIndexOp>(loc,
    //  ubForInput + ubForFilter - 2); Value s1MMin1Indx =
    //  rewriter.create<arith::ConstantIndexOp>(loc, ubForInput - 1);

    IntegerSet setForIf2 = IntegerSet::get(
        2, 0, {ExprLowerBoundVal2, ExprUpperBoundVal2}, {false, false});

    auto ifOp2 = rewriter.create<affine::AffineIfOp>(
        loc, TypeRange{floatType}, setForIf2, ValueRange{iv, iv2},
        true /*else*/);
    rewriter.setInsertionPointToStart(ifOp2.getThenBlock());

    // val2 = x[n+k-(L-1)] else, val2 = 0
    AffineMap addMap2 = AffineMap::get(2, 0, ExprLowerBoundVal2);
    // load x[n+k-(L-1)]
    DEBUG_PRINT_NO_ARGS();
    Value loadInputForVal2 = rewriter.create<AffineLoadOp>(
        loc, firFilterResSymmOpAdaptor.getLhs(), addMap2, ValueRange{iv, iv2});
    rewriter.create<AffineYieldOp>(loc, ValueRange{loadInputForVal2});
    // else block
    rewriter.setInsertionPointToStart(ifOp2.getElseBlock());
    Value const0ForElse2 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
    rewriter.create<AffineYieldOp>(loc, ValueRange{const0ForElse2});
    rewriter.setInsertionPointAfter(ifOp2);

    // temp = val1 + val2
    //  sum = sum + h[k] . temp

    Value Val1Plus2 = rewriter.create<arith::AddFOp>(loc, ifOp.getResult(0),
                                                     ifOp2.getResult(0));

    // load filter and then mult and then sum
    Value loadFilter = rewriter.create<affine::AffineLoadOp>(
        loc, firFilterResSymmOpAdaptor.getRhs(), iv2);

    Value filterMulInput =
        rewriter.create<arith::MulFOp>(loc, Val1Plus2, loadFilter);
    Value sumNext =
        rewriter.create<arith::AddFOp>(loc, filterMulInput, getIterArg);
    rewriter.create<AffineYieldOp>(loc, ValueRange{sumNext});
    // rewriter.setInsertionPointToEnd(forOp2->getBlock());
    rewriter.setInsertionPointAfter(forOp2);
    DEBUG_PRINT_NO_ARGS();
    // Middle - point
    // if 0 <= n - (L-1)/2 < M
    // sum2 = sum + h[L-1/2] . x[n-(L-1)/2)]
    // y[n] = sum2

    // if 0 <= n - (L-1)/2 < M
    // AffineExpr ExprLowerBoundVal3 = d0 - s0; //s0 = (L-1)/2
    // AffineExpr ExprUpperBoundVal3 = d0 - s1; //s1 = M+ (L-1)/2
    int64_t midFilterLen = (ubForFilter - 1) / 2;
    AffineExpr ExprLowerBoundVal3 =
        rewriter.getAffineDimExpr(0) -
        rewriter.getAffineConstantExpr(midFilterLen);
    // UpperBound: n - (L-1)/2 <= M - 1 ie, M-1 + mid - n
    AffineExpr ExprUpperBoundVal3 =
        rewriter.getAffineConstantExpr(ubForInput + midFilterLen - 1) -
        rewriter.getAffineDimExpr(0);

    AffineMap addMap3 = AffineMap::get(1, 0, ExprLowerBoundVal3);

    IntegerSet setForIf3 = IntegerSet::get(
        1, 0, {ExprLowerBoundVal3, ExprUpperBoundVal3}, {false, false});

    auto ifOp3 = rewriter.create<affine::AffineIfOp>(
        loc, TypeRange{floatType}, setForIf3, ValueRange{iv}, true /*else*/);
    rewriter.setInsertionPointToStart(ifOp3.getThenBlock());

    // val3 = x[n-(L-1)/2)] else, val3 = 0
    // load x[n-(L-1)/2)]
    DEBUG_PRINT_NO_ARGS();
    Value loadInputForVal3 = rewriter.create<AffineLoadOp>(
        loc, firFilterResSymmOpAdaptor.getLhs(), addMap3, ValueRange{iv});
    rewriter.create<AffineYieldOp>(loc, ValueRange{loadInputForVal3});
    // else block
    rewriter.setInsertionPointToStart(ifOp3.getElseBlock());
    Value const0ForElse3 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
    rewriter.create<AffineYieldOp>(loc, ValueRange{const0ForElse3});
    rewriter.setInsertionPointAfter(ifOp3);

    // sum2 = sum + h[L-1/2] . x[n-(L-1)/2)]
    //  y[n] = sum2
    // load filter and then mult and then sum
    Value midFilterLenIndx =
        rewriter.create<arith::ConstantIndexOp>(loc, midFilterLen);

    Value loadFilterMid = rewriter.create<affine::AffineLoadOp>(
        loc, firFilterResSymmOpAdaptor.getRhs(), midFilterLenIndx);
    Value filterMulInput2 =
        rewriter.create<arith::MulFOp>(loc, ifOp3.getResult(0), loadFilterMid);
    Value sum2 = rewriter.create<arith::AddFOp>(loc, filterMulInput2,
                                                forOp2.getResult(0));
    // rewriter.create<AffineStoreOp>(loc, forOp2.getResult(0) , alloc, iv);
    rewriter.create<AffineStoreOp>(loc, sum2, alloc, iv);
    rewriter.setInsertionPointAfter(forOp1);
    DEBUG_PRINT_NO_ARGS();
    // ifOp->dump();
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: RunLenEncodingOp operations

//===----------------------------------------------------------------------===//

#define TryWhileLoop 0
#define TryLoadStoreForWhile 0
#define TryPassIterIndex 0 // Not working
#define TryScf 0
#define TryRLE 1
struct RunLenEncodingOpLowering : public ConversionPattern {
  RunLenEncodingOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::RunLenEncodingOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //   y_rle[i] =  x[i] , if x[i] != x[i-1] , 1<=i<n
    //  CountOfXi , at n<=i < 2n -1

    // steps:
    //  count = 1 , y[0] = x[0] , k = 0
    //  for i=1 to len/2
    //  load prev = a[i-1] , current = a[i]
    //  if prev == current
    //  count = count + 1
    // else
    // store count at index k + N/2
    // y[k] = current
    // y[k + N/2] = count
    // count = 1 and k = k+1
    // if count > 1 ie, for last element
    //  store the count value at k + N/2

    DEBUG_PRINT_NO_ARGS();

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    auto tensorType1 = RankedTensorType::get({1}, rewriter.getIndexType());

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto memRefType2 = convertTensorToMemRef(tensorType1);

    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
    auto allocK = insertAllocAndDealloc(memRefType2, loc, rewriter);

    // count = 1 , y[0] = x[0] ,
    // loop from 0 to len
    RunLenEncodingOpAdaptor runLenEncodingAdaptor(operands);
    DEBUG_PRINT_NO_ARGS();

    //  len/2,k = n ie, len/2
    int64_t lb = 1;
    int64_t N = tensorType.getShape()[0];
    int64_t ub = N / 2; // output len is twice the input len
    int64_t step = 1;
    int64_t k = 0;
    int64_t lb1 = 0;

    Value const0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

    // init all output memory with zero
    affine::AffineForOp forOp1 =
        rewriter.create<AffineForOp>(loc, lb1, N, step);
    DEBUG_PRINT_NO_ARGS();
    auto iv1 = forOp1.getInductionVar();
    rewriter.setInsertionPointToStart(forOp1.getBody());
    rewriter.create<AffineStoreOp>(loc, const0, alloc, iv1);
    rewriter.setInsertionPointAfter(forOp1);

    DEBUG_PRINT_NO_ARGS();
    // load from X,
    Value constantIndx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value inputX0 = rewriter.create<AffineLoadOp>(
        loc, runLenEncodingAdaptor.getInput(), ValueRange{constantIndx0});
    rewriter.create<AffineStoreOp>(loc, inputX0, alloc,
                                   ValueRange{constantIndx0});

#if TryRLE

    // Initial count and k values as SSA values, count = 1 , k = 0
    // for i=1 to len/2
    // load prev = a[i-1] , current = a[i]
    // if prev == current
    // count = count + 1
    // else
    // store count at index k + N/2
    // y[k + N/2] = count
    // k = k +1
    // y[k] = current
    // count = 1
    // for last element
    //  store the count value at k + N/2
    // y[k + N/2] = count
    Value countVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(1));
    Value Indx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    Value IndxNBy2 = rewriter.create<arith::ConstantIndexOp>(loc, ub);
    Value kVal = rewriter.create<arith::ConstantIndexOp>(loc, k);
    rewriter.create<AffineStoreOp>(loc, kVal, allocK, ValueRange{Indx0});

    Type floatType = rewriter.getF64Type();
    // Type indexType = rewriter.getIndexType();
    //// // for i=1 to len/2
    // load prev = a[i-1] , current = a[i]
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step, ValueRange{countVal});
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());
    DEBUG_PRINT_NO_ARGS();

    auto countArg = forOpY.getRegionIterArgs()[0];

    Value current = rewriter.create<AffineLoadOp>(
        loc, runLenEncodingAdaptor.getInput(), ivY);
    //
    AffineExpr d0;
    bindDims(rewriter.getContext(), d0);
    AffineExpr ExprIMinus1 = d0 - rewriter.getAffineConstantExpr(1);
    AffineMap mapExprIMinus1 = AffineMap::get(1, 0, ExprIMinus1);
    Value prev = rewriter.create<AffineLoadOp>(
        loc, runLenEncodingAdaptor.getInput(), mapExprIMinus1, ValueRange{ivY});
    DEBUG_PRINT_NO_ARGS();
    // for i=1 to len/2
    // load prev = a[i-1] , current = a[i]
    // if prev == current
    // count = count + 1
    // else
    // store count at index k + N/2
    // y[k + N/2] = count
    // k = k +1
    // y[k] = current
    // count = 1
    // for last element
    //  store the count value at k + N/2
    // y[k + N/2] = count
    // TypeRange typeRange = TypeRange{rewriter.getF64Type() ,
    // rewriter.getIndexType()}; TypeRange typeRange =
    // TypeRange({rewriter.getF64Type(), rewriter.getIndexType()});

    // auto ifOp = rewriter.create<scf::IfOp>(loc,
    // TypeRange{rewriter.getF64Type(), rewriter.getIndexType()},
    // rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, prev,
    // current), true, true);
    auto CmpPrevCurrent = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, prev, current);

    // create if block with else condition
    //  if prev == current
    //  count = count + 1
    // auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange{floatType ,
    // indexType}, CmpPrevCurrent , true /* else=1 */);
    auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange{floatType},
                                           CmpPrevCurrent, true /* else=1 */);

    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    DEBUG_PRINT_NO_ARGS();

    auto CountPlusOne = rewriter.create<arith::AddFOp>(loc, countArg, countVal);
    DEBUG_PRINT_NO_ARGS();
    rewriter.create<scf::YieldOp>(loc, ValueRange{CountPlusOne});
    // else
    // store count at index k + N/2
    // y[k + N/2] = count
    // k = k +1
    // y[k] = current
    // count = 1
    rewriter.setInsertionPointToStart(ifOp.elseBlock());
    // // out[k + N/2]= count
    Value loadKVal =
        rewriter.create<AffineLoadOp>(loc, allocK, ValueRange{Indx0});

    Value kPlusNBy2 = rewriter.create<arith::AddIOp>(
        loc, rewriter.getIndexType(), loadKVal, IndxNBy2);
    rewriter.create<memref::StoreOp>(loc, countArg, alloc, kPlusNBy2);
    // k = k+1
    Value Indx1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value kPlusOne = rewriter.create<arith::AddIOp>(
        loc, rewriter.getIndexType(), loadKVal, Indx1);
    rewriter.create<AffineStoreOp>(loc, kPlusOne, allocK, ValueRange{Indx0});

    // y[k + 1] = current
    rewriter.create<memref::StoreOp>(loc, current, alloc, kPlusOne);

    DEBUG_PRINT_NO_ARGS();
    rewriter.create<scf::YieldOp>(loc, ValueRange{countVal});
    rewriter.setInsertionPointAfter(ifOp);
    // ifOp.dump();
    Value countRes = ifOp.getResult(0);

    rewriter.create<AffineYieldOp>(loc, ValueRange{countRes});
    rewriter.setInsertionPointAfter(forOpY);
    // forOpY->dump();

    // check for last countArg value if countArg > 1, then store it at last
    Value finalCountArg = forOpY.getResult(0);
    Value finalkArg =
        rewriter.create<AffineLoadOp>(loc, allocK, ValueRange{Indx0});

    // //if count>1 ,then store count at index k + N/2
    // auto ifOp1 = rewriter.create<scf::IfOp>(loc, CmpCountGt1 , false /*
    // else=0 */);
    // rewriter.setInsertionPointToStart(ifOp1.thenBlock());
    DEBUG_PRINT_NO_ARGS();
    Value finalkPlusNBy2 = rewriter.create<arith::AddIOp>(
        loc, rewriter.getIndexType(), finalkArg, IndxNBy2);

    rewriter.create<memref::StoreOp>(loc, finalCountArg, alloc, finalkPlusNBy2);
    DEBUG_PRINT_NO_ARGS();
    // rewriter.setInsertionPointAfter(ifOp1);
#endif

#if TryPassIterIndex
    // store k at its location & load and do addition to 1 and
    Value kVal = rewriter.create<arith::ConstantIndexOp>(loc, ub - 1);
    Value Indx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    auto kValStore =
        rewriter.create<AffineStoreOp>(loc, kVal, alloc2, ValueRange{Indx0});

    Type floatType = rewriter.getF64Type();
    Type indexType = rewriter.getIndexType();
    affine::AffineForOp forOpY = rewriter.create<AffineForOp>(
        loc, lb, ub, step, ValueRange{inputX0, kVal});
    // affine::AffineForOp forOpY = rewriter.create<AffineForOp>(loc, lb, ub,
    // step, ValueRange{countVal, kVal});

    auto ivY = forOpY.getInductionVar();
    auto prev = forOpY.getRegionIterArgs()[0];
    auto kArg = forOpY.getRegionIterArgs()[1];
    rewriter.setInsertionPointToStart(forOpY.getBody());

    Value Indx00 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value current = rewriter.create<AffineLoadOp>(
        loc, runLenEncodingAdaptor.getInput(), ivY);
    Value loadKVal =
        rewriter.create<AffineLoadOp>(loc, alloc2, ValueRange{Indx0});
    Value const1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(1));
    Value currentPlus1 = rewriter.create<arith::AddFOp>(loc, prev, const1);

    auto CmpPrevCurrent = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGE, current, const1);

    // create if block with else condition
    //  if prev == current, count++
    auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange{floatType},
                                           CmpPrevCurrent, true /* else=1 */);
    // auto ifOp = rewriter.create<scf::IfOp>(loc,  CmpPrevCurrent , true /*
    // else=1 */);

    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    DEBUG_PRINT_NO_ARGS();

    // store count at N+i
    //  Value countPlus1 = rewriter.create<arith::AddFOp>(loc, countArg,
    //  countVal);
    Value Indx1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value kPlusOne = rewriter.create<arith::AddIOp>(
        loc, rewriter.getIndexType(), kArg, Indx1);

    rewriter.create<AffineStoreOp>(loc, current, alloc, ValueRange{kArg});
    // rewriter.create<AffineStoreOp>(loc, current, alloc,
    // ValueRange{kPlusOne});
    rewriter.create<memref::StoreOp>(loc, current, alloc, ValueRange{kPlusOne});
    rewriter.create<AffineStoreOp>(loc, kPlusOne, alloc2, ValueRange{Indx0});
    rewriter.create<scf::YieldOp>(loc, ValueRange{currentPlus1});

    rewriter.setInsertionPointToStart(ifOp.elseBlock());
    rewriter.create<AffineStoreOp>(loc, currentPlus1, alloc, ValueRange{ivY});
    // yield the values
    //  rewriter.create<AffineYieldOp>(loc, ValueRange{kPlusOne });
    rewriter.create<scf::YieldOp>(loc, ValueRange{currentPlus1});

    rewriter.setInsertionPointAfter(ifOp);
    Value countRes = ifOp.getResult(0);
    // Value kRes = ifOp.getResult(1);
    // rewriter.create<AffineYieldOp>(loc, ValueRange{countRes,kRes });
    rewriter.create<AffineYieldOp>(loc, ValueRange{countRes, Indx00});

    rewriter.setInsertionPointAfter(forOpY);

#endif

#if TryWhileLoop

    auto kVal = rewriter.create<arith::ConstantIndexOp>(loc, k);
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step, ValueRange{kVal});
    auto ivY = forOpY.getInductionVar();
    // auto countArg = forOpY.getRegionIterArgs()[0];
    auto kArg = forOpY.getRegionIterArgs()[0];
    rewriter.setInsertionPointToStart(forOpY.getBody());

    Value current = rewriter.create<AffineLoadOp>(
        loc, runLenEncodingAdaptor.getInput(), ivY);

    // store count at N+i
    //  Value countPlus1 = rewriter.create<arith::AddFOp>(loc, countArg,
    //  countVal);
    Value Indx1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value kPlusOne = rewriter.create<arith::AddIOp>(
        loc, rewriter.getIndexType(), kArg, Indx1);
    // Value constInt1 =
    // rewriter.create<arith::ConstantIntOp>(loc,rewriter.getI64IntegerAttr(1),
    // rewriter.getI64Type() );

    // Value kPlusOneIndex = rewriter.create<arith::IndexCastOp>(loc,
    // rewriter.getIndexType(), kPlusOne);

    // kPlusOne.dump();
    // Value kArg1 = rewriter.create<arith::IndexCastUIOp>(loc,
    // rewriter.getIndexType(), kArg);

    // rewriter.create<AffineStoreOp>(loc, countPlus1, alloc, mapExprNPlusI,
    // ValueRange{kPlusOne}); rewriter.create<AffineStoreOp>(loc, countPlus1,
    // alloc, ValueRange{kArg}); Store the result
    // rewriter.create<AffineStoreOp>(loc, current, alloc, ivY); //working
    rewriter.create<AffineStoreOp>(loc, current, alloc, ValueRange{kArg});
    // yield the values
    rewriter.create<AffineYieldOp>(loc, ValueRange{kPlusOne});
    // rewriter.create<AffineYieldOp>(loc, ValueRange{countPlus1 , kPlusOne});
    rewriter.setInsertionPointAfter(forOpY);

#endif

#if TryLoadStoreForWhile
    // store k at its location & load and do addition to 1 and
    Value kVal = rewriter.create<arith::ConstantIndexOp>(loc, ub - 1);
    Value Indx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    auto kValStore =
        rewriter.create<AffineStoreOp>(loc, kVal, alloc2, ValueRange{Indx0});
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step, ValueRange{inputX0});
    auto ivY = forOpY.getInductionVar();
    auto prev = forOpY.getRegionIterArgs()[0];
    // auto kArg = forOpY.getRegionIterArgs()[0];
    rewriter.setInsertionPointToStart(forOpY.getBody());

    Value Indx00 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value current = rewriter.create<AffineLoadOp>(
        loc, runLenEncodingAdaptor.getInput(), ivY);
    Value loadKVal =
        rewriter.create<AffineLoadOp>(loc, alloc2, ValueRange{Indx0});
    Value const1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(1));
    Value currentPlus1 = rewriter.create<arith::AddFOp>(loc, prev, const1);

    auto CmpPrevCurrent = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGE, current, const1);

    // create if block with else condition
    //  if prev == current, count++
    //  auto ifOp = rewriter.create<scf::IfOp>(loc, TypeRange{floatType ,
    //  indexType}, CmpPrevCurrent , true /* else=1 */);
    auto ifOp =
        rewriter.create<scf::IfOp>(loc, CmpPrevCurrent, true /* else=1 */);

    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    DEBUG_PRINT_NO_ARGS();

    // store count at N+i
    //  Value countPlus1 = rewriter.create<arith::AddFOp>(loc, countArg,
    //  countVal);
    Value Indx1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    Value kPlusOne = rewriter.create<arith::AddIOp>(
        loc, rewriter.getIndexType(), loadKVal, Indx1);

    rewriter.create<AffineStoreOp>(loc, current, alloc, ValueRange{ivY});
    // rewriter.create<AffineStoreOp>(loc, current, alloc,
    // ValueRange{kPlusOne});
    rewriter.create<memref::StoreOp>(loc, current, alloc, ValueRange{kPlusOne});
    rewriter.create<AffineStoreOp>(loc, kPlusOne, alloc2, ValueRange{Indx0});

    rewriter.setInsertionPointToStart(ifOp.elseBlock());
    rewriter.create<AffineStoreOp>(loc, currentPlus1, alloc, ValueRange{ivY});
    // yield the values
    //  rewriter.create<AffineYieldOp>(loc, ValueRange{kPlusOne });
    rewriter.setInsertionPointAfter(ifOp);
    rewriter.create<AffineYieldOp>(loc, ValueRange{current});

    rewriter.setInsertionPointAfter(forOpY);

#endif

    // debug
    //  forOpY->dump();
    //  affine.store %cst, %alloc_10[] : memref<f64>
    //  %0 = affine.load %alloc_11[4] : memref<10xf64>
    //  affine.store %0, %alloc[0] : memref<1xf64>

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: lmsFilterResponse operations
//===----------------------------------------------------------------------===//

struct LMSFilterResponseOpLowering : public ConversionPattern {
  LMSFilterResponseOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::LMSFilterResponseOp::getOperationName(), 1,
                          ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //  for (int n = 0; n < NUM_SAMPLES; n++) {
    //      // Calculate the filter output y[n]
    //      y[n] = 0;
    //      for (int i = 0; i < FILTER_LENGTH; i++) {
    //          if (n - i >= 0) { // affine if
    //              y[n] = y[n] + (w[i] * x[n - i]);
    //          }
    //      }

    //     // Calculate the error e[n]
    //     e[n] = d[n] - y[n];

    //     // Update the filter weights w[i]
    //     for (int i = 0; i < FILTER_LENGTH; i++) {
    //         if (n - i >= 0) {
    //             w[i] +=  MU * e[n] * x[n - i];
    //         }
    //     }
    // }

    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    LMSFilterOpAdaptor lmsFilterAdaptor(operands);
    // Value alpha = rewriter.create<arith::ConstantOp>(loc,
    // rewriter.getF64Type(),
    //                                                      rewriter.getF64FloatAttr(1));
    Value zeroval = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
    Value mu = rewriter.create<AffineLoadOp>(loc, lmsFilterAdaptor.getMu());

    // For loop -- iterate from 0 to last
    int64_t lb = 0;
    int64_t numSamples = tensorType.getShape()[0];
    int64_t step = 1;

    Value GetFilterLOp = op->getOperand(3);
    dsp::ConstantOp constantOp3rdArg =
        GetFilterLOp.getDefiningOp<dsp::ConstantOp>();
    DenseElementsAttr constant3rdValue = constantOp3rdArg.getValue();
    ;
    auto elements1 = constant3rdValue.getValues<FloatAttr>();
    float filterlenval = elements1[0].getValueAsDouble();
    auto FilterLength = (uint64_t)filterlenval;

    auto yMemRefType = MemRefType::get({numSamples}, rewriter.getF64Type());
    auto wAlloc = rewriter.create<memref::AllocOp>(loc, yMemRefType);

    affine::AffineForOp forOp1 =
        rewriter.create<AffineForOp>(loc, lb, numSamples, step);
    auto iv = forOp1.getInductionVar();

    rewriter.setInsertionPointToStart(forOp1.getBody());

    // For affine expression: #map1 = affine_map<(%arg0)[] : (%arg0 - 1)
    AffineExpr d0, d1, s0;
    bindDims(rewriter.getContext(), d0, d1);
    // AffineExpr ExprForXSlice = rewriter.getAffineDimExpr(0) -
    // rewriter.getAffineDimExpr(1); //d0 - d1;
    AffineExpr ExprForXSlice = d0 - d1;
    AffineMap addMapForLMSFilter = AffineMap::get(2, 0, ExprForXSlice);
    IntegerSet set1 = IntegerSet::get(2, 0, {ExprForXSlice}, {false});

    // y[n] = 0;
    // rewriter.create<AffineStoreOp>(loc, zeroval, alloc, ValueRange{iv});
    // Allocate and initialize array for y
    // Value constantIndx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    rewriter.create<AffineStoreOp>(loc, zeroval, alloc, ValueRange{iv});

    affine::AffineForOp forOp2 =
        rewriter.create<AffineForOp>(loc, lb, FilterLength, step);
    auto iv2 = forOp2.getInductionVar();

    rewriter.setInsertionPointToStart(forOp2.getBody());

    auto ifOp = rewriter.create<affine::AffineIfOp>(
        loc, set1, ValueRange{iv, iv2}, false /*no else*/);
    rewriter.setInsertionPointToStart(ifOp.getThenBlock());

    Value inputX =
        rewriter.create<AffineLoadOp>(loc, lmsFilterAdaptor.getLhs(),
                                      addMapForLMSFilter, ValueRange{iv, iv2});
    Value w = rewriter.create<AffineLoadOp>(loc, wAlloc,
                                            ValueRange{iv2}); // memRefType

    Value wmulx = rewriter.create<arith::MulFOp>(loc, inputX, w);
    Value ybefore = rewriter.create<AffineLoadOp>(loc, alloc, ValueRange{iv});
    Value sumNext = rewriter.create<arith::AddFOp>(loc, wmulx, ybefore);
    rewriter.create<AffineStoreOp>(loc, sumNext, alloc, ValueRange{iv});
    rewriter.setInsertionPointAfter(ifOp);
    rewriter.setInsertionPointAfter(forOp2);

    //  get e[n] = d[n] - y[n]

    Value desiredX = rewriter.create<AffineLoadOp>(
        loc, lmsFilterAdaptor.getRhs(), ValueRange{iv});
    Value ynew = rewriter.create<AffineLoadOp>(loc, alloc, ValueRange{iv});

    Value err = rewriter.create<arith::SubFOp>(loc, desiredX, ynew);

    affine::AffineForOp forOp3 =
        rewriter.create<AffineForOp>(loc, lb, FilterLength, step);
    auto iv3 = forOp3.getInductionVar();

    rewriter.setInsertionPointToStart(forOp3.getBody());

    auto ifOp2 = rewriter.create<affine::AffineIfOp>(
        loc, set1, ValueRange{iv, iv3}, false /*no else*/);
    rewriter.setInsertionPointToStart(ifOp2.getThenBlock());

    Value inputX2 =
        rewriter.create<AffineLoadOp>(loc, lmsFilterAdaptor.getLhs(),
                                      addMapForLMSFilter, ValueRange{iv, iv3});

    Value Prevw2 = rewriter.create<AffineLoadOp>(loc, wAlloc, ValueRange{iv3});

    // f(u(n),e(n),μ)=μe(n)u∗(n)
    Value mul1 = rewriter.create<arith::MulFOp>(loc, err, inputX2);
    Value mul2 = rewriter.create<arith::MulFOp>(loc, mu, mul1);

    // FInal w[n]
    Value answer = rewriter.create<arith::AddFOp>(loc, Prevw2, mul2);

    rewriter.create<AffineStoreOp>(loc, answer, wAlloc, ValueRange{iv3});
    rewriter.setInsertionPointAfter(ifOp2);
    rewriter.setInsertionPointAfter(forOp3);

    rewriter.setInsertionPointAfter(forOp1);
    // debug
    //  forOp1->dump();

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Quantization operations
//===----------------------------------------------------------------------===//

struct QuantizationOpLowering : public ConversionPattern {
  QuantizationOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::QuantizationOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //  y_quantized[i] = Round(a[i] - min) / step) * step + min
    //    where, step = (max-min)/ NoOfLevels , NoOLevels = 2^NoOfBits

    // 	steps:
    // 		1) given NoOfLevels
    // 		2) Then calculate stepSize = (Max-Min)/NoOfLevels
    // 		3) iterate for all the elements and calculate quantizedCoeff

    // 			GetLevelForVal =  (a[i] - min)/step
    // 			RoundedVal = arith.FPToSI(GetLevelForVal)
    // 			QuantVal = RoundedVal * step + min_val

    DEBUG_PRINT_NO_ARGS();

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);

    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // create another memory location for getting NoOfLevels

    // Value constant1 = rewriter.create<arith::ConstantOp>(loc,
    // rewriter.getF64Type(),
    //                                                      rewriter.getF64FloatAttr(1));

    // 1) Then calculate stepSize = (Max-Min)/NoOfLevels

    QuantizationOpAdaptor quantizationAdaptor(operands);
    DEBUG_PRINT_NO_ARGS();
    Value getMaxMemref = quantizationAdaptor.getMax();
    auto getMax =
        rewriter.create<AffineLoadOp>(loc, getMaxMemref, ValueRange{});

    Value getMinMemref = quantizationAdaptor.getMin();
    auto getMin =
        rewriter.create<AffineLoadOp>(loc, getMinMemref, ValueRange{});

    Value getNLevelsMemref = quantizationAdaptor.getNlevels();
    auto getNlevels =
        rewriter.create<AffineLoadOp>(loc, getNLevelsMemref, ValueRange{});

    Value MaxMinusMin = rewriter.create<arith::SubFOp>(loc, getMax, getMin);
    Value StepSize =
        rewriter.create<arith::DivFOp>(loc, MaxMinusMin, getNlevels);

    // iterate for all the elements and calculate quantizedCoeff

    // 			GetLevelForVal =  (a[i] - min)/step
    // 			RoundedVal = arith.FPToSI(GetLevelForVal)
    // 			QuantVal = RoundedVal * step + min_val
    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    DEBUG_PRINT_NO_ARGS();

    // for loop from 0 to len
    //  use iter_arg as passing value for the loop
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());
    // Use iter_arg for taking prev_val
    // Get iter_arg

    // 			GetLevelForVal =  (a[i] - min)/step

    // 			QuantVal = RoundedVal * step + min_val

    Value inputX =
        rewriter.create<AffineLoadOp>(loc, quantizationAdaptor.getInput(), ivY);
    Value inputMinusMin = rewriter.create<arith::SubFOp>(loc, inputX, getMin);
    Value aMinusMinDivStep =
        rewriter.create<arith::DivFOp>(loc, inputMinusMin, StepSize);

    // 	RoundedVal = arith.FPToSI(GetLevelForVal)
    Value RoundedVal = rewriter.create<arith::FPToSIOp>(
        loc, rewriter.getI64Type(), aMinusMinDivStep);
    Value RoundValFloat = rewriter.create<arith::SIToFPOp>(
        loc, rewriter.getF64Type(), RoundedVal);

    // 	QuantVal = RoundedVal * step + min_val
    Value RoundedMulStep =
        rewriter.create<arith::MulFOp>(loc, RoundValFloat, StepSize);
    Value QuantVal =
        rewriter.create<arith::AddFOp>(loc, RoundedMulStep, getMin);
    rewriter.create<AffineStoreOp>(loc, QuantVal, alloc, ValueRange{ivY});
    rewriter.setInsertionPointAfter(forOpY);

    // debug
    //  forOpY->dump();
    //  affine.store %cst, %alloc_10[] : memref<f64>
    //  %0 = affine.load %alloc_11[4] : memref<10xf64>
    //  affine.store %0, %alloc[0] : memref<1xf64>

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: lmsFilter operations
//===----------------------------------------------------------------------===//

struct LMSFilterOpLowering : public ConversionPattern {
  LMSFilterOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::LMSFilterOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //  for (int n = 0; n < NUM_SAMPLES; n++) {
    //      // Calculate the filter output y[n]
    //      y[n] = 0;
    //      for (int i = 0; i < FILTER_LENGTH; i++) {
    //          if (n - i >= 0) { // affine if
    //              y[n] = y[n] + (w[i] * x[n - i]);
    //          }
    //      }

    //     // Calculate the error e[n]
    //     e[n] = d[n] - y[n];

    //     // Update the filter weights w[i]
    //     for (int i = 0; i < FILTER_LENGTH; i++) {
    //         if (n - i >= 0) {
    //             w[i] +=  MU * e[n] * x[n - i];
    //         }
    //     }
    // }

    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    LMSFilterOpAdaptor lmsFilterAdaptor(operands);
    // Value alpha = rewriter.create<arith::ConstantOp>(loc,
    // rewriter.getF64Type(),
    //                                                      rewriter.getF64FloatAttr(1));
    Value zeroval = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
    Value mu = rewriter.create<AffineLoadOp>(loc, lmsFilterAdaptor.getMu());

    // For loop -- iterate from 0 to last
    int64_t lb = 0;
    int64_t numSamples = tensorType.getShape()[0];
    int64_t step = 1;

    Value GetFilterLOp = op->getOperand(3);
    dsp::ConstantOp constantOp3rdArg =
        GetFilterLOp.getDefiningOp<dsp::ConstantOp>();
    DenseElementsAttr constant3rdValue = constantOp3rdArg.getValue();
    ;
    auto elements1 = constant3rdValue.getValues<FloatAttr>();
    float filterlenval = elements1[0].getValueAsDouble();
    auto FilterLength = (uint64_t)filterlenval;

    Value GetItersLOp = op->getOperand(4);
    dsp::ConstantOp constantOp4thArg =
        GetItersLOp.getDefiningOp<dsp::ConstantOp>();
    DenseElementsAttr constant4thValue = constantOp4thArg.getValue();
    ;
    auto elements = constant4thValue.getValues<FloatAttr>();
    float interationsval = elements[0].getValueAsDouble();
    auto TotalIterations = (uint64_t)interationsval;

    auto yMemRefType = MemRefType::get({numSamples}, rewriter.getF64Type());
    auto yAlloc = rewriter.create<memref::AllocOp>(loc, yMemRefType);

    affine::AffineForOp forOpiter =
        rewriter.create<AffineForOp>(loc, lb, TotalIterations, step);
    rewriter.setInsertionPointToStart(forOpiter.getBody());
    affine::AffineForOp forOp1 =
        rewriter.create<AffineForOp>(loc, lb, numSamples, step);
    auto iv = forOp1.getInductionVar();

    rewriter.setInsertionPointToStart(forOp1.getBody());

    // For affine expression: #map1 = affine_map<(%arg0)[] : (%arg0 - 1)
    AffineExpr d0, d1, s0;
    bindDims(rewriter.getContext(), d0, d1);
    // AffineExpr ExprForXSlice = rewriter.getAffineDimExpr(0) -
    // rewriter.getAffineDimExpr(1); //d0 - d1;
    AffineExpr ExprForXSlice = d0 - d1;
    AffineMap addMapForLMSFilter = AffineMap::get(2, 0, ExprForXSlice);
    IntegerSet set1 = IntegerSet::get(2, 0, {ExprForXSlice}, {false});

    // y[n] = 0;
    // rewriter.create<AffineStoreOp>(loc, zeroval, alloc, ValueRange{iv});
    // Allocate and initialize array for y
    // Value constantIndx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    rewriter.create<AffineStoreOp>(loc, zeroval, yAlloc, ValueRange{iv});

    affine::AffineForOp forOp2 =
        rewriter.create<AffineForOp>(loc, lb, FilterLength, step);
    auto iv2 = forOp2.getInductionVar();

    rewriter.setInsertionPointToStart(forOp2.getBody());

    auto ifOp = rewriter.create<affine::AffineIfOp>(
        loc, set1, ValueRange{iv, iv2}, false /*no else*/);
    rewriter.setInsertionPointToStart(ifOp.getThenBlock());

    Value inputX =
        rewriter.create<AffineLoadOp>(loc, lmsFilterAdaptor.getLhs(),
                                      addMapForLMSFilter, ValueRange{iv, iv2});
    Value Prevw = rewriter.create<AffineLoadOp>(loc, alloc,
                                                ValueRange{iv2}); // memRefType

    Value wmulx = rewriter.create<arith::MulFOp>(loc, inputX, Prevw);
    Value ybefore = rewriter.create<AffineLoadOp>(loc, yAlloc, ValueRange{iv});
    Value sumNext = rewriter.create<arith::AddFOp>(loc, wmulx, ybefore);
    rewriter.create<AffineStoreOp>(loc, sumNext, yAlloc, ValueRange{iv});
    rewriter.setInsertionPointAfter(ifOp);
    rewriter.setInsertionPointAfter(forOp2);

    //  get e[n] = d[n] - y[n]

    Value desiredX = rewriter.create<AffineLoadOp>(
        loc, lmsFilterAdaptor.getRhs(), ValueRange{iv});
    Value ynew = rewriter.create<AffineLoadOp>(loc, yAlloc, ValueRange{iv});

    Value err = rewriter.create<arith::SubFOp>(loc, desiredX, ynew);

    affine::AffineForOp forOp3 =
        rewriter.create<AffineForOp>(loc, lb, FilterLength, step);
    auto iv3 = forOp3.getInductionVar();

    rewriter.setInsertionPointToStart(forOp3.getBody());

    auto ifOp2 = rewriter.create<affine::AffineIfOp>(
        loc, set1, ValueRange{iv, iv3}, false /*no else*/);
    rewriter.setInsertionPointToStart(ifOp2.getThenBlock());

    Value inputX2 =
        rewriter.create<AffineLoadOp>(loc, lmsFilterAdaptor.getLhs(),
                                      addMapForLMSFilter, ValueRange{iv, iv3});

    Value Prevw2 = rewriter.create<AffineLoadOp>(loc, alloc, ValueRange{iv3});

    // f(u(n),e(n),μ)=μe(n)u∗(n)
    Value mul1 = rewriter.create<arith::MulFOp>(loc, err, inputX2);
    Value mul2 = rewriter.create<arith::MulFOp>(loc, mu, mul1);

    // FInal w[n]
    Value answer = rewriter.create<arith::AddFOp>(loc, Prevw2, mul2);

    rewriter.create<AffineStoreOp>(loc, answer, alloc, ValueRange{iv3});
    rewriter.setInsertionPointAfter(ifOp2);
    rewriter.setInsertionPointAfter(forOp3);

    rewriter.setInsertionPointAfter(forOp1);
    rewriter.setInsertionPointAfter(forOpiter);
    // debug
    //  forOp1->dump();

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Threshold operations
//===----------------------------------------------------------------------===//

struct ThresholdOpLowering : public ConversionPattern {
  ThresholdOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::ThresholdOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //   y[n] = a[n] , if a[i] >= threshld or, a[i] <= -threshld
    //     = 0 , else

    DEBUG_PRINT_NO_ARGS();

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);

    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    Value constant0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

    // y[n] = a[n] , if a[i] >= threshld or, a[i] <= -threshld
    // loop from 0 to len

    // load from X,
    ThresholdOpAdaptor thresholdAdaptor(operands);
    DEBUG_PRINT_NO_ARGS();

    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    DEBUG_PRINT_NO_ARGS();

    // for loop from 0 to len(Output)
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());

    Value inputX =
        rewriter.create<AffineLoadOp>(loc, thresholdAdaptor.getInput(), ivY);

    // Load the threshold value from the memref
    auto thresholdMemRef = thresholdAdaptor.getThreshld();
    auto threshold =
        rewriter.create<AffineLoadOp>(loc, thresholdMemRef, ValueRange{});

    // Compare a[i] <= threshold
    auto cmp1 = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLE,
                                               inputX, threshold);

    // Compare a[i] >= -threshold
    auto negThreshold = rewriter.create<arith::NegFOp>(loc, threshold);
    auto cmp2 = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE,
                                               inputX, negThreshold);

    // Combine the comparisons using AND
    auto cmpAnd = rewriter.create<arith::AndIOp>(loc, cmp1, cmp2);

    // Use select to choose between 0 and a[i]
    auto selectOp =
        rewriter.create<arith::SelectOp>(loc, cmpAnd, constant0, inputX);

    // Store the result
    rewriter.create<AffineStoreOp>(loc, selectOp, alloc, ivY);

    rewriter.setInsertionPointAfter(forOpY);
    // debug
    //  forOpY->dump();
    //  affine.store %cst, %alloc_10[] : memref<f64>
    //  %0 = affine.load %alloc_11[4] : memref<10xf64>
    //  affine.store %0, %alloc[0] : memref<1xf64>

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: HighPassFIRHammingOptimizedOp operations
//===----------------------------------------------------------------------===//

struct HighPassFIRHammingOptimizedOpLowering : public ConversionPattern {
  HighPassFIRHammingOptimizedOpLowering(MLIRContext *ctx)
      : ConversionPattern(
            dsp::HighPassFIRHammingOptimizedOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //  y_highFIRHamming[n] = -1 * [wc/pi * sinc(wc * (n- (N-1)/2))] * [0.54 -
    //  0.46 cos(2 *pi * n/N-1)], 0<= n < (N-1)/2 : = 1 - wc/pi , n = (N-1)/2

    // and also, y_FIRHamming[N-1-n] = y[n] ie, store at n and also at N-1-n

    // 1 loops : first from 0 <= n < (N-1)/2 - 1
    //

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // first from 0 <= i < (N-1)/2 - 1
    int64_t lb = 0;
    int64_t N = tensorType.getShape()[0];
    int64_t ub = (N - 1) / 2;
    int64_t step = 1;

    DEBUG_PRINT_NO_ARGS();
    HighPassFIRHammingOptimizedOpAdaptor highPassFIRHammingOptimizedOpAdaptor(
        operands);
    // Handle middle y[mid] = wc / pi
    int64_t midIndx = ub;
    Value constantIndxMid =
        rewriter.create<arith::ConstantIndexOp>(loc, midIndx);
    // rewriter.create<AffineStoreOp>(loc, constant0, alloc,
    // ValueRange{constantIndx0});
    Value wc = rewriter.create<AffineLoadOp>(
        loc, highPassFIRHammingOptimizedOpAdaptor.getWc(), ValueRange{});
    Value constant1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(1));
    Value constantMinus1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(-1));
    Value constpi = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(3.14159265359));
    Value wcByPi = rewriter.create<arith::DivFOp>(loc, wc, constpi);
    Value OneMinusWcByPi =
        rewriter.create<arith::SubFOp>(loc, constant1, wcByPi);
    rewriter.create<AffineStoreOp>(loc, OneMinusWcByPi, alloc,
                                   ValueRange{constantIndxMid});

    // first from 0 <= i < (N-1)/2 - 1

    // calculate i-(N-1)/2

    Value Nminus1By2 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr((float)ub));

    // calculate 0.54 - 0.46 cos(2 *pi * n/N-1)
    Value constant0_54 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0.54));
    Value constant0_46 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0.46));
    Value const2pi = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(6.28318530718));
    Value NMinus1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr((float)N - 1));

    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());
    // convert index to f64
    Value IndxY = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivY);
    Value i =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxY);

    // get sin(wc * (i - (N-1)/ 2))
    Value iMinusMid = rewriter.create<arith::SubFOp>(loc, i, Nminus1By2);
    Value mulwc_iMinusMid = rewriter.create<arith::MulFOp>(loc, wc, iMinusMid);

    Value GetSin = rewriter.create<math::SinOp>(loc, mulwc_iMinusMid);

    // sin(wc*(i-(N-1)/2)) / pi * (i-(N-1)/2)

    Value piMuliMinusMid =
        rewriter.create<arith::MulFOp>(loc, constpi, iMinusMid);
    Value GetDiv = rewriter.create<arith::DivFOp>(loc, GetSin, piMuliMinusMid);

    // [sin(wc*(i-(N-1)/2)) / pi * (i-(N-1)/2)] * [0.54-0.46 cos(2*pi*i/N-1)

    // get 2*pi * k / (N -1)
    Value mul2pi_k = rewriter.create<arith::MulFOp>(loc, const2pi, i);
    Value divIndxByNMinus1 =
        rewriter.create<arith::DivFOp>(loc, mul2pi_k, NMinus1);

    // get cos(2*pi * k/(N-1)
    Value GetCos = rewriter.create<math::CosOp>(loc, divIndxByNMinus1);
    Value MulCos0_46 =
        rewriter.create<arith::MulFOp>(loc, constant0_46, GetCos);
    Value Sub0_54_Cos =
        rewriter.create<arith::SubFOp>(loc, constant0_54, MulCos0_46);

    // Multiply Sub0_54_Cos and GetDiv -- sin(wc*(i-(N-1)/2)) / pi * (i-(N-1)/2)
    Value MulFilterHamming =
        rewriter.create<arith::MulFOp>(loc, GetDiv, Sub0_54_Cos);
    Value MulByMinus1 =
        rewriter.create<arith::MulFOp>(loc, constantMinus1, MulFilterHamming);
    rewriter.create<AffineStoreOp>(loc, MulByMinus1, alloc, ValueRange{ivY});

    // also , store same value at N-1-i using affine-Map
    // For affine expression: #map1 = affine_map<(%arg0)[N] : (N - 1 -%arg0)
    AffineExpr d0, s0;
    bindDims(rewriter.getContext(), d0);
    bindSymbols(rewriter.getContext(), s0);
    // calulate N - 1 - i
    AffineExpr ExprForNMinus1minusI = s0 - d0;
    AffineMap addMapForNMinus1minusI =
        AffineMap::get(1, 1, ExprForNMinus1minusI);

    // store at N-1-i index , result
    Value constantNMinus1Indx =
        rewriter.create<arith::ConstantIndexOp>(loc, N - 1);
    rewriter.create<AffineStoreOp>(loc, MulByMinus1, alloc,
                                   addMapForNMinus1minusI,
                                   ValueRange{ivY, constantNMinus1Indx});
    rewriter.setInsertionPointAfter(forOpY);

    // debug
    //  forOpX->dump();
    //  forOpY->dump();

    // affine.for %arg0 = 0 to 3 {
    //   %12 = arith.index_castui %arg0 : index to i32
    //   %13 = arith.uitofp %12 : i32 to f64
    //   %14 = arith.subf %13, %cst_3 : f64
    //   %15 = arith.mulf %9, %14 : f64
    //   %16 = math.sin %15 : f64
    //   %17 = arith.mulf %14, %cst_9 : f64
    //   %18 = arith.divf %16, %17 : f64
    //   %19 = arith.mulf %13, %cst_0 : f64
    //   %20 = arith.divf %19, %cst : f64
    //   %21 = math.cos %20 : f64
    //   %22 = arith.mulf %21, %cst_1 : f64
    //   %23 = arith.subf %cst_2, %22 : f64
    //   %24 = arith.mulf %18, %23 : f64
    //   %25 = arith.mulf %24, %cst_4 : f64
    //   affine.store %25, %alloc[%arg0] : memref<7xf64>
    //   affine.store %25, %alloc[-%arg0 + 6] : memref<7xf64>
    // }

    // }
    // }
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: FIRFilterHammingOptimizedOp operations
//===----------------------------------------------------------------------===//

struct FIRFilterHammingOptimizedOpLowering : public ConversionPattern {
  FIRFilterHammingOptimizedOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::FIRFilterHammingOptimizedOp::getOperationName(),
                          1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //   y_FIRHamming[n] = [wc/pi * sinc(wc * (n- (N-1)/2))] * [0.54 - 0.46
    //   cos(2 *pi * n/N-1)], 0<= n < (N-1)/2 :
    //  = wc/pi * 1 , n = (N-1)/2

    // and also, y_FIRHamming[N-1-n] = y[n] ie, store at n and also at N-1-n

    // 1 loops : first from 0 <= n < (N-1)/2 - 1
    //

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // first from 0 <= i < (N-1)/2 - 1
    int64_t lb = 0;
    int64_t N = tensorType.getShape()[0];
    int64_t ub = (N - 1) / 2;
    int64_t step = 1;

    DEBUG_PRINT_NO_ARGS();
    FIRFilterHammingOptimizedOpAdaptor firFilterHammingOptimizedOpAdaptor(
        operands);
    // Handle middle y[mid] = wc / pi
    int64_t midIndx = ub;
    Value constantIndxMid =
        rewriter.create<arith::ConstantIndexOp>(loc, midIndx);
    // rewriter.create<AffineStoreOp>(loc, constant0, alloc,
    // ValueRange{constantIndx0});
    Value wc = rewriter.create<AffineLoadOp>(
        loc, firFilterHammingOptimizedOpAdaptor.getWc(), ValueRange{});

    Value constpi = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(3.14159265359));
    Value wcByPi = rewriter.create<arith::DivFOp>(loc, wc, constpi);

    rewriter.create<AffineStoreOp>(loc, wcByPi, alloc,
                                   ValueRange{constantIndxMid});

    // first from 0 <= i < (N-1)/2 - 1

    // calculate i-(N-1)/2

    Value Nminus1By2 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr((float)ub));

    // calculate 0.54 - 0.46 cos(2 *pi * n/N-1)
    Value constant0_54 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0.54));
    Value constant0_46 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0.46));
    Value const2pi = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(6.28318530718));
    Value NMinus1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr((float)N - 1));

    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());
    // convert index to f64
    Value IndxY = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivY);
    Value i =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxY);

    // get sin(wc * (i - (N-1)/ 2))
    Value iMinusMid = rewriter.create<arith::SubFOp>(loc, i, Nminus1By2);
    Value mulwc_iMinusMid = rewriter.create<arith::MulFOp>(loc, wc, iMinusMid);

    Value GetSin = rewriter.create<math::SinOp>(loc, mulwc_iMinusMid);

    // sin(wc*(i-(N-1)/2)) / pi * (i-(N-1)/2)

    Value piMuliMinusMid =
        rewriter.create<arith::MulFOp>(loc, constpi, iMinusMid);
    Value GetDiv = rewriter.create<arith::DivFOp>(loc, GetSin, piMuliMinusMid);

    // [sin(wc*(i-(N-1)/2)) / pi * (i-(N-1)/2)] * [0.54-0.46 cos(2*pi*i/N-1)

    // get 2*pi * k / (N -1)
    Value mul2pi_k = rewriter.create<arith::MulFOp>(loc, const2pi, i);
    Value divIndxByNMinus1 =
        rewriter.create<arith::DivFOp>(loc, mul2pi_k, NMinus1);

    // get cos(2*pi * k/(N-1)
    Value GetCos = rewriter.create<math::CosOp>(loc, divIndxByNMinus1);
    Value MulCos0_46 =
        rewriter.create<arith::MulFOp>(loc, constant0_46, GetCos);
    Value Sub0_54_Cos =
        rewriter.create<arith::SubFOp>(loc, constant0_54, MulCos0_46);

    // Multiply Sub0_54_Cos and GetDiv -- sin(wc*(i-(N-1)/2)) / pi * (i-(N-1)/2)
    Value MulFilterHamming =
        rewriter.create<arith::MulFOp>(loc, GetDiv, Sub0_54_Cos);
    rewriter.create<AffineStoreOp>(loc, MulFilterHamming, alloc,
                                   ValueRange{ivY});

    // also , store same value at N-1-i using affine-Map
    // For affine expression: #map1 = affine_map<(%arg0)[N] : (N - 1 -%arg0)
    AffineExpr d0, s0;
    bindDims(rewriter.getContext(), d0);
    bindSymbols(rewriter.getContext(), s0);
    // calulate N - 1 - i
    AffineExpr ExprForNMinus1minusI = s0 - d0;
    AffineMap addMapForNMinus1minusI =
        AffineMap::get(1, 1, ExprForNMinus1minusI);

    // store at N-1-i index , result
    Value constantNMinus1Indx =
        rewriter.create<arith::ConstantIndexOp>(loc, N - 1);
    rewriter.create<AffineStoreOp>(loc, MulFilterHamming, alloc,
                                   addMapForNMinus1minusI,
                                   ValueRange{ivY, constantNMinus1Indx});
    rewriter.setInsertionPointAfter(forOpY);

    // debug
    //  forOpX->dump();
    //  forOpY->dump();

    // %cst = arith.constant 6.2831853071800001 : f64
    // %cst_0 = arith.constant 4.600000e-01 : f64
    // %cst_1 = arith.constant 5.400000e-01 : f64
    // %cst_2 = arith.constant 4.000000e+00 : f64
    // %alloc = memref.alloc() : memref<4xf64>
    // %alloc_3 = memref.alloc() : memref<f64>
    // affine.store %cst_2, %alloc_3[] : memref<f64>
    // affine.for %arg0 = 0 to 4 {
    //   %0 = arith.index_castui %arg0 : index to i32
    //   %1 = arith.uitofp %0 : i32 to f64
    //   %2 = arith.mulf %1, %cst : f64
    //   %3 = arith.divf %2, %cst_2 : f64
    //   %4 = math.cos %3 : f64
    //   %5 = arith.mulf %4, %cst_0 : f64
    //   %6 = arith.subf %cst_1, %5 : f64
    //   affine.store %6, %alloc[%arg0] : memref<4xf64>
    // }

    // }
    // }
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: GetRangeOfVectorOp operations
//===----------------------------------------------------------------------===//

struct GetRangeOfVectorOpLowering : public ConversionPattern {
  GetRangeOfVectorOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::GetRangeOfVectorOp::getOperationName(), 1, ctx) {
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //   y[0] = first:
    //   y[i] = y[i-1] + step for  1<=i<N
    //
    // Alt:  y[0] = first , prev_val = first
    //   for i =1 to N
    //    y[i] = prev_val
    //    prev_val = prev_val + step

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
    GetRangeOfVectorOpAdaptor getRangeOfVectorOpOpAdaptor(operands);

    Value GetValueAtIndx2ndArg = op->getOperand(0);
    dsp::ConstantOp constantOp2ndArg =
        GetValueAtIndx2ndArg.getDefiningOp<dsp::ConstantOp>();
    DenseElementsAttr constantRhsValue = constantOp2ndArg.getValue();
    ;
    auto elements = constantRhsValue.getValues<FloatAttr>();
    float FirstValue = elements[0].getValueAsDouble();

    DEBUG_PRINT_WITH_ARGS("FirstValue is", FirstValue);
    Value GetStepOp = op->getOperand(2);
    dsp::ConstantOp constantOp3rdArg =
        GetStepOp.getDefiningOp<dsp::ConstantOp>();
    DenseElementsAttr constant3rdValue = constantOp3rdArg.getValue();
    ;
    auto elements1 = constant3rdValue.getValues<FloatAttr>();
    float StepValue = elements1[0].getValueAsDouble();

    // first from 1 <= i < N
    int64_t lb = 1;
    int64_t ub = tensorType.getShape()[0];
    // int64_t ub = (N-1) / 2 ;
    int64_t step = 1;

    DEBUG_PRINT_NO_ARGS();

    float valAtIndxI = FirstValue;

    Value constantIndx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value constantFirst = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(valAtIndxI));
    Value constantStep = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(StepValue));

    rewriter.create<AffineStoreOp>(loc, constantFirst, alloc,
                                   ValueRange{constantIndx0});

    // loop from 1 <= i < N

    affine::AffineForOp forOpY = rewriter.create<AffineForOp>(
        loc, lb, ub, step, ValueRange{constantFirst});
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());
    // Use iter_arg for taking prev_val
    // Get iter_arg
    auto getIterArg = forOpY.getBody()->getArgument(1);
    // getIterArg.dump();

    Value sumNext =
        rewriter.create<arith::AddFOp>(loc, getIterArg, constantStep);
    rewriter.create<AffineStoreOp>(loc, sumNext, alloc, ValueRange{ivY});
    // llvm::errs() << "LINE " << __LINE__ << " file= " << __FILE__ << "\n" ;
    rewriter.create<AffineYieldOp>(loc, ValueRange{sumNext});
    rewriter.setInsertionPointAfter(forOpY);

    // debug
    //  forOpX->dump();
    //  forOpY->dump();

    // %cst = arith.constant 6.2831853071800001 : f64
    // %cst_0 = arith.constant 4.600000e-01 : f64
    // %cst_1 = arith.constant 5.400000e-01 : f64
    // %cst_2 = arith.constant 4.000000e+00 : f64
    // %alloc = memref.alloc() : memref<4xf64>
    // %alloc_3 = memref.alloc() : memref<f64>
    // affine.store %cst_2, %alloc_3[] : memref<f64>
    // affine.for %arg0 = 0 to 4 {
    //   %0 = arith.index_castui %arg0 : index to i32
    //   %1 = arith.uitofp %0 : i32 to f64
    //   %2 = arith.mulf %1, %cst : f64
    //   %3 = arith.divf %2, %cst_2 : f64
    //   %4 = math.cos %3 : f64
    //   %5 = arith.mulf %4, %cst_0 : f64
    //   %6 = arith.subf %cst_1, %5 : f64
    //   affine.store %6, %alloc[%arg0] : memref<4xf64>
    // }

    // }
    // }
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: HighPassFIRFilterOp operations
//===----------------------------------------------------------------------===//

struct HighPassFIRFilterOpLowering : public ConversionPattern {
  HighPassFIRFilterOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::HighPassFIRFilterOp::getOperationName(), 1,
                          ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //   y_lpf[n] = wc/pi * sinc(wc * (n- (N-1)/2)) , n!= (N-1)/2 :
    //            = wc/pi , n = (N-1)/2
    //  y_hpf[n] = dirac(n- (N-1)/2) - y_lpf[n] = -1 * wc/pi * sinc(wc * (n-
    //  (N-1)/2)) , n!= (N-1)/2 :
    //           = 1 - wc/pi , n = (N-1)/2

    // 2 loops : first from 0 <= n <= (N-1)/2 - 1
    //      2nd from (N-1)/2 +1 <= n < N

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // first from 0 <= i <= (N-1)/2 - 1
    int64_t lb = 0;
    int64_t N = tensorType.getShape()[0];
    int64_t ub = (N - 1) / 2;
    int64_t step = 1;

    DEBUG_PRINT_NO_ARGS();
    HighPassFIRFilterOpAdaptor highPassfirFilterOpAdaptor(operands);
    // Handle middle y[mid] = wc / pi
    int64_t midIndx = ub;
    Value constantIndxMid =
        rewriter.create<arith::ConstantIndexOp>(loc, midIndx);
    // rewriter.create<AffineStoreOp>(loc, constant0, alloc,
    // ValueRange{constantIndx0});
    Value constant1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(1));
    Value constantMinus1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(-1));

    Value wc = rewriter.create<AffineLoadOp>(
        loc, highPassfirFilterOpAdaptor.getWc(), ValueRange{});

    Value constpi = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(3.14159265359));
    Value wcByPi = rewriter.create<arith::DivFOp>(loc, wc, constpi);
    Value OneMinusWcByPi =
        rewriter.create<arith::SubFOp>(loc, constant1, wcByPi);
    rewriter.create<AffineStoreOp>(loc, OneMinusWcByPi, alloc,
                                   ValueRange{constantIndxMid});

    // first from 0 <= i <= (N-1)/2 - 1

    // calculate i-(N-1)/2
    Value Nminus1By2 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr((float)ub));
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());
    // convert index to f64
    Value IndxY = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivY);
    Value i =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxY);

    // get sin(wc * (i - (N-1)/ 2))
    Value iMinusMid = rewriter.create<arith::SubFOp>(loc, i, Nminus1By2);
    Value mulwc_iMinusMid = rewriter.create<arith::MulFOp>(loc, wc, iMinusMid);

    Value GetSin = rewriter.create<math::SinOp>(loc, mulwc_iMinusMid);

    // get sin(wc*i) / pi * i

    Value piMuliMinusMid =
        rewriter.create<arith::MulFOp>(loc, constpi, iMinusMid);
    Value GetDiv = rewriter.create<arith::DivFOp>(loc, GetSin, piMuliMinusMid);
    Value MulByMinus1 =
        rewriter.create<arith::MulFOp>(loc, constantMinus1, GetDiv);
    rewriter.create<AffineStoreOp>(loc, MulByMinus1, alloc, ValueRange{ivY});
    // llvm::errs() << "LINE " << __LINE__ << " file= " << __FILE__ << "\n" ;
    rewriter.setInsertionPointAfter(forOpY);

    // 2nd loop from (N-1)/2 + 1 <= i < N
    lb = ub + 1;
    ub = N;

    affine::AffineForOp forOp1 =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto iv1 = forOp1.getInductionVar();
    rewriter.setInsertionPointToStart(forOp1.getBody());
    // convert index to f64
    Value Indx1 = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), iv1);
    Value i1 =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), Indx1);

    // get sin(wc * (i1 - (N-1)/ 2))
    Value iMinusMid1 = rewriter.create<arith::SubFOp>(loc, i1, Nminus1By2);
    Value mulwc_iMinusMid1 =
        rewriter.create<arith::MulFOp>(loc, wc, iMinusMid1);
    Value GetSin1 = rewriter.create<math::SinOp>(loc, mulwc_iMinusMid1);

    // get sin(i1 - (N-1)/ 2) / (i1 - (N-1)/ 2) * pi
    //  get sin(wc*i1) / pi * i1

    Value piMuliMinusMid1 =
        rewriter.create<arith::MulFOp>(loc, constpi, iMinusMid1);
    Value GetDiv1 =
        rewriter.create<arith::DivFOp>(loc, GetSin1, piMuliMinusMid1);

    Value GetDiv1MulNeg1 =
        rewriter.create<arith::MulFOp>(loc, constantMinus1, GetDiv1);

    rewriter.create<AffineStoreOp>(loc, GetDiv1MulNeg1, alloc, ValueRange{iv1});
    // llvm::errs() << "LINE " << __LINE__ << " file= " << __FILE__ << "\n" ;
    rewriter.setInsertionPointAfter(forOp1);

    // debug
    //  forOpX->dump();
    //  forOpY->dump();

    // %cst = arith.constant 6.2831853071800001 : f64
    // %cst_0 = arith.constant 4.600000e-01 : f64
    // %cst_1 = arith.constant 5.400000e-01 : f64
    // %cst_2 = arith.constant 4.000000e+00 : f64
    // %alloc = memref.alloc() : memref<4xf64>
    // %alloc_3 = memref.alloc() : memref<f64>
    // affine.store %cst_2, %alloc_3[] : memref<f64>
    // affine.for %arg0 = 0 to 4 {
    //   %0 = arith.index_castui %arg0 : index to i32
    //   %1 = arith.uitofp %0 : i32 to f64
    //   %2 = arith.mulf %1, %cst : f64
    //   %3 = arith.divf %2, %cst_2 : f64
    //   %4 = math.cos %3 : f64
    //   %5 = arith.mulf %4, %cst_0 : f64
    //   %6 = arith.subf %cst_1, %5 : f64
    //   affine.store %6, %alloc[%arg0] : memref<4xf64>
    // }

    // }
    // }
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: LowPassFIRFilterOp operations
//===----------------------------------------------------------------------===//

struct LowPassFIRFilterOpLowering : public ConversionPattern {
  LowPassFIRFilterOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::LowPassFIRFilterOp::getOperationName(), 1, ctx) {
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //   y_lpf[n] = wc/pi * sinc(wc * (n- (N-1)/2)) , n!= (N-1)/2 :
    //            = wc/pi , n = (N-1)/2

    // 2 loops : first from 0 <= n <= (N-1)/2 - 1
    //      2nd from (N-1)/2 +1 <= n < N

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // first from 0 <= i <= (N-1)/2 - 1
    int64_t lb = 0;
    int64_t N = tensorType.getShape()[0];
    int64_t ub = (N - 1) / 2;
    int64_t step = 1;

    DEBUG_PRINT_NO_ARGS();
    LowPassFIRFilterOpAdaptor lowPassfirFilterOpAdaptor(operands);
    // Handle middle y[mid] = wc / pi
    int64_t midIndx = ub;
    Value constantIndxMid =
        rewriter.create<arith::ConstantIndexOp>(loc, midIndx);
    // rewriter.create<AffineStoreOp>(loc, constant0, alloc,
    // ValueRange{constantIndx0});
    Value wc = rewriter.create<AffineLoadOp>(
        loc, lowPassfirFilterOpAdaptor.getWc(), ValueRange{});

    Value constpi = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(3.14159265359));
    Value wcByPi = rewriter.create<arith::DivFOp>(loc, wc, constpi);

    rewriter.create<AffineStoreOp>(loc, wcByPi, alloc,
                                   ValueRange{constantIndxMid});

    // first from 0 <= i <= (N-1)/2 - 1

    // calculate i-(N-1)/2
    Value Nminus1By2 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr((float)ub));
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());
    // convert index to f64
    Value IndxY = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivY);
    Value i =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxY);

    // get sin(wc * (i - (N-1)/ 2))
    Value iMinusMid = rewriter.create<arith::SubFOp>(loc, i, Nminus1By2);
    Value mulwc_iMinusMid = rewriter.create<arith::MulFOp>(loc, wc, iMinusMid);

    Value GetSin = rewriter.create<math::SinOp>(loc, mulwc_iMinusMid);

    // get sin(wc*i) / pi * i

    Value piMuliMinusMid =
        rewriter.create<arith::MulFOp>(loc, constpi, iMinusMid);
    Value GetDiv = rewriter.create<arith::DivFOp>(loc, GetSin, piMuliMinusMid);
    rewriter.create<AffineStoreOp>(loc, GetDiv, alloc, ValueRange{ivY});
    // llvm::errs() << "LINE " << __LINE__ << " file= " << __FILE__ << "\n" ;
    rewriter.setInsertionPointAfter(forOpY);

    // 2nd loop from (N-1)/2 + 1 <= i < N
    lb = ub + 1;
    ub = N;

    affine::AffineForOp forOp1 =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto iv1 = forOp1.getInductionVar();
    rewriter.setInsertionPointToStart(forOp1.getBody());
    // convert index to f64
    Value Indx1 = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), iv1);
    Value i1 =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), Indx1);

    // get sin(wc * (i1 - (N-1)/ 2))
    Value iMinusMid1 = rewriter.create<arith::SubFOp>(loc, i1, Nminus1By2);
    Value mulwc_iMinusMid1 =
        rewriter.create<arith::MulFOp>(loc, wc, iMinusMid1);
    Value GetSin1 = rewriter.create<math::SinOp>(loc, mulwc_iMinusMid1);

    // get sin(i1 - (N-1)/ 2) / (i1 - (N-1)/ 2) * pi
    //  get sin(wc*i1) / pi * i1

    Value piMuliMinusMid1 =
        rewriter.create<arith::MulFOp>(loc, constpi, iMinusMid1);
    Value GetDiv1 =
        rewriter.create<arith::DivFOp>(loc, GetSin1, piMuliMinusMid1);
    rewriter.create<AffineStoreOp>(loc, GetDiv1, alloc, ValueRange{iv1});
    // llvm::errs() << "LINE " << __LINE__ << " file= " << __FILE__ << "\n" ;
    rewriter.setInsertionPointAfter(forOp1);

    // debug
    //  forOpX->dump();
    //  forOpY->dump();

    // %cst = arith.constant 6.2831853071800001 : f64
    // %cst_0 = arith.constant 4.600000e-01 : f64
    // %cst_1 = arith.constant 5.400000e-01 : f64
    // %cst_2 = arith.constant 4.000000e+00 : f64
    // %alloc = memref.alloc() : memref<4xf64>
    // %alloc_3 = memref.alloc() : memref<f64>
    // affine.store %cst_2, %alloc_3[] : memref<f64>
    // affine.for %arg0 = 0 to 4 {
    //   %0 = arith.index_castui %arg0 : index to i32
    //   %1 = arith.uitofp %0 : i32 to f64
    //   %2 = arith.mulf %1, %cst : f64
    //   %3 = arith.divf %2, %cst_2 : f64
    //   %4 = math.cos %3 : f64
    //   %5 = arith.mulf %4, %cst_0 : f64
    //   %6 = arith.subf %cst_1, %5 : f64
    //   affine.store %6, %alloc[%arg0] : memref<4xf64>
    // }

    // }
    // }
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: SetElemAtIndx operations
//===----------------------------------------------------------------------===//

struct SetElemAtIndxOpLowering : public ConversionPattern {
  SetElemAtIndxOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::SetElemAtIndxOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //   output = input[index]

    // replace this upsampling op with the output_mem_allocation op

    DEBUG_PRINT_NO_ARGS();

    // output for result type
    SetElemAtIndxOpAdaptor setElemAtIndxAdaptor(operands);
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    // auto tensorType =
    // llvm::cast<RankedTensorType>(setElemAtIndxAdaptor.getInput());
    // iterate to result1 --not needed for now but for future reference

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // For loop -- iterate from 1 to last
    //  int64_t lb = 0 ;
    //  int64_t ub = tensorType.getShape()[0];
    //  int64_t step = 1;
    //  affine::AffineForOp forOpY = rewriter.create<AffineForOp>(loc, lb, ub,
    //  step); auto ivY = forOpY.getInductionVar();
    //  rewriter.setInsertionPointToStart(forOpY.getBody());

    // Value inputX = rewriter.create<AffineLoadOp>(loc,
    // setElemAtIndxAdaptor.getInput(), ValueRange{ivY});
    // rewriter.create<AffineStoreOp>(loc, inputX, alloc, ValueRange{ivY});

    // rewriter.setInsertionPointAfter(forOpY);
    DEBUG_PRINT_WITH_ARGS("\nCheck for index --here");
    // load from X, using 2nd operand as index

    // Value GetValueAtIndx2ndArg = setElemAtIndxAdaptor.getIndx(); //
    // getOperand(1);
    DEBUG_PRINT_NO_ARGS();
    Value GetValueAtIndx2ndArg = op->getOperand(1);
    dsp::ConstantOp constantOp2ndArg =
        GetValueAtIndx2ndArg.getDefiningOp<dsp::ConstantOp>();
    DenseElementsAttr constantRhsValue = constantOp2ndArg.getValue();
    ;
    auto elements = constantRhsValue.getValues<FloatAttr>();
    float SecondValue = elements[0].getValueAsDouble();
    int SecondValueInt = (int64_t)SecondValue;
    DEBUG_PRINT_WITH_ARGS("Indx is", SecondValueInt);

    Value constantIndx2Indx =
        rewriter.create<arith::ConstantIndexOp>(loc, SecondValueInt);
    Value constantIndx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    // Value constant0 = rewriter.create<arith::ConstantOp>(loc,
    // rewriter.getF64Type(),
    //                                                      rewriter.getF64FloatAttr(15));

    // Value ValToStore = setElemAtIndxAdaptor.getVal();
    // Value ValToStore = op->getOperand(2);
    Value ValToStore = rewriter.create<AffineLoadOp>(
        loc, setElemAtIndxAdaptor.getVal(), ValueRange{constantIndx0});
    // Value ValToStore = rewriter.create<AffineLoadOp>(loc,
    // setElemAtIndxAdaptor.getVal(), ValueRange{});

    // rewriter.create<AffineStoreOp>(loc, constant0, alloc,
    // ValueRange{constantIndx2Indx});
    rewriter.create<AffineStoreOp>(loc, ValToStore,
                                   setElemAtIndxAdaptor.getInput(),
                                   ValueRange{constantIndx2Indx});

    // debug
    //  forOpY->dump();
    //  affine.store %cst, %alloc_10[] : memref<f64>
    //  %0 = affine.load %alloc_11[4] : memref<10xf64>
    //  affine.store %0, %alloc[0] : memref<1xf64>

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: GetElemAtIndx operations
//===----------------------------------------------------------------------===//

struct GetElemAtIndxOpLowering : public ConversionPattern {
  GetElemAtIndxOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::GetElemAtIndxOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //   output = input[index]

    // replace this upsampling op with the output_mem_allocation op

    DEBUG_PRINT_NO_ARGS();

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    // iterate to result1 --not needed for now but for future reference

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    // auto memRefType2 = convertTensorToMemRef(tensorType1);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // Value constant0 = rewriter.create<arith::ConstantOp>(loc,
    // rewriter.getF64Type(),
    //                                                      rewriter.getF64FloatAttr(0));

    DEBUG_PRINT_WITH_ARGS("\nCheck for index --here");
    // load from X, using 2nd operand as index
    GetElemAtIndxOpAdaptor getElemAtIndxAdaptor(operands);
    // Value GetValueAtIndx2ndArg = getElemAtIndxAdaptor.getIndx(); //
    // getOperand(1);
    DEBUG_PRINT_NO_ARGS();
    Value GetValueAtIndx2ndArg = op->getOperand(1);
    dsp::ConstantOp constantOp2ndArg =
        GetValueAtIndx2ndArg.getDefiningOp<dsp::ConstantOp>();
    DenseElementsAttr constantRhsValue = constantOp2ndArg.getValue();
    ;
    auto elements = constantRhsValue.getValues<FloatAttr>();
    float SecondValue = elements[0].getValueAsDouble();
    int SecondValueInt = (int64_t)SecondValue;
    DEBUG_PRINT_WITH_ARGS("Indx is", SecondValueInt);

    Value constantIndx2Indx =
        rewriter.create<arith::ConstantIndexOp>(loc, SecondValueInt);
    Value constantIndx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    Value inputX = rewriter.create<AffineLoadOp>(
        loc, getElemAtIndxAdaptor.getInput(), ValueRange{constantIndx2Indx});
    rewriter.create<AffineStoreOp>(loc, inputX, alloc,
                                   ValueRange{constantIndx0});

    // debug
    //  forOpX->dump();
    //  forOpY->dump();
    //  affine.store %cst, %alloc_10[] : memref<f64>
    //  %0 = affine.load %alloc_11[4] : memref<10xf64>
    //  affine.store %0, %alloc[0] : memref<1xf64>

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: SincOp operations
//===----------------------------------------------------------------------===//

struct SincOpLowering : public ConversionPattern {
  SincOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::SincOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //   y = sinc(wc * n) = [1, sin(wc)/pi , sin(2* wc)/2*pi , ... sin(n *
    //   wc)/n*pi] , 0<=n<=N

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // For loop -- iterate from 1 to last
    int64_t lb = 1;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    DEBUG_PRINT_NO_ARGS();
    // get constants -- 0.54 & 0.46
    Value constantIndx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    // rewriter.create<AffineStoreOp>(loc, constant0, alloc,
    // ValueRange{constantIndx0});

    Value constant1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(1));
    Value constpi = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(3.14159265359));
    rewriter.create<AffineStoreOp>(loc, constant1, alloc,
                                   ValueRange{constantIndx0});

    // For loop
    SincOpAdaptor sincOpAdaptor(operands);
    // loop for Y
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());
    // convert index to f64
    Value IndxY = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivY);
    Value i =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxY);

    // get wc * i
    Value wc =
        rewriter.create<AffineLoadOp>(loc, sincOpAdaptor.getWc(), ValueRange{});

    Value mulwc_i = rewriter.create<arith::MulFOp>(loc, wc, i);

    // get sin(wc*i) / pi * i
    Value GetSin = rewriter.create<math::SinOp>(loc, mulwc_i);
    Value piMuli = rewriter.create<arith::MulFOp>(loc, constpi, i);
    Value GetDiv = rewriter.create<arith::DivFOp>(loc, GetSin, piMuli);
    rewriter.create<AffineStoreOp>(loc, GetDiv, alloc, ValueRange{ivY});
    // llvm::errs() << "LINE " << __LINE__ << " file= " << __FILE__ << "\n" ;
    rewriter.setInsertionPointAfter(forOpY);
    // debug
    //  forOpX->dump();
    //  forOpY->dump();

    // %cst = arith.constant 6.2831853071800001 : f64
    // %cst_0 = arith.constant 4.600000e-01 : f64
    // %cst_1 = arith.constant 5.400000e-01 : f64
    // %cst_2 = arith.constant 4.000000e+00 : f64
    // %alloc = memref.alloc() : memref<4xf64>
    // %alloc_3 = memref.alloc() : memref<f64>
    // affine.store %cst_2, %alloc_3[] : memref<f64>
    // affine.for %arg0 = 0 to 4 {
    //   %0 = arith.index_castui %arg0 : index to i32
    //   %1 = arith.uitofp %0 : i32 to f64
    //   %2 = arith.mulf %1, %cst : f64
    //   %3 = arith.divf %2, %cst_2 : f64
    //   %4 = math.cos %3 : f64
    //   %5 = arith.mulf %4, %cst_0 : f64
    //   %6 = arith.subf %cst_1, %5 : f64
    //   affine.store %6, %alloc[%arg0] : memref<4xf64>
    // }

    // }
    // }
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: FFT1DImg operations
//===----------------------------------------------------------------------===//

struct FFT1DImgOpLowering : public ConversionPattern {
  FFT1DImgOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::FFT1DImgOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //   y[k] = y_real[k] + j *y_img[k]
    //  y_img = sumOver_n(x[n]*sin[2*pi * k *n/N ] * -1
    // init  output mem for y_real & y_img as 0
    // iterate for output from k=0 to last
    // iterate for all x from n=0 to last
    // perform the calculations : ie x[n] * cos[2*pi * k *n/N ] and sum and
    // store them at y[k]
    //
    // replace this upsampling op with the output_mem_allocation op

    // DEBUG_PRINT_NO_ARGS() ;

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    // iterate to result1 --not needed for now but for future reference
    //  auto tensorType1 =
    //  llvm::cast<RankedTensorType>(*std::next(op->result_type_begin(), 1));

    // DEBUG_PRINT_NO_ARGS() ;
    // tensorType.getShape()[0]
    // llvm::errs() << "tensorType1.getShape()[0] " << tensorType1.getShape()[0]
    // << " func= " << __func__ << "\n";

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    // auto memRefType2 = convertTensorToMemRef(tensorType1);
    auto alloc_img = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // affine.for %y = 0 to 4 {
    //     affine.store %cst_3, %alloc_real[%y] : memref<4xf64>
    //     affine.store %cst_3, %alloc_img[%y] : memref<4xf64>
    // }
    Value constant0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

    // For loop -- iterate from 1 to last
    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    affine::AffineForOp forOp1 =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto iv = forOp1.getInductionVar();
    rewriter.setInsertionPointToStart(forOp1.getBody());
    rewriter.create<AffineStoreOp>(loc, constant0, alloc_img, ValueRange{iv});
    rewriter.setInsertionPointAfter(forOp1);

    // loop for Y
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());

    // loop for X
    affine::AffineForOp forOpX =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivX = forOpX.getInductionVar();
    rewriter.setInsertionPointToStart(forOpX.getBody());

    // load from X, & y1 & y2
    FFT1DImgOpAdaptor fft1DImgAdaptor(operands);
    Value inputX = rewriter.create<AffineLoadOp>(
        loc, fft1DImgAdaptor.getInput(), ValueRange{ivX});
    Value loadYImg =
        rewriter.create<AffineLoadOp>(loc, alloc_img, ValueRange{ivY});

    // convert index to f64
    Value IndxY = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivY);
    Value k =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxY);

    Value IndxX = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivX);
    Value i =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxX);

    // get 2*pi * k * i / N
    Value muli_k = rewriter.create<arith::MulFOp>(loc, k, i);

    Value const2pi = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(6.28318530718));
    Value mul2piKI = rewriter.create<arith::MulFOp>(loc, const2pi, muli_k);

    // getOperand().getType()
    // auto inputTensorType =
    // llvm::cast<RankedTensorType>(op->getOperand(0).getType());
    float LengthOfInput = (float)ub;
    Value N = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(LengthOfInput));
    // Value N = inputTensorType.getShape()[0];

    Value divIndxByN = rewriter.create<arith::DivFOp>(loc, mul2piKI, N);

    // Img part = -1 * Sum(x[i] * sin(div) )
    Value GetSin = rewriter.create<math::SinOp>(loc, divIndxByN);
    Value xMulSin = rewriter.create<arith::MulFOp>(loc, inputX, GetSin);
    Value imgSum = rewriter.create<arith::SubFOp>(loc, loadYImg, xMulSin);

    // Value constMinus1 = rewriter.create<arith::ConstantOp>(loc,
    // rewriter.getF64Type(),
    //                                                      rewriter.getF64FloatAttr(-1));
    // Value NegImgSum = rewriter.create<arith::MulFOp>(loc, constMinus1 ,
    // imgSum);
    rewriter.create<AffineStoreOp>(loc, imgSum, alloc_img, ValueRange{ivY});
    // x[n-1]
    rewriter.setInsertionPointAfter(forOpX);
    // Calculate y[k] = 1/N * y[k]

    rewriter.setInsertionPointAfter(forOpY);
    // debug
    //  forOpX->dump();
    //  forOpY->dump();
    //  affine.for %y = 0 to 4 {
    //      affine.store %cst_3, %alloc_real[%y] : memref<4xf64>
    //      affine.store %cst_3, %alloc_img[%y] : memref<4xf64>
    //  }

    // affine.for %y = 0 to 4 {
    // //   %0 = affine.load %alloc_3[%arg0] : memref<4xf64>
    // //   affine.store %0, %alloc_real[%arg0] : memref<4xf64>
    // affine.for %x = 0 to 4 {
    //     // CAcluations
    //           %1 = affine.load %alloc_3[%x] : memref<4xf64>
    //           %2 = affine.load %alloc_real[%y] : memref<4xf64>
    //           %3 = affine.load %alloc_img[%y] : memref<4xf64>
    //           // index cast for multiply
    //           %4 = arith.index_castui %y : index to i32
    //           %k = arith.uitofp %4 : i32 to f64
    //           %6 = arith.index_castui %x : index to i32
    //           %i = arith.uitofp %6 : i32 to f64
    //         //   %8 = arith.index_castui %arg3 : index to i32
    //         //   %9 = arith.uitofp %8 : i32 to f64
    //         //   %10 = arith.index_castui %arg4 : index to i32
    //         //   %11 = arith.uitofp %10 : i32 to f64

    //           %mul_1 = arith.mulf %i, %k : f64
    //           %mul = arith.mulf %mul_1, %cst_2pi : f64
    //         //  ixk / N
    //           %div = arith.divf %mul, %N : f64
    //         //   cos of the above
    //           %res_cos = math.cos %div : f64
    //         //   %16 = arith.addf %14, %15 : f64
    //         //   %res_sin = arith.mulf %16, %cst_0 : f64

    //           %res_sin = math.sin %div : f64
    //           %real_prod = arith.mulf %1, %res_cos : f64
    //           %img_prod_1 = arith.mulf %1, %res_sin : f64
    //           %img_prod = arith.mulf %cst_5, %img_prod_1 : f64

    //           %real = arith.addf %2, %real_prod : f64
    //           %img = arith.addf %3, %img_prod : f64
    //           affine.store %real, %alloc_real[%y] : memref<4xf64>
    //         //    dsp.print %alloc_real : memref<4xf64>
    //           affine.store %img, %alloc_img[%y] : memref<4xf64>

    // }
    // }
    // rewriter.replaceOp(op, alloc_real);
    rewriter.replaceOp(op, alloc_img);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: FFT1DReal operations
//===----------------------------------------------------------------------===//

struct FFT1DRealOpLowering : public ConversionPattern {
  FFT1DRealOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::FFT1DRealOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //   y[k] = y_real[k] + j *y_img[k]
    //  y_real = sumOver_n(x[n]*cos[2*pi * k *n/N ]
    //  y_img = sumOver_n(x[n]*sin[2*pi * k *n/N ] * -1
    // init  output mem for y_real & y_img as 0
    // iterate for output from k=0 to last
    // iterate for all x from n=0 to last
    // perform the calculations : ie x[n] * cos[2*pi * k *n/N ] and sum and
    // store them at y[k]
    //
    // replace this upsampling op with the output_mem_allocation op

    // DEBUG_PRINT_NO_ARGS() ;

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    // iterate to result1 --not needed for now but for future reference
    //  auto tensorType1 =
    //  llvm::cast<RankedTensorType>(*std::next(op->result_type_begin(), 1));

    // DEBUG_PRINT_NO_ARGS() ;
    // tensorType.getShape()[0]
    // llvm::errs() << "tensorType1.getShape()[0] " << tensorType1.getShape()[0]
    // << " func= " << __func__ << "\n";

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    // auto memRefType2 = convertTensorToMemRef(tensorType1);
    auto alloc_real = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // affine.for %y = 0 to 4 {
    //     affine.store %cst_3, %alloc_real[%y] : memref<4xf64>
    //     affine.store %cst_3, %alloc_img[%y] : memref<4xf64>
    // }
    Value constant0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

    // For loop -- iterate from 1 to last
    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    affine::AffineForOp forOp1 =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto iv = forOp1.getInductionVar();
    rewriter.setInsertionPointToStart(forOp1.getBody());
    rewriter.create<AffineStoreOp>(loc, constant0, alloc_real, ValueRange{iv});
    rewriter.setInsertionPointAfter(forOp1);

    // loop for Y
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());

    // loop for X
    affine::AffineForOp forOpX =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivX = forOpX.getInductionVar();
    rewriter.setInsertionPointToStart(forOpX.getBody());

    // load from X, & y1 & y2
    FFT1DRealOpAdaptor fft1DrealAdaptor(operands);
    Value inputX = rewriter.create<AffineLoadOp>(
        loc, fft1DrealAdaptor.getInput(), ValueRange{ivX});
    Value loadYReal =
        rewriter.create<AffineLoadOp>(loc, alloc_real, ValueRange{ivY});

    // convert index to f64
    Value IndxY = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivY);
    Value k =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxY);

    Value IndxX = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivX);
    Value i =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxX);

    // get 2*pi * k * i / N
    Value muli_k = rewriter.create<arith::MulFOp>(loc, k, i);

    Value const2pi = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(6.28318530718));
    Value mul2piKI = rewriter.create<arith::MulFOp>(loc, const2pi, muli_k);

    // getOperand().getType()
    // auto inputTensorType =
    // llvm::cast<RankedTensorType>(op->getOperand(0).getType());
    float LengthOfInput = (float)ub;
    Value N = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(LengthOfInput));
    // Value N = inputTensorType.getShape()[0];

    Value divIndxByN = rewriter.create<arith::DivFOp>(loc, mul2piKI, N);

    // Real part = Sum(x[i] * cos(div) )
    Value GetCos = rewriter.create<math::CosOp>(loc, divIndxByN);
    Value xMulCos = rewriter.create<arith::MulFOp>(loc, inputX, GetCos);
    Value realSum = rewriter.create<arith::AddFOp>(loc, loadYReal, xMulCos);
    rewriter.create<AffineStoreOp>(loc, realSum, alloc_real, ValueRange{ivY});

    // DEBUG_PRINT_NO_ARGS() ;

    rewriter.setInsertionPointAfter(forOpX);
    // forOpX->dump();
    // rewriter.create<AffineYieldOp>(loc, ValueRange{alloc_real, alloc_img});
    rewriter.setInsertionPointAfter(forOpY);
    // debug
    //  forOpX->dump();
    //  forOpY->dump();
    //  affine.for %y = 0 to 4 {
    //      affine.store %cst_3, %alloc_real[%y] : memref<4xf64>
    //      affine.store %cst_3, %alloc_img[%y] : memref<4xf64>
    //  }

    // affine.for %y = 0 to 4 {
    // //   %0 = affine.load %alloc_3[%arg0] : memref<4xf64>
    // //   affine.store %0, %alloc_real[%arg0] : memref<4xf64>
    // affine.for %x = 0 to 4 {
    //     // CAcluations
    //           %1 = affine.load %alloc_3[%x] : memref<4xf64>
    //           %2 = affine.load %alloc_real[%y] : memref<4xf64>
    //           %3 = affine.load %alloc_img[%y] : memref<4xf64>
    //           // index cast for multiply
    //           %4 = arith.index_castui %y : index to i32
    //           %k = arith.uitofp %4 : i32 to f64
    //           %6 = arith.index_castui %x : index to i32
    //           %i = arith.uitofp %6 : i32 to f64
    //         //   %8 = arith.index_castui %arg3 : index to i32
    //         //   %9 = arith.uitofp %8 : i32 to f64
    //         //   %10 = arith.index_castui %arg4 : index to i32
    //         //   %11 = arith.uitofp %10 : i32 to f64

    //           %mul_1 = arith.mulf %i, %k : f64
    //           %mul = arith.mulf %mul_1, %cst_2pi : f64
    //         //  ixk / N
    //           %div = arith.divf %mul, %N : f64
    //         //   cos of the above
    //           %res_cos = math.cos %div : f64
    //         //   %16 = arith.addf %14, %15 : f64
    //         //   %res_sin = arith.mulf %16, %cst_0 : f64

    //           %res_sin = math.sin %div : f64
    //           %real_prod = arith.mulf %1, %res_cos : f64
    //           %img_prod_1 = arith.mulf %1, %res_sin : f64
    //           %img_prod = arith.mulf %cst_5, %img_prod_1 : f64

    //           %real = arith.addf %2, %real_prod : f64
    //           %img = arith.addf %3, %img_prod : f64
    //           affine.store %real, %alloc_real[%y] : memref<4xf64>
    //         //    dsp.print %alloc_real : memref<4xf64>
    //           affine.store %img, %alloc_img[%y] : memref<4xf64>

    // }
    // }
    // rewriter.replaceOp(op, alloc_real);
    rewriter.replaceOp(op, alloc_real);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: SquareOp operations
//===----------------------------------------------------------------------===//

struct SquareOpLowering : public ConversionPattern {
  SquareOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::SquareOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    // output = 0
    // iterate for len = 0 to inputLen
    //   elem = a[i]
    //   output[i] = elem * elem
    //   store output

    // DEBUG_PRINT_NO_ARGS() ;

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // For loop
    SquareOpAdaptor squareOpAdaptor(operands);
    // DEBUG_PRINT_NO_ARGS() ;

    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    // for loop
    affine::AffineForOp forOp1 =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto iv = forOp1.getInductionVar();

    rewriter.setInsertionPointToStart(forOp1.getBody());

    // DEBUG_PRINT_NO_ARGS() ;
    Value elemIn =
        rewriter.create<AffineLoadOp>(loc, squareOpAdaptor.getInput(), iv);
    Value square = rewriter.create<arith::MulFOp>(loc, elemIn, elemIn);

    // store the result
    rewriter.create<AffineStoreOp>(loc, square, alloc, iv);

    rewriter.setInsertionPointAfter(forOp1);
    // debug
    //  forOp1->dump();
    //    affine.for %arg0 = 0 to 5 {
    //    %0 = affine.load %alloc_6[%arg0] : memref<5xf64>
    //    %1 = arith.mulf %0, %0 : f64
    //    affine.store %1, %alloc_5[%arg0] : memref<5xf64>
    //  }
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: SumOp operations
//===----------------------------------------------------------------------===//

struct SumOpLowering : public ConversionPattern {
  SumOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::SumOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    // output = 0
    // iterate for len = 0 to inputLen
    //   output = load output
    //   elem = a[i]
    //   output = output + elem
    //   store output

    // DEBUG_PRINT_NO_ARGS() ;

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // For loop
    SumOpAdaptor sumOpAdaptor(operands);
    // DEBUG_PRINT_NO_ARGS() ;
    auto inputType = llvm::dyn_cast<RankedTensorType>(
        op->getOperand(0).getType()); // op->getOperand(
    // auto inputType =
    // llvm::dyn_cast<RankedTensorType>(sumOpAdaptor.getInput().getType());
    // DEBUG_PRINT_NO_ARGS() ;

    int64_t lb = 0;
    int64_t ub = inputType.getShape()[0];
    int64_t step = 1;

    // init 0 for output
    Value constantIndx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    // Value GetInputX0 = rewriter.create<AffineLoadOp>(loc,
    // lowPassFilterAdaptor.getLhs(), /* iv */ ValueRange{constantIndx0});
    Value constant0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
    // Value elemIn = rewriter.create<AffineLoadOp>(loc,
    // upsamplingAdaptor.getLhs(), iv); DEBUG_PRINT_NO_ARGS() ;
    rewriter.create<AffineStoreOp>(loc, constant0, alloc,
                                   ValueRange{constantIndx0});

    affine::AffineForOp forOp1 =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto iv = forOp1.getInductionVar();

    rewriter.setInsertionPointToStart(forOp1.getBody());

    // DEBUG_PRINT_NO_ARGS() ;
    Value elemIn =
        rewriter.create<AffineLoadOp>(loc, sumOpAdaptor.getInput(), iv);
    Value loadSum =
        rewriter.create<AffineLoadOp>(loc, alloc, ValueRange{constantIndx0});

    Value sum = rewriter.create<arith::AddFOp>(loc, elemIn, loadSum);

    // store the result
    rewriter.create<AffineStoreOp>(loc, sum, alloc, ValueRange{constantIndx0});

    rewriter.setInsertionPointAfter(forOp1);
    // debug
    //  forOp1->dump();
    //    %cont3 = arith.const 3 : f64
    //    affine.for %arg0 = 0 to 8 {
    //     %elem1 = affine.load input[%arg0]
    //     #map1 = affine_map<(%arg0)[] : (%arg0 + 1)
    //     #map2 = affine_map<(%arg0)[] : (%arg0 + 2)
    //     %elem2 = affine.load input[#map1] <-- affine apply
    //     %elem3 = affine.load input[#map2]

    //    %sum1 = arith.addf %elem1 , %elem2
    //    %sum2 = arith.addf %sum1, %elem3
    //    %res = arith.divf %sum2 ,
    //    affine.store %sum2, out[%arg0]
    // }
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: FIRFilterResponse operations
//===----------------------------------------------------------------------===//
struct filterOpLowering : public ConversionPattern {
  filterOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::filterOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // dsp.filterOp has 3 operands -- both of type tensor f64

    // Pseudo-code:
    //  y[i] = sum(b[j] * x(i-j) - a[j] *x[i-j] ) j=1 to i and  i=1 to len(x)
    //  also, y[0] = b[0] * x[0]

    // 1) calculate y[0]
    // 2) iterate for indx=1 to input_len:
    //     load y[indx] = b[0] * x[indx]
    //     3) iterate for j=1 to indx :
    //             load b[j] , x[i-j] , a[j] , y[i-j]
    //             y[indx] = y[indx] + b[j] * x[i-j] - a[j]*y[i-j]

    auto loc = op->getLoc();
    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
    filterOpAdaptor filterOpAdaptor1(operands);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // IR:
    // ConstantIndx0
    // b0 = affine.load(b, ConstantIndx0)
    // x0 = affine.load(x, ConstantIndx0)
    // tempY0 = arith.mulf(b0,x0)

    // lb = 1, ub = x.size() , ivY = forLoopY.inductionVariable()
    // forLoopY
    // xIvY = affine.load(x,ivY )
    // tempYIndx = affine.mulf(b0, xIvY)
    // affine.store(xIvY, y, ivY)

    //     forloopJ , ivJ = forloopJ.inductionVariable()
    //         //optional get min ivY and len(b) -- iterate for this
    //         load (b,ivJ) ; (x, map(ivY - ivJ)) , (a, ivJ) ,
    //         (y, map(ivY - ivJ) ), (y , ivJ)

    //         tempBxX = arith.mulf(b , x)
    //         tempAxY = arith.mulf(a , Y_i-j)
    //         tempB_A = arith.subf( tempBxX - tempAxY)
    //         sumY_A = arith.addf( Y , tempB_A )
    //         affine.store(sumY_A , y , ivY)

    // ConstantIndx0
    // b0 = affine.load(b, ConstantIndx0)
    // x0 = affine.load(x, ConstantIndx0)
    // tempY0 = arith.mulf(b0,x0)

    Value constantIndx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value b0 = rewriter.create<affine::AffineLoadOp>(
        loc, filterOpAdaptor1.getB(), ValueRange{constantIndx0});
    Value x0 = rewriter.create<affine::AffineLoadOp>(
        loc, filterOpAdaptor1.getX(), ValueRange{constantIndx0});
    Value tempY0 = rewriter.create<arith::MulFOp>(loc, b0, x0);

    // store at Y0
    rewriter.create<affine::AffineStoreOp>(loc, tempY0, alloc,
                                           ValueRange{constantIndx0});

    // For loop -- iterate from 1 to last
    //  lb = 1, ub = x.size() , ivY = forLoopY.inductionVariable()
    //      forLoopY
    //      xIvY = affine.load(x,ivY )
    //      tempYIndx = affine.mulf(b0, xIvY)
    //      affine.store(tempYIndx, y, ivY)

    int64_t lb = 1;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    // DEBUG_PRINT_NO_ARGS() ;

    // loop for Y
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());

    Value xIvY = rewriter.create<affine::AffineLoadOp>(
        loc, filterOpAdaptor1.getX(), ivY);
    Value b0mulxIvY = rewriter.create<arith::MulFOp>(loc, b0, xIvY);
    rewriter.create<affine::AffineStoreOp>(loc, b0mulxIvY, alloc, ivY);

    // loop for X-- 1 to upperIndx ie, ivY
    //  forloopJ , ivJ = forloopJ.inductionVariable()
    //  //optional get min ivY and len(b) -- iterate for this
    //  load (b,ivJ) ; (x, map(ivY - ivJ)) , (a, ivJ) ,
    //  (y, map(ivY - ivJ) ), (y , ivJ)

    // tempBxX = arith.mulf(b , x)
    // tempAxY = arith.mulf(a , Y_i-j)
    // tempB_A = arith.subf( tempBxX - tempAxY)
    // sumY_A = arith.addf( Y , tempB_A )
    // affine.store(sumY_A , y , ivY)

    // look for here
    //  DEBUG_PRINT_NO_ARGS() ;
    // Future -- try to loop
    //  Value forlb = rewriter.create<arith::ConstantIndexOp>(loc, 1);
    AffineExpr expr0;
    bindDims(rewriter.getContext(), expr0);
    // AffineMap lbMap = AffineMap::get(1, 0, expr0);

    // affine::AffineForOp forOpJ = rewriter.create<AffineForOp>(loc, lbMap,
    // ValueRange{forlb} ,lbMap , ValueRange{ivY}, step);
    affine::AffineForOp forOpJ =
        rewriter.create<AffineForOp>(loc, lb, ub, step);

    auto ivJ = forOpJ.getInductionVar();
    rewriter.setInsertionPointToStart(forOpJ.getBody());

    // load from X, & Y
    //  DCTOpAdaptor dctAdaptor(operands);
    // For affine expression: #map1 = affine_map<(%ivY , ivJ)[] : (%ivY - ivJ)
    AffineExpr d0, d1, s0;
    bindDims(rewriter.getContext(), d0, d1);
    // AffineExpr ExprForIndxYminusX = rewriter.getAffineDimExpr(0) -
    // rewriter.getAffineDimExpr(1); //d0 - d1;
    AffineExpr ExprForIndxYminusX = d0 - d1;

    AffineMap addMapForYminusX = AffineMap::get(2, 0, ExprForIndxYminusX);

    // load (b,ivJ) ; (x, map(ivY - ivJ)) , (a, ivJ) ,
    // (y, map(ivY - ivJ) ), (y , ivJ)
    Value inputX = rewriter.create<AffineLoadOp>(
        loc, filterOpAdaptor1.getX(), addMapForYminusX, ValueRange{ivY, ivJ});
    Value inputB = rewriter.create<AffineLoadOp>(loc, filterOpAdaptor1.getB(),
                                                 ValueRange{ivJ});
    Value inputA = rewriter.create<AffineLoadOp>(loc, filterOpAdaptor1.getA(),
                                                 ValueRange{ivJ});
    Value inputPrevY = rewriter.create<AffineLoadOp>(
        loc, alloc, addMapForYminusX, ValueRange{ivY, ivJ});
    Value outY = rewriter.create<AffineLoadOp>(loc, alloc, ValueRange{ivY});

    // tempBxX = arith.mulf(b , x)
    // tempAxY = arith.mulf(a , Y_i-j)
    // tempB_A = arith.subf( tempBxX - tempAxY)
    // sumY_A = arith.addf( Y , tempB_A )
    // affine.store(sumY_A , y , ivY)

    Value tempBxX = rewriter.create<arith::MulFOp>(loc, inputB, inputX);
    Value tempAxY = rewriter.create<arith::MulFOp>(loc, inputA, inputPrevY);
    Value tempBminusA = rewriter.create<arith::SubFOp>(loc, tempBxX, tempAxY);
    Value sumY_A = rewriter.create<arith::AddFOp>(loc, outY, tempBminusA);
    rewriter.create<affine::AffineStoreOp>(loc, sumY_A, alloc, ivY);

    rewriter.setInsertionPointAfter(forOpJ);
    rewriter.setInsertionPointAfter(forOpY);
    // forOpJ->dump();

    // debug
    //  forOpJ->dump();
    //  forOpY->dump();
    //  affine.for %y = 0 to 4 {
    //      affine.store %cst_3, %alloc[%y] : memref<4xf64>
    //      affine.store %cst_3, %alloc_img[%y] : memref<4xf64>
    //  }

    // affine.for %y = 0 to 4 {
    // //   %0 = affine.load %alloc_3[%arg0] : memref<4xf64>
    // //   affine.store %0, %alloc[%arg0] : memref<4xf64>
    // affine.for %x = 0 to 4 {
    //     // CAcluations
    //           %1 = affine.load %alloc_3[%x] : memref<4xf64>
    //           %2 = affine.load %alloc[%y] : memref<4xf64>
    //           %3 = affine.load %alloc_img[%y] : memref<4xf64>
    //           // index cast for multiply
    //           %4 = arith.index_castui %y : index to i32
    //           %k = arith.uitofp %4 : i32 to f64
    //           %6 = arith.index_castui %x : index to i32
    //           %i = arith.uitofp %6 : i32 to f64
    //         //   %8 = arith.index_castui %arg3 : index to i32
    //         //   %9 = arith.uitofp %8 : i32 to f64
    //         //   %10 = arith.index_castui %arg4 : index to i32
    //         //   %11 = arith.uitofp %10 : i32 to f64

    //           %mul_1 = arith.mulf %i, %k : f64
    //           %mul = arith.mulf %mul_1, %cst_2pi : f64
    //         //  ixk / N
    //           %div = arith.divf %mul, %N : f64
    //         //   cos of the above
    //           %res_cos = math.cos %div : f64
    //         //   %16 = arith.addf %14, %15 : f64
    //         //   %res_sin = arith.mulf %16, %cst_0 : f64

    //           %res_sin = math.sin %div : f64
    //           %real_prod = arith.mulf %1, %res_cos : f64
    //           %img_prod_1 = arith.mulf %1, %res_sin : f64
    //           %img_prod = arith.mulf %cst_5, %img_prod_1 : f64

    //           %real = arith.addf %2, %real_prod : f64
    //           %img = arith.addf %3, %img_prod : f64
    //           affine.store %real, %alloc[%y] : memref<4xf64>
    //         //    dsp.print %alloc : memref<4xf64>
    //           affine.store %img, %alloc_img[%y] : memref<4xf64>

    // }
    // }
    rewriter.replaceOp(op, alloc);
    // rewriter.replaceOp(op, ValueRange{alloc,alloc_img});

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: DCT operations
//===----------------------------------------------------------------------===//

struct DCTOpLowering : public ConversionPattern {
  DCTOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::DCTOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //   y[k] = sqrt(2/N) * SumOverAllN( x[n] cos(pi * k * (n +0.5)/N)) ,
    //   0<=n<=N-1 :
    //  for y[0] , the answer will be multiplied by 1/sqrt(2)

    // init  output mem for y as 0
    // iterate for output from k=0 to last
    // iterate for all x from n=0 to last
    // perform the calculations : ie x[n] cos(pi * k * (n +0.5)/N) and sum and
    // store them at y[k]
    //
    // replace this upsampling op with the output_mem_allocation op

    // DEBUG_PRINT_NO_ARGS() ;

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
    DCTOpAdaptor dctAdaptor(operands);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // constant values:
    const float sqrt2 = 1.41421356237;
    const float pi = 3.14159265358;

    // affine.for %y = 0 to 4 {
    //     affine.store %cst_3, %alloc[%y] : memref<4xf64>
    // }
    Value constant0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

    // For loop -- iterate from 0 to last
    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    affine::AffineForOp forOp1 =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto iv = forOp1.getInductionVar();
    rewriter.setInsertionPointToStart(forOp1.getBody());
    rewriter.create<AffineStoreOp>(loc, constant0, alloc, ValueRange{iv});
    rewriter.setInsertionPointAfter(forOp1);
    // DEBUG_PRINT_NO_ARGS() ;

    // loop for Y
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());

    // loop for X
    affine::AffineForOp forOpX =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivX = forOpX.getInductionVar();
    rewriter.setInsertionPointToStart(forOpX.getBody());

    // load from X, & Y
    //  DCTOpAdaptor dctAdaptor(operands);
    Value inputX = rewriter.create<AffineLoadOp>(loc, dctAdaptor.getInput(),
                                                 ValueRange{ivX});
    Value loadYReal =
        rewriter.create<AffineLoadOp>(loc, alloc, ValueRange{ivY});

    // convert index to f64
    Value IndxY = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivY);
    Value k =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxY);

    Value IndxX = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivX);
    Value i =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxX);

    // get pi * k * (i + 0.5) / N
    Value constant0_5 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0.5));

    Value add_i_half = rewriter.create<arith::AddFOp>(loc, i, constant0_5);
    Value muli_k = rewriter.create<arith::MulFOp>(loc, k, add_i_half);

    Value constpi = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(pi));
    Value mulpiKI_half = rewriter.create<arith::MulFOp>(loc, constpi, muli_k);

    // Get N
    // DEBUG_PRINT_NO_ARGS() ;
    float LengthOfInput = (float)ub;
    Value N = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(LengthOfInput));

    Value divIndxByN = rewriter.create<arith::DivFOp>(loc, mulpiKI_half, N);

    // Get cos ( pi * k * (n +0.5)/N))
    // DEBUG_PRINT_NO_ARGS() ;
    Value GetCos = rewriter.create<math::CosOp>(loc, divIndxByN);
    Value xMulCos = rewriter.create<arith::MulFOp>(loc, inputX, GetCos);
    Value realSum = rewriter.create<arith::AddFOp>(loc, loadYReal, xMulCos);
    rewriter.create<AffineStoreOp>(loc, realSum, alloc, ValueRange{ivY});

    rewriter.setInsertionPointAfter(forOpX);

    // multiply Y(k) with sqrt(2) / sqrt(N)
    //  DEBUG_PRINT_NO_ARGS() ;
    Value loadYReal1 =
        rewriter.create<AffineLoadOp>(loc, alloc, ValueRange{ivY});
    Value constSqrt2 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(sqrt2));
    // Type floatType = rewriter.getF64Type();
    Value N2 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(LengthOfInput));
    // Define fast math flags
    // auto fastMathFlags = arith::FastMathFlagsAttr::get(
    //   rewriter.getContext(), arith::FastMathFlags::none);
    // arith::FastMathFlags::ApproximateSqrt |
    // arith::FastMathFlags::AllowReciprocal);
    Value sqrtN = rewriter.create<math::RsqrtOp>(loc, N2);
    // Value sqrtN = rewriter.create<math::RsqrtOp>(loc, TypeRange{ floatType }
    // , N2 , fastMathFlags );

    Value mulSqrt2ByN = rewriter.create<arith::MulFOp>(loc, constSqrt2, sqrtN);
    Value mulSqrt2ByNByY =
        rewriter.create<arith::MulFOp>(loc, mulSqrt2ByN, loadYReal1);
    // DEBUG_PRINT_NO_ARGS() ;
    rewriter.create<AffineStoreOp>(loc, mulSqrt2ByNByY, alloc, ValueRange{ivY});
    rewriter.setInsertionPointAfter(forOpY);

    // get Y0 multiplied by sqrt(2)
    Value constantIndx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value GetY0 = rewriter.create<AffineLoadOp>(
        loc, alloc, /* iv */ ValueRange{constantIndx0});
    Value valSqrt2 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(sqrt2));
    Value Y0MulSqrt2 = rewriter.create<arith::DivFOp>(loc, GetY0, valSqrt2);
    rewriter.create<AffineStoreOp>(loc, Y0MulSqrt2, alloc,
                                   ValueRange{constantIndx0});

    // debug
    //  forOpX->dump();
    //  forOpY->dump();
    //  affine.for %y = 0 to 4 {
    //      affine.store %cst_3, %alloc[%y] : memref<4xf64>
    //      affine.store %cst_3, %alloc_img[%y] : memref<4xf64>
    //  }

    // affine.for %y = 0 to 4 {
    // //   %0 = affine.load %alloc_3[%arg0] : memref<4xf64>
    // //   affine.store %0, %alloc[%arg0] : memref<4xf64>
    // affine.for %x = 0 to 4 {
    //     // CAcluations
    //           %1 = affine.load %alloc_3[%x] : memref<4xf64>
    //           %2 = affine.load %alloc[%y] : memref<4xf64>
    //           %3 = affine.load %alloc_img[%y] : memref<4xf64>
    //           // index cast for multiply
    //           %4 = arith.index_castui %y : index to i32
    //           %k = arith.uitofp %4 : i32 to f64
    //           %6 = arith.index_castui %x : index to i32
    //           %i = arith.uitofp %6 : i32 to f64
    //         //   %8 = arith.index_castui %arg3 : index to i32
    //         //   %9 = arith.uitofp %8 : i32 to f64
    //         //   %10 = arith.index_castui %arg4 : index to i32
    //         //   %11 = arith.uitofp %10 : i32 to f64

    //           %mul_1 = arith.mulf %i, %k : f64
    //           %mul = arith.mulf %mul_1, %cst_2pi : f64
    //         //  ixk / N
    //           %div = arith.divf %mul, %N : f64
    //         //   cos of the above
    //           %res_cos = math.cos %div : f64
    //         //   %16 = arith.addf %14, %15 : f64
    //         //   %res_sin = arith.mulf %16, %cst_0 : f64

    //           %res_sin = math.sin %div : f64
    //           %real_prod = arith.mulf %1, %res_cos : f64
    //           %img_prod_1 = arith.mulf %1, %res_sin : f64
    //           %img_prod = arith.mulf %cst_5, %img_prod_1 : f64

    //           %real = arith.addf %2, %real_prod : f64
    //           %img = arith.addf %3, %img_prod : f64
    //           affine.store %real, %alloc[%y] : memref<4xf64>
    //         //    dsp.print %alloc : memref<4xf64>
    //           affine.store %img, %alloc_img[%y] : memref<4xf64>

    // }
    // }
    rewriter.replaceOp(op, alloc);
    // rewriter.replaceOp(op, ValueRange{alloc,alloc_img});

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: HammingWindowOp operations
//===----------------------------------------------------------------------===//

struct HammingWindowOpLowering : public ConversionPattern {
  HammingWindowOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::HammingWindowOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //   y[k] = 0.54 - 0.46 cos(2 *pi * k/N-1) , 0<=n<N
    // DEBUG_PRINT_NO_ARGS() ;

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    // llvm::errs() << "tensorType " << tensorType.get;
    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // For loop -- iterate from 1 to last
    DEBUG_PRINT_NO_ARGS();
    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    DEBUG_PRINT_NO_ARGS();
    // get constants -- 0.54 & 0.46
    Value constant0_54 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0.54));
    Value constant0_46 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0.46));
    Value const2pi = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(6.28318530718));

    // loop for Y
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());
    // convert index to f64
    Value IndxY = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivY);
    Value k =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxY);

    // get 2*pi * k / (N -1)
    Value mul2pi_k = rewriter.create<arith::MulFOp>(loc, const2pi, k);

    // getOperand().getType()
    // auto inputTensorType =
    // llvm::cast<RankedTensorType>(op->getOperand(0).getType());
    float LengthOfInput = (float)ub;
    Value NMinus1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(),
        rewriter.getF64FloatAttr(LengthOfInput - 1));

    Value divIndxByNMinus1 =
        rewriter.create<arith::DivFOp>(loc, mul2pi_k, NMinus1);

    // get cos(2*pi * k/(N-1)
    Value GetCos = rewriter.create<math::CosOp>(loc, divIndxByNMinus1);
    Value MulCos0_46 =
        rewriter.create<arith::MulFOp>(loc, constant0_46, GetCos);
    Value Sub0_54_Cos =
        rewriter.create<arith::SubFOp>(loc, constant0_54, MulCos0_46);
    rewriter.create<AffineStoreOp>(loc, Sub0_54_Cos, alloc, ValueRange{ivY});
    DEBUG_PRINT_NO_ARGS();
    rewriter.setInsertionPointAfter(forOpY);
    // debug
    //  forOpX->dump();
    //  forOpY->dump();

    // %cst = arith.constant 6.2831853071800001 : f64
    // %cst_0 = arith.constant 4.600000e-01 : f64
    // %cst_1 = arith.constant 5.400000e-01 : f64
    // %cst_2 = arith.constant 4.000000e+00 : f64
    // %alloc = memref.alloc() : memref<4xf64>
    // %alloc_3 = memref.alloc() : memref<f64>
    // affine.store %cst_2, %alloc_3[] : memref<f64>
    // affine.for %arg0 = 0 to 4 {
    //   %0 = arith.index_castui %arg0 : index to i32
    //   %1 = arith.uitofp %0 : i32 to f64
    //   %2 = arith.mulf %1, %cst : f64
    //   %3 = arith.divf %2, %cst_2 : f64
    //   %4 = math.cos %3 : f64
    //   %5 = arith.mulf %4, %cst_0 : f64
    //   %6 = arith.subf %cst_1, %5 : f64
    //   affine.store %6, %alloc[%arg0] : memref<4xf64>
    // }

    // }
    // }
    rewriter.replaceOp(op, alloc);
    // rewriter.replaceOp(op, ValueRange{alloc,alloc_img});

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: IFFT1DOp operations
//===----------------------------------------------------------------------===//

struct IFFT1DOpLowering : public ConversionPattern {
  IFFT1DOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::IFFT1DOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //   y[k] = y_real[k] + j *y_img[k]
    //  y_real = sumOver_n(x[k]*cos[2*pi * k *n/N ]
    //  y_img = sumOver_n(x[k]*sin[2*pi * k *n/N ]
    //  here, x[k] is complex ie, x_real[k] + x_complex[k]
    // so, y[k] = sumOver_n(x[k]e^(2*pi * k *n/N))
    //  ==>   = x_real[k]cos(2*pi * k *n/N) - x_complex[k]sin(2*pi * k *n/N)

    // init  output mem for y_real
    // iterate for output from k=0 to last
    // iterate for all x from n=0 to last
    // perform the calculations : ie x_real[k]cos(2*pi * k *n/N) -
    // x_complex[k]sin(2*pi * k *n/N) and sum and store them at y[k]
    //

    DEBUG_PRINT_NO_ARGS();

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    // iterate to result1 --not needed for now but for future reference
    //  DEBUG_PRINT_NO_ARGS() ;

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc_real = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // affine.for %y = 0 to 4 {
    //     affine.store %cst_3, %alloc_real[%y] : memref<4xf64>
    //     affine.store %cst_3, %alloc_img[%y] : memref<4xf64>
    // }
    Value constant0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

    DEBUG_PRINT_NO_ARGS();
    // For loop -- iterate from 0 to last
    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    affine::AffineForOp forOp1 =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto iv = forOp1.getInductionVar();
    rewriter.setInsertionPointToStart(forOp1.getBody());
    rewriter.create<AffineStoreOp>(loc, constant0, alloc_real, ValueRange{iv});
    rewriter.setInsertionPointAfter(forOp1);

    // loop for Y
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());

    // loop for X
    affine::AffineForOp forOpX =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivX = forOpX.getInductionVar();
    rewriter.setInsertionPointToStart(forOpX.getBody());

    // load from X, & y1 & y2
    IFFT1DOpAdaptor ifft1DAdaptor(operands);
    Value inputReal = rewriter.create<AffineLoadOp>(
        loc, ifft1DAdaptor.getReal(), ValueRange{ivX});
    Value loadYReal =
        rewriter.create<AffineLoadOp>(loc, alloc_real, ValueRange{ivY});

    // convert index to f64
    Value IndxY = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivY);
    Value k =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxY);

    Value IndxX = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivX);
    Value i =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxX);

    // get 2*pi * k * i / N
    Value muli_k = rewriter.create<arith::MulFOp>(loc, k, i);

    Value const2pi = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(6.28318530718));
    Value mul2piKI = rewriter.create<arith::MulFOp>(loc, const2pi, muli_k);

    // getOperand().getType()
    // auto inputTensorType =
    // llvm::cast<RankedTensorType>(op->getOperand(0).getType());
    float LengthOfInput = (float)ub;
    Value N = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(LengthOfInput));
    // Value N = inputTensorType.getShape()[0];

    Value divIndxByN = rewriter.create<arith::DivFOp>(loc, mul2piKI, N);

    // Real Cos part = x_real[i] * cos(div)
    Value GetCos = rewriter.create<math::CosOp>(loc, divIndxByN);
    Value xMulCos = rewriter.create<arith::MulFOp>(loc, inputReal, GetCos);

    // Real Sin part =  x_complex[i] * sin(div)
    Value inputImg = rewriter.create<AffineLoadOp>(loc, ifft1DAdaptor.getImg(),
                                                   ValueRange{ivX});
    Value GetSin = rewriter.create<math::SinOp>(loc, divIndxByN);
    Value xMulSin = rewriter.create<arith::MulFOp>(loc, inputImg, GetSin);

    // Get real Ans = x_real[i] * cos(div) - x_complex[i] * sin(div)
    // Then sum over real_Ans by loading YReal
    Value realAns = rewriter.create<arith::SubFOp>(loc, xMulCos, xMulSin);
    Value realSum = rewriter.create<arith::AddFOp>(loc, loadYReal, realAns);
    rewriter.create<AffineStoreOp>(loc, realSum, alloc_real, ValueRange{ivY});

    // x[n-1]
    DEBUG_PRINT_NO_ARGS();
    // Value xMinusPrevX = rewriter.create<arith::SubFOp>(loc, inputX ,PrevX );

    rewriter.setInsertionPointAfter(forOpX);
    // Calculate y[k] = 1/N * y[k]
    Value loadY =
        rewriter.create<AffineLoadOp>(loc, alloc_real, ValueRange{ivY});
    // float LengthOfInput = (float) ub;
    Value N1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(LengthOfInput));
    Value SumDivByN = rewriter.create<arith::DivFOp>(loc, loadY, N1);
    rewriter.create<AffineStoreOp>(loc, SumDivByN, alloc_real, ValueRange{ivY});

    rewriter.setInsertionPointAfter(forOpY);
    // debug
    //  forOpX->dump();
    //  forOpY->dump();
    //  affine.for %y = 0 to 4 {
    //      affine.store %cst_3, %alloc_real[%y] : memref<4xf64>
    //      affine.store %cst_3, %alloc_img[%y] : memref<4xf64>
    //  }

    // affine.for %y = 0 to 4 {
    // //   %0 = affine.load %alloc_3[%arg0] : memref<4xf64>
    // //   affine.store %0, %alloc_real[%arg0] : memref<4xf64>
    // affine.for %x = 0 to 4 {
    //     // CAcluations
    //           %1 = affine.load %alloc_3[%x] : memref<4xf64>
    //           %2 = affine.load %alloc_real[%y] : memref<4xf64>
    //           %3 = affine.load %alloc_img[%y] : memref<4xf64>
    //           // index cast for multiply
    //           %4 = arith.index_castui %y : index to i32
    //           %k = arith.uitofp %4 : i32 to f64
    //           %6 = arith.index_castui %x : index to i32
    //           %i = arith.uitofp %6 : i32 to f64
    //         //   %8 = arith.index_castui %arg3 : index to i32
    //         //   %9 = arith.uitofp %8 : i32 to f64
    //         //   %10 = arith.index_castui %arg4 : index to i32
    //         //   %11 = arith.uitofp %10 : i32 to f64

    //           %mul_1 = arith.mulf %i, %k : f64
    //           %mul = arith.mulf %mul_1, %cst_2pi : f64
    //         //  ixk / N
    //           %div = arith.divf %mul, %N : f64
    //         //   cos of the above
    //           %res_cos = math.cos %div : f64
    //         //   %16 = arith.addf %14, %15 : f64
    //         //   %res_sin = arith.mulf %16, %cst_0 : f64

    //           %res_sin = math.sin %div : f64
    //           %real_prod = arith.mulf %1, %res_cos : f64
    //           %img_prod_1 = arith.mulf %1, %res_sin : f64
    //           %img_prod = arith.mulf %cst_5, %img_prod_1 : f64

    //           %real = arith.addf %2, %real_prod : f64
    //           %img = arith.addf %3, %img_prod : f64
    //           affine.store %real, %alloc_real[%y] : memref<4xf64>
    //         //    dsp.print %alloc_real : memref<4xf64>
    //           affine.store %img, %alloc_img[%y] : memref<4xf64>

    // }
    // }
    rewriter.replaceOp(op, alloc_real);
    // rewriter.replaceOp(op, ValueRange{alloc_real,alloc_img});

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: FFT1D operations
//===----------------------------------------------------------------------===//

struct FFT1DOpLowering : public ConversionPattern {
  FFT1DOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::FFT1DOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //   y[k] = y_real[k] + j *y_img[k]
    //  y_real = sumOver_n(x[n]*cos[2*pi * k *n/N ]
    //  y_img = sumOver_n(x[n]*sin[2*pi * k *n/N ] * -1
    // init  output mem for y_real & y_img as 0
    // iterate for output from k=0 to last
    // iterate for all x from n=0 to last
    // perform the calculations : ie x[n] * cos[2*pi * k *n/N ] and sum and
    // store them at y[k]
    //
    // replace this upsampling op with the output_mem_allocation op

    // DEBUG_PRINT_NO_ARGS() ;

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    // iterate to result1 --not needed for now but for future reference
    //  auto tensorType1 =
    //  llvm::cast<RankedTensorType>(*std::next(op->result_type_begin(), 1));

    // DEBUG_PRINT_NO_ARGS() ;
    // tensorType.getShape()[0]
    // llvm::errs() << "tensorType1.getShape()[0] " << tensorType1.getShape()[0]
    // << " func= " << __func__ << "\n";

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    // auto memRefType2 = convertTensorToMemRef(tensorType1);
    auto alloc_real = insertAllocAndDealloc(memRefType, loc, rewriter);
    auto alloc_img = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // affine.for %y = 0 to 4 {
    //     affine.store %cst_3, %alloc_real[%y] : memref<4xf64>
    //     affine.store %cst_3, %alloc_img[%y] : memref<4xf64>
    // }
    Value constant0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

    // For loop -- iterate from 1 to last
    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    affine::AffineForOp forOp1 =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto iv = forOp1.getInductionVar();
    rewriter.setInsertionPointToStart(forOp1.getBody());
    rewriter.create<AffineStoreOp>(loc, constant0, alloc_real, ValueRange{iv});
    rewriter.create<AffineStoreOp>(loc, constant0, alloc_img, ValueRange{iv});
    rewriter.setInsertionPointAfter(forOp1);

    // loop for Y
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());

    // loop for X
    affine::AffineForOp forOpX =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivX = forOpX.getInductionVar();
    rewriter.setInsertionPointToStart(forOpX.getBody());

    // load from X, & y1 & y2
    FFT1DOpAdaptor fft1DAdaptor(operands);
    Value inputX = rewriter.create<AffineLoadOp>(loc, fft1DAdaptor.getInput(),
                                                 ValueRange{ivX});
    Value loadYReal =
        rewriter.create<AffineLoadOp>(loc, alloc_real, ValueRange{ivY});
    Value loadYImg =
        rewriter.create<AffineLoadOp>(loc, alloc_img, ValueRange{ivY});

    // convert index to f64
    Value IndxY = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivY);
    Value k =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxY);

    Value IndxX = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), ivX);
    Value i =
        rewriter.create<arith::UIToFPOp>(loc, rewriter.getF64Type(), IndxX);

    // get 2*pi * k * i / N
    Value muli_k = rewriter.create<arith::MulFOp>(loc, k, i);

    Value const2pi = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(6.28318530718));
    Value mul2piKI = rewriter.create<arith::MulFOp>(loc, const2pi, muli_k);

    // getOperand().getType()
    // auto inputTensorType =
    // llvm::cast<RankedTensorType>(op->getOperand(0).getType());
    float LengthOfInput = (float)ub;
    Value N = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(LengthOfInput));
    // Value N = inputTensorType.getShape()[0];

    Value divIndxByN = rewriter.create<arith::DivFOp>(loc, mul2piKI, N);

    // Real part = Sum(x[i] * cos(div) )
    Value GetCos = rewriter.create<math::CosOp>(loc, divIndxByN);
    Value xMulCos = rewriter.create<arith::MulFOp>(loc, inputX, GetCos);
    Value realSum = rewriter.create<arith::AddFOp>(loc, loadYReal, xMulCos);
    rewriter.create<AffineStoreOp>(loc, realSum, alloc_real, ValueRange{ivY});

    // Img part = -1 * Sum(x[i] * sin(div) )
    Value GetSin = rewriter.create<math::SinOp>(loc, divIndxByN);
    Value xMulSin = rewriter.create<arith::MulFOp>(loc, inputX, GetSin);
    Value imgSum = rewriter.create<arith::SubFOp>(loc, loadYImg, xMulSin);

    // Value constMinus1 = rewriter.create<arith::ConstantOp>(loc,
    // rewriter.getF64Type(),
    //                                                      rewriter.getF64FloatAttr(-1));
    // Value NegImgSum = rewriter.create<arith::MulFOp>(loc, constMinus1 ,
    // imgSum);
    rewriter.create<AffineStoreOp>(loc, imgSum, alloc_img, ValueRange{ivY});
    // x[n-1]
    //  DEBUG_PRINT_NO_ARGS() ;
    //  Value xMinusPrevX = rewriter.create<arith::SubFOp>(loc, inputX ,PrevX );

    rewriter.setInsertionPointAfter(forOpX);
    // forOpX->dump();
    // rewriter.create<AffineYieldOp>(loc, ValueRange{alloc_real, alloc_img});
    rewriter.setInsertionPointAfter(forOpY);
    // debug
    //  forOpX->dump();
    //  forOpY->dump();
    //  affine.for %y = 0 to 4 {
    //      affine.store %cst_3, %alloc_real[%y] : memref<4xf64>
    //      affine.store %cst_3, %alloc_img[%y] : memref<4xf64>
    //  }

    // affine.for %y = 0 to 4 {
    // //   %0 = affine.load %alloc_3[%arg0] : memref<4xf64>
    // //   affine.store %0, %alloc_real[%arg0] : memref<4xf64>
    // affine.for %x = 0 to 4 {
    //     // CAcluations
    //           %1 = affine.load %alloc_3[%x] : memref<4xf64>
    //           %2 = affine.load %alloc_real[%y] : memref<4xf64>
    //           %3 = affine.load %alloc_img[%y] : memref<4xf64>
    //           // index cast for multiply
    //           %4 = arith.index_castui %y : index to i32
    //           %k = arith.uitofp %4 : i32 to f64
    //           %6 = arith.index_castui %x : index to i32
    //           %i = arith.uitofp %6 : i32 to f64
    //         //   %8 = arith.index_castui %arg3 : index to i32
    //         //   %9 = arith.uitofp %8 : i32 to f64
    //         //   %10 = arith.index_castui %arg4 : index to i32
    //         //   %11 = arith.uitofp %10 : i32 to f64

    //           %mul_1 = arith.mulf %i, %k : f64
    //           %mul = arith.mulf %mul_1, %cst_2pi : f64
    //         //  ixk / N
    //           %div = arith.divf %mul, %N : f64
    //         //   cos of the above
    //           %res_cos = math.cos %div : f64
    //         //   %16 = arith.addf %14, %15 : f64
    //         //   %res_sin = arith.mulf %16, %cst_0 : f64

    //           %res_sin = math.sin %div : f64
    //           %real_prod = arith.mulf %1, %res_cos : f64
    //           %img_prod_1 = arith.mulf %1, %res_sin : f64
    //           %img_prod = arith.mulf %cst_5, %img_prod_1 : f64

    //           %real = arith.addf %2, %real_prod : f64
    //           %img = arith.addf %3, %img_prod : f64
    //           affine.store %real, %alloc_real[%y] : memref<4xf64>
    //         //    dsp.print %alloc_real : memref<4xf64>
    //           affine.store %img, %alloc_img[%y] : memref<4xf64>

    // }
    // }
    // rewriter.replaceOp(op, alloc_real);
    rewriter.replaceOp(op, ValueRange{alloc_real, alloc_img});

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: HighPassFilter operations
//===----------------------------------------------------------------------===//

struct HighPassFilterOpLowering : public ConversionPattern {
  HighPassFilterOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::HighPassFilterOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    // init first value of output with first value of input: y[0] = x[0]
    // iterate for output from 1st to last
    // y[i] = x[i] - x[i -1 ]
    //  replace this upsampling op with the output_mem_allocation op

    DEBUG_PRINT_NO_ARGS();

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // Init y for the first index ie, index0
    Value constantIndx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    HighPassFilterOpAdaptor highPassFilterAdaptor(operands);
    Value GetInputX0 =
        rewriter.create<AffineLoadOp>(loc, highPassFilterAdaptor.getInput(),
                                      /* iv */ ValueRange{constantIndx0});
    rewriter.create<AffineStoreOp>(loc, GetInputX0, alloc,
                                   ValueRange{constantIndx0});

    // For loop -- iterate from 1 to last
    int64_t lb = 1;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    affine::AffineForOp forOp1 =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto iv = forOp1.getInductionVar();

    rewriter.setInsertionPointToStart(forOp1.getBody());

    // For affine expression: #map1 = affine_map<(%arg0)[] : (%arg0 - 1)
    AffineExpr d0, s0;
    bindDims(rewriter.getContext(), d0);
    AffineExpr ExprForPrevX = d0 - 1;
    AffineMap addMapForHighPassFilter = AffineMap::get(1, 0, ExprForPrevX);

    // x[n-1]
    DEBUG_PRINT_NO_ARGS();
    Value PrevX = rewriter.create<AffineLoadOp>(
        loc, highPassFilterAdaptor.getInput(), addMapForHighPassFilter,
        ValueRange{iv}); // memRefType
    // PrevX.dump();
    Value inputX = rewriter.create<AffineLoadOp>(
        loc, highPassFilterAdaptor.getInput(), ValueRange{iv});

    // get y[i] = x[i] - x[i -1 ]
    Value xMinusPrevX = rewriter.create<arith::SubFOp>(loc, inputX, PrevX);
    // Value cosRes = rewriter.create<math::CosOp>(loc, xMinusPrevX);
    rewriter.create<AffineStoreOp>(
        loc, xMinusPrevX, alloc,
        ValueRange{iv}); // PrevX //AddmulAlphaXAndPreYAlphaMinus1

    rewriter.setInsertionPointAfter(forOp1);
    // debug
    //  forOp1->dump();
    //  init first value of output with first value of input: y[0] = x[0]
    //  iterate for output from 1st to last
    //  y[i] = x[i] - x[i -1 ]
    //  replace this upsampling op with the output_mem_allocation op
    //   %indx0 = arith.constantIndex 0 : index
    //  %0 = affine.load in[indx0 ] : f64
    //   affine.store %0 ,out[indx0]
    //  affine.for %arg0 = 1 to len_y {
    //     #map1 = affine_map<(%arg0)[] : (%arg0 - 1)
    //     %1 = affine.load in[#map1]
    //      %load_in = affine.load in[%arg0]
    //      %2 = arith.subf %const1 , alpha
    //      affine.store %2, out[%arg0]
    // }
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: LowPassFilter operations
//===----------------------------------------------------------------------===//

struct LowPassFilter1stOrderOpLowering : public ConversionPattern {
  LowPassFilter1stOrderOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::LowPassFilter1stOrderOp::getOperationName(), 1,
                          ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    // init first value of output with first value of input: y[0] = x[0]
    // iterate for output from 1st to last
    // y[i] = (1 - alpha) * y[i-1] + alpha * x[i]
    //  replace this upsampling op with the output_mem_allocation op

    // DEBUG_PRINT_NO_ARGS() ;

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // Init y for the first index ie, index0
    Value constantIndx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    LowPassFilter1stOrderOpAdaptor lowPassFilterAdaptor(operands);
    Value GetInputX0 = rewriter.create<AffineLoadOp>(
        loc, lowPassFilterAdaptor.getLhs(), /* iv */ ValueRange{constantIndx0});
    rewriter.create<AffineStoreOp>(loc, GetInputX0, alloc,
                                   ValueRange{constantIndx0});

    // For loop -- iterate from 1 to last
    int64_t lb = 1;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    affine::AffineForOp forOp1 =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto iv = forOp1.getInductionVar();

    rewriter.setInsertionPointToStart(forOp1.getBody());

    // For affine expression: #map1 = affine_map<(%arg0)[] : (%arg0 - 1)
    AffineExpr d0, s0;
    bindDims(rewriter.getContext(), d0);
    AffineExpr ExprForPrevY = d0 - 1;
    AffineMap addMapForLowPassFilter = AffineMap::get(1, 0, ExprForPrevY);

    // y[n-1]
    //  DEBUG_PRINT_NO_ARGS() ;
    //  Value PrevY = rewriter.create<AffineLoadOp>(loc,
    //  lowPassFilterAdaptor.getLhs(), addMapForLowPassFilter,
    //                ValueRange{iv});
    //  Value PrevY = rewriter.create<AffineLoadOp>(loc,
    //  (*op->result_type_begin()), addMapForLowPassFilter,
    //                ValueRange{iv}); //memRefType
    Value PrevY = rewriter.create<AffineLoadOp>(
        loc, alloc, addMapForLowPassFilter, ValueRange{iv}); // memRefType
    // PrevY.dump();
    Value constant1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(1));
    // Value alpha = lowPassFilterAdaptor.getRhs(); //op->getOperand(1);
    Value alpha = rewriter.create<AffineLoadOp>(
        loc, lowPassFilterAdaptor.getRhs(), /* iv */ ValueRange{});
    // get y[n] = (1- alpha ) * y[n-1] + alpha * x[n]
    Value oneMinusAlpha = rewriter.create<arith::SubFOp>(loc, constant1, alpha);
    Value mulPrevYAlphaMinus1 =
        rewriter.create<arith::MulFOp>(loc, oneMinusAlpha, PrevY);

    Value inputX = rewriter.create<AffineLoadOp>(
        loc, lowPassFilterAdaptor.getLhs(), ValueRange{iv});
    Value mulAlphaX = rewriter.create<arith::MulFOp>(loc, alpha, inputX);

    Value AddmulAlphaXAndPreYAlphaMinus1 =
        rewriter.create<arith::AddFOp>(loc, mulPrevYAlphaMinus1, mulAlphaX);
    // DEBUG_PRINT_NO_ARGS() ;
    // AddmulAlphaXAndPreYAlphaMinus1.dump();
    // forOp1->dump();

    rewriter.create<AffineStoreOp>(
        loc, AddmulAlphaXAndPreYAlphaMinus1, alloc,
        ValueRange{iv}); // PrevY //AddmulAlphaXAndPreYAlphaMinus1

    rewriter.setInsertionPointAfter(forOp1);
    // debug
    //  forOp1->dump();
    //  init first value of output with first value of input: y[0] = x[0]
    //  iterate for output from 1st to last
    //  y[i] = (1 - alpha) * y[i-1] + alpha * x[i]
    //  replace this upsampling op with the output_mem_allocation op
    //   %indx0 = arith.constantIndex 0 : index
    //  %0 = affine.load in[indx0 ] : f64
    //   affine.store %0 ,out[indx0]
    //  affine.for %arg0 = 1 to len_y {
    //     #map1 = affine_map<(%arg0)[] : (%arg0 - 1)
    //     %1 = affine.load out[#map1]
    //      %2 = arith.subf %const1 , alpha
    //      %3 = arith.mulf %2 , %1

    //      %load_in = affine.load in[%arg0]
    //      %4 = arith.mulf alpha, %load_in
    //      %5 = arith.addf %4, %3
    //      affine.store %5, out[%arg0]
    // }
    //   %2ndOperand = arith.const 3 : f64
    //   affine.for %arg0 = 0 to input_len {
    //      %elem1 = affine.load input[%arg0] <-- affine apply
    //      #map1 = affine_map<(%arg0)[2ndOperand] : (%arg0 * 2ndOperand)
    //
    //      affine.store %elem1, out[#map1]
    // }
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Upsampling operations
//===----------------------------------------------------------------------===//

struct UpSamplingOpLowering : public ConversionPattern {
  UpSamplingOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::UpsamplingOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    // init all out values with 0 using affine loop
    // Update certain y_values with corresponding x
    // iterate for input : i = 0 to len
    // get the corresponding output mapping index = M * i
    //  store in y at that index
    //  replace this upsampling op with the output_mem_allocation op

    // DEBUG_PRINT_NO_ARGS() ;

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // For loop
    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    // init all the output mem location with 0
    affine::AffineForOp forOpSetOut0Loop =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivforOpSetOut0Loop = forOpSetOut0Loop.getInductionVar();

    rewriter.setInsertionPointToStart(forOpSetOut0Loop.getBody());
    Value constant0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
    // store the result
    rewriter.create<AffineStoreOp>(loc, constant0, alloc, ivforOpSetOut0Loop);
    rewriter.setInsertionPointAfter(forOpSetOut0Loop);

    Value upsampling2ndArg = op->getOperand(1);
    UpsamplingOpAdaptor upsamplingAdaptor(operands);
    auto inputType =
        llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    int64_t ub2 = inputType.getShape()[0]; // tensorType.getShape()[0];
    // create another for loop for updating corresponding y with x
    affine::AffineForOp forOp1 =
        rewriter.create<AffineForOp>(loc, lb, ub2, step);
    auto iv = forOp1.getInductionVar();

    rewriter.setInsertionPointToStart(forOp1.getBody());
    // Load input elem

    Value elemIn =
        rewriter.create<AffineLoadOp>(loc, upsamplingAdaptor.getLhs(), iv);

    // Value elemIn = rewriter.create<AffineLoadOp>(loc,
    // upsamplingAdaptor.getLhs(), addMapForUpSampling,
    //               ValueRange{iv,constantSamplingRateIndx});

    // For affine expression: #map1 = affine_map<(%arg0)[2ndOperand] : (%arg0 *
    // 2ndOperand)
    AffineExpr d0, s0;
    bindDims(rewriter.getContext(), d0);
    bindSymbols(rewriter.getContext(), s0);

    // AffineExpr ExprForUpSampling = rewriter.getAffineDimExpr(0) *
    // rewriter.getAffineSymbolExpr(0);
    AffineExpr ExprForUpSampling = d0 * s0;
    // Value constant3 = rewriter.create<arith::ConstantOp>(loc,
    // rewriter.getI64Type(),
    // rewriter.getIntegerAttr(rewriter.getIntegerType(64), 3));
    Value constant3 =
        rewriter.create<arith::ConstantIndexOp>(loc, 3); // working
    constant3.dump();

    int64_t SecondValueInt = 1;

    dsp::ConstantOp constantOp2ndArg =
        upsampling2ndArg.getDefiningOp<dsp::ConstantOp>();
    DenseElementsAttr constantRhsValue = constantOp2ndArg.getValue();
    ;
    auto elements = constantRhsValue.getValues<FloatAttr>();
    float SecondValue = elements[0].getValueAsDouble();
    SecondValueInt = (int64_t)SecondValue;

    // Value downSamplingRateAsIndex = rewriter.create<arith::IndexCastOp>(loc,
    // rewriter.getIndexType(),UpsamplingRate);
    Value constantSamplingRateIndx =
        rewriter.create<arith::ConstantIndexOp>(loc, SecondValueInt);
    constantSamplingRateIndx.dump();

    AffineMap addMapForUpSampling = AffineMap::get(1, 1, ExprForUpSampling);

    // DEBUG_PRINT_NO_ARGS() ;
    // Value elem2 = rewriter.create<AffineLoadOp>(loc,
    // upsamplingAdaptor.getLhs(), addMapForUpSampling,
    //               ValueRange{iv,constantSamplingRateIndx});
    // elem2.dump();
    // store the result
    rewriter.create<AffineStoreOp>(loc, elemIn, alloc, addMapForUpSampling,
                                   ValueRange{iv, constantSamplingRateIndx});

    rewriter.setInsertionPointAfter(forOp1);
    // debug
    //  forOp1->dump();
    //    %0 = arith.const 0 : f64
    //    affine.for %arg0 = 0 to out_y {
    //       affine.store %0, out[%arg0]
    //  }
    //    %2ndOperand = arith.const 3 : f64
    //    affine.for %arg0 = 0 to input_len {
    //       %elem1 = affine.load input[%arg0] <-- affine apply
    //       #map1 = affine_map<(%arg0)[2ndOperand] : (%arg0 * 2ndOperand)
    //
    //       affine.store %elem1, out[#map1]
    //  }
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Downsampling operations
//===----------------------------------------------------------------------===//

struct DownSamplingOpLowering : public ConversionPattern {
  DownSamplingOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::DownsamplingOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    // iterate for output len : i = 0 to len
    // get the input elem using  input mapping index = M* i
    //  store in y
    //  replace this op with the output_mem

    // DEBUG_PRINT_NO_ARGS() ;

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // For loop
    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    affine::AffineForOp forOp1 =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto iv = forOp1.getInductionVar();

    rewriter.setInsertionPointToStart(forOp1.getBody());
    DownsamplingOpAdaptor downsamplingAdaptor(operands);

    // For affine expression: #map1 = affine_map<(%arg0)[2ndOperand] : (%arg0 *
    // 2ndOperand)
    AffineExpr d0, s0;
    bindDims(rewriter.getContext(), d0);
    bindSymbols(rewriter.getContext(), s0);

    // AffineExpr ExprForDownSampling = rewriter.getAffineDimExpr(0) *
    // rewriter.getAffineSymbolExpr(0);
    AffineExpr ExprForDownSampling = d0 * s0;
    // Value constant3 = rewriter.create<arith::ConstantOp>(loc,
    // rewriter.getI64Type(),
    // rewriter.getIntegerAttr(rewriter.getIntegerType(64), 3));
    Value constant3 =
        rewriter.create<arith::ConstantIndexOp>(loc, 3); // working
    constant3.dump();

    int64_t SecondValueInt = 1;
    Value downsampling2ndArg = op->getOperand(1);
    dsp::ConstantOp constantOp2ndArg =
        downsampling2ndArg.getDefiningOp<dsp::ConstantOp>();
    DenseElementsAttr constantRhsValue = constantOp2ndArg.getValue();
    ;
    auto elements = constantRhsValue.getValues<FloatAttr>();
    float SecondValue = elements[0].getValueAsDouble();
    SecondValueInt = (int64_t)SecondValue;

    // Value downSamplingRateAsIndex = rewriter.create<arith::IndexCastOp>(loc,
    // rewriter.getIndexType(),DownsamplingRate);
    Value constantSamplingRateIndx =
        rewriter.create<arith::ConstantIndexOp>(loc, SecondValueInt);
    constantSamplingRateIndx.dump();

    AffineMap addMapForDownSampling = AffineMap::get(1, 1, ExprForDownSampling);
    // AffineMap addMapForDownSampling = AffineMap::get(1, 1, ValueRange{d0,s0
    // }); AffineMap addMapForDownSampling = AffineMap::get(1, 1,
    // ExprForDownSampling, rewriter.getContext()); AffineMap
    // addMapForDownSampling = AffineMap::get(1, 0, { d0}); //Working
    // DEBUG_PRINT_NO_ARGS() ;
    Value elem2 = rewriter.create<AffineLoadOp>(
        loc, downsamplingAdaptor.getLhs(), addMapForDownSampling,
        ValueRange{iv, constantSamplingRateIndx});
    elem2.dump();
    // store the result
    rewriter.create<AffineStoreOp>(loc, elem2, alloc, iv);

    rewriter.setInsertionPointAfter(forOp1);
    // debug
    //  forOp1->dump();
    //    %2ndOperand = arith.const 3 : f64
    //    affine.for %arg0 = 0 to 10 {
    //     #map1 = affine_map<(%arg0)[2ndOperand] : (%arg0 * 2ndOperand)
    //     %elem1 = affine.load input[#map1] <-- affine apply
    //     affine.store %elem1, out[%arg0]
    //  }
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: MedianFilterOp operations
//===----------------------------------------------------------------------===//

struct MedianFilterOpLowering : public ConversionPattern {
  MedianFilterOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::MedianFilterOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), 1);

    // For loop
    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    affine::AffineForOp forOp1 =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto iv = forOp1.getInductionVar();

    rewriter.setInsertionPointToStart(forOp1.getBody());
    MedianFilterOpAdaptor medianFilterOpAdaptor(operands);

    Value elem1 = rewriter.create<AffineLoadOp>(
        loc, medianFilterOpAdaptor.getInput(), iv);
    AffineExpr ExprForElem2 =
        rewriter.getAffineDimExpr(0) + rewriter.getAffineConstantExpr(1);
    AffineExpr ExprForElem3 =
        rewriter.getAffineDimExpr(0) + rewriter.getAffineConstantExpr(2);
    AffineMap addMapForElem2 = AffineMap::get(1, 0, ExprForElem2);
    AffineMap addMapForElem3 = AffineMap::get(1, 0, ExprForElem3);
    Value elem2 = rewriter.create<AffineLoadOp>(
        loc, medianFilterOpAdaptor.getInput(), addMapForElem2, ValueRange{iv});
    Value elem3 = rewriter.create<AffineLoadOp>(
        loc, medianFilterOpAdaptor.getInput(), addMapForElem3, ValueRange{iv});

    // sum
    Value sum1 = rewriter.create<arith::AddFOp>(loc, elem1, elem2);
    Value sum = rewriter.create<arith::AddFOp>(loc, sum1, elem3);

    // min
    Value minElem1Elem2 = rewriter.create<arith::MinimumFOp>(loc, elem1, elem2);
    Value min = rewriter.create<arith::MinimumFOp>(loc, minElem1Elem2, elem3);

    // max
    Value maxElem1Elem2 = rewriter.create<arith::MaximumFOp>(loc, elem1, elem2);
    Value max = rewriter.create<arith::MaximumFOp>(loc, maxElem1Elem2, elem3);

    // median
    Value min_plus_max = rewriter.create<arith::AddFOp>(loc, min, max);
    Value median = rewriter.create<arith::SubFOp>(loc, sum, min_plus_max);

    // store in alloc
    rewriter.create<AffineStoreOp>(loc, median, alloc, iv);
    rewriter.setInsertionPointAfter(forOp1);
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: SlidingWindowAvg operations
//===----------------------------------------------------------------------===//

struct SlidingWindowAvgOpLowering : public ConversionPattern {
  SlidingWindowAvgOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::SlidingWindowAvgOp::getOperationName(), 1, ctx) {
  }

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    // iterate for len = len - 2
    // get 3 elements
    // get the sum
    // get the avg = sum / 3
    //  store the result to output_mem
    //  replace this op with the output_mem

    // DEBUG_PRINT_NO_ARGS() ;

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    Value constant3 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(3));
    // For loop
    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    affine::AffineForOp forOp1 =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto iv = forOp1.getInductionVar();

    rewriter.setInsertionPointToStart(forOp1.getBody());
    SlidingWindowAvgOpAdaptor slidingWinAvgAdaptor(operands);

    Value elem1 =
        rewriter.create<AffineLoadOp>(loc, slidingWinAvgAdaptor.getInput(), iv);

    // affine-maps for elem2 and elem3
    AffineExpr ExprForElem2 =
        rewriter.getAffineDimExpr(0) + rewriter.getAffineConstantExpr(1);
    AffineExpr ExprForElem3 =
        rewriter.getAffineDimExpr(0) + rewriter.getAffineConstantExpr(2);

    AffineMap addMapForElem2 = AffineMap::get(1, 0, ExprForElem2);
    AffineMap addMapForElem3 = AffineMap::get(1, 0, ExprForElem3);
    Value elem2 = rewriter.create<AffineLoadOp>(
        loc, slidingWinAvgAdaptor.getInput(), addMapForElem2, ValueRange{iv});
    Value elem3 = rewriter.create<AffineLoadOp>(
        loc, slidingWinAvgAdaptor.getInput(), addMapForElem3, ValueRange{iv});

    Value sum1 = rewriter.create<arith::AddFOp>(loc, elem1, elem2);
    Value sum2 = rewriter.create<arith::AddFOp>(loc, sum1, elem3);
    Value avg = rewriter.create<arith::DivFOp>(loc, sum2, constant3);

    // store the result
    rewriter.create<AffineStoreOp>(loc, avg, alloc, iv);

    rewriter.setInsertionPointAfter(forOp1);
    // debug
    //  forOp1->dump();
    //    %cont3 = arith.const 3 : f64
    //    affine.for %arg0 = 0 to 8 {
    //     %elem1 = affine.load input[%arg0]
    //     #map1 = affine_map<(%arg0)[] : (%arg0 + 1)
    //     #map2 = affine_map<(%arg0)[] : (%arg0 + 2)
    //     %elem2 = affine.load input[#map1] <-- affine apply
    //     %elem3 = affine.load input[#map2]

    //    %sum1 = arith.addf %elem1 , %elem2
    //    %sum2 = arith.addf %sum1, %elem3
    //    %res = arith.divf %sum2 ,
    //    affine.store %sum2, out[%arg0]
    // }
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: FIRFilterResponse operations
//===----------------------------------------------------------------------===//
struct FIRFilterResponseOpLowering : public ConversionPattern {
  FIRFilterResponseOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::FIRFilterResponseOp::getOperationName(), 1,
                          ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // dsp.FIRFilterResponseOp has 2 operands -- both of type tensor f64

    // Get the location of FIRFilterResponseOp
    auto loc = op->getLoc();

    // Pseudo-Code
    //  y[n] = sum( h[k] * x[n-k]) k = 0 to lenOfh

    // Range for each element of the output tensor -- i = %arg0
    //   Create a tempValue = 0
    //   Range for each of the elements of filter len -- k = %arg1
    //   check for the condition that %arg0  - %arg1 >= 0 && < inputLen
    //   get elem1 = filter[k] , elem2 = x[i-k]
    //  use affine-map expression for calculating i-k
    //   tempValue = tempValue + elem1 * elem2
    // y[i] = tempValue

    lowerOpToLoopsFIR(
        op, operands, rewriter,
        [loc, op](OpBuilder &builder, ValueRange memRefOperands,
                  ValueRange loopIvs) {
          // ValueRange loopIvs) {

          // Generate an adaptor for the remapped operands of the
          // BinaryOp. This allows for using the nice named accessors
          // that are generated by the ODS.
          dsp::FIRFilterResponseOpAdaptor firFilterAdaptor(memRefOperands);

          // Generate loads for the element of 'lhs' and 'rhs' at the
          // inner loop.
          // auto lhsTensor = delayAdaptor.getLhs();
          auto lhsTensor = builder.create<affine::AffineLoadOp>(
              loc, firFilterAdaptor.getLhs(), loopIvs);

          // auto rhsScalar = op->getOperand(1);
          auto rhsScalar = builder.create<affine::AffineLoadOp>(
              loc, firFilterAdaptor.getRhs(), loopIvs);

          auto resultMulOp =
              builder.create<arith::MulFOp>(loc, lhsTensor, rhsScalar);

          return resultMulOp;
        });

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Delay operations
//===----------------------------------------------------------------------===//
struct DelayOpLowering : public ConversionPattern {
  DelayOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::DelayOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // dsp.DelayOp has 2 operands -- both of type tensor f64

    // Get the location of delayop
    auto loc = op->getLoc();

    // Pseudo-code
    // 2 affine loops --
    // first from 0 to delay_2ndArg
    //           here, inside AffineNest
    //           create affine:load from the arith.const operation with value 0
    //           use affine:store to store at result_op at indx
    //
    // 2nd from delay_2ndArg to lengthOfOperand0 of delayOp
    //           here, inside AffineNest
    //           create affine:load from input memref & indx = indx -
    //           delay_2ndArg create affine:store at result_op indx

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

    // For loop
    int64_t ub = tensorType.getShape()[0];

    // Get 2nd Arg
    DelayOpAdaptor delayOpAdaptor(operands);

    Value constant0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
    DEBUG_PRINT_NO_ARGS();
    // Creating SSA values for the lower bound and upper bound
    Value lowerBound = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 0));
    // Cast the f64 value directly to the index type
    Value inputUnit = rewriter.create<AffineLoadOp>(
        loc, delayOpAdaptor.getRhs(), ValueRange{});
    Value i64UpperBound =
        rewriter.create<arith::FPToSIOp>(loc, rewriter.getI64Type(), inputUnit);
    // Cast the i64 value to index type
    Value delay2ndArg = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), i64UpperBound);
    // Value inputLen = rewriter.create<arith::ConstantOp>(loc,
    // rewriter.getIndexType(), rewriter.getIntegerAttr(rewriter.getIndexType(),
    // ub));
    DEBUG_PRINT_WITH_ARGS("print delay2ndArg.dump() for debugging");

    DEBUG_PRINT_NO_ARGS();
    // Create an empty affine map list
    // SmallVector<AffineMap, 4> lbMaps, ubMaps;
    // Create identity affine maps for bounds
    // AffineMap lbMap = AffineMap::get(/*dimCount=*/0, /*symbolCount=*/0,
    // rewriter.getContext()); AffineMap ubMap = AffineMap::get(/*dimCount=*/0,
    // /*symbolCount=*/0, rewriter.getContext());

    // Create an AffineForOp with SSA values for the bounds
    Value step1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    scf::ForOp forOp1 =
        rewriter.create<scf::ForOp>(loc, lowerBound, delay2ndArg, step1);
    // Affine loop with non-int loop indices
    //  affine::AffineForOp forOp1 = rewriter.create<affine::AffineForOp>(loc,
    //  lowerBound, lbMap, inputLen, ubMap, 1);
    DEBUG_PRINT_NO_ARGS();

    auto iv = forOp1.getInductionVar();

    rewriter.setInsertionPointToStart(forOp1.getBody());
    // store the result
    //  rewriter.create<AffineStoreOp>(loc, constant0, alloc, iv);
    rewriter.create<memref::StoreOp>(loc, constant0, alloc, iv);

    rewriter.setInsertionPointAfter(forOp1);

    // Create the constants for lb2, step1, and calculate ub2
    Value lb2 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
    Value lenOfInput = rewriter.create<arith::ConstantIndexOp>(
        loc, /*length of input*/ ub); // Replace with the actual length
    Value ub2 = rewriter.create<arith::SubIOp>(loc, lenOfInput, delay2ndArg);
    Value step2 = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Create the second scf.for loop
    scf::ForOp forOp2 = rewriter.create<scf::ForOp>(loc, lb2, ub2, step2);
    Value iv2 = forOp2.getInductionVar();

    // Set insertion point to the start of the loop body
    rewriter.setInsertionPointToStart(forOp2.getBody());

    // Load value from allocIP[iv2]
    Value loadedVal =
        rewriter.create<memref::LoadOp>(loc, delayOpAdaptor.getLhs(), iv2);

    // Calculate the index iv2 + delaySecondArg
    Value newIndex = rewriter.create<arith::AddIOp>(loc, iv2, delay2ndArg);

    // Store the loaded value at alloc[newIndex]
    rewriter.create<memref::StoreOp>(loc, loadedVal, alloc, newIndex);
    rewriter.setInsertionPointAfter(forOp2);
    DEBUG_PRINT_NO_ARGS();
    // For 2nd loop --
    // loop from 0 to lenOfInput - 2ndArg
    //  load from index
    //  store at index + 2ndArg

    // forOp1.dump();
    // Expected MLIR-Affine
    // %0 = affine.load %alloc_0[] : memref<f64>
    // %1 = arith.fptosi %0 : f64 to i64
    // %2 = arith.index_cast %1 : i64 to index
    // %c1_15 = arith.constant 1 : index
    // scf.for %arg0 = %c0_14 to %2 step %c1_15 {
    //   memref.store %cst_13, %alloc[%arg0] : memref<10xf64>
    // }
    // %c0_16 = arith.constant 0 : index
    // %c10 = arith.constant 10 : index
    // %3 = arith.subi %c10, %2 : index
    // %c1_17 = arith.constant 1 : index
    // scf.for %arg0 = %c0_16 to %3 step %c1_17 {
    //   %4 = memref.load %alloc_1[%arg0] : memref<10xf64>
    //   %5 = arith.addi %arg0, %2 : index
    //   memref.store %4, %alloc[%5] : memref<10xf64>
    // }

    rewriter.replaceOp(op, alloc);
    DEBUG_PRINT_NO_ARGS();
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Gain operations
//===----------------------------------------------------------------------===//
struct GainOpLowering : public ConversionPattern {
  GainOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::GainOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // dsp.GainOp has 2 operands -- both of type tensor f64 , 2ndOperand should
    // have only 1 element

    // Get the location of GainOp
    auto loc = op->getLoc();

    // Pseudo-code:
    //   y[i] = y[i] * gain for  0<=i<N
    //

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
    GainOpAdaptor gainOpOpAdaptor(operands);
    // Value GetValueAtIndx2ndArg = op->getOperand(1);
    // dsp::ConstantOp constantOp2ndArg =
    // GetValueAtIndx2ndArg.getDefiningOp<dsp::ConstantOp>(); DenseElementsAttr
    // constantRhsValue = constantOp2ndArg.getValue();; auto elements =
    // constantRhsValue.getValues<FloatAttr>(); float gain =
    // elements[0].getValueAsDouble();

    // Value gain = gainOpOpAdaptor.getRhs();

    DEBUG_PRINT_NO_ARGS();

    // first from 1 <= i < N
    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    DEBUG_PRINT_NO_ARGS();

    // loop from 0 <= i < N

    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());

    Value getLhs =
        rewriter.create<AffineLoadOp>(loc, gainOpOpAdaptor.getLhs(), ivY);
    Value getRhs = rewriter.create<AffineLoadOp>(loc, gainOpOpAdaptor.getRhs(),
                                                 ValueRange{});
    Value mulProd = rewriter.create<arith::MulFOp>(loc, getLhs, getRhs);
    rewriter.create<AffineStoreOp>(loc, mulProd, alloc, ValueRange{ivY});
    DEBUG_PRINT_NO_ARGS();
    rewriter.setInsertionPointAfter(forOpY);

    // debug
    //  forOpX->dump();
    //  forOpY->dump();

    // %cst = arith.constant 6.2831853071800001 : f64
    // %cst_0 = arith.constant 4.600000e-01 : f64
    // %cst_1 = arith.constant 5.400000e-01 : f64
    // %cst_2 = arith.constant 4.000000e+00 : f64
    // %alloc = memref.alloc() : memref<4xf64>
    // %alloc_3 = memref.alloc() : memref<f64>
    // affine.store %cst_2, %alloc_3[] : memref<f64>
    // affine.for %arg0 = 0 to 4 {
    //   %0 = arith.index_castui %arg0 : index to i32
    //   %1 = arith.uitofp %0 : i32 to f64
    //   %2 = arith.mulf %1, %cst : f64
    //   %3 = arith.divf %2, %cst_2 : f64
    //   %4 = math.cos %3 : f64
    //   %5 = arith.mulf %4, %cst_0 : f64
    //   %6 = arith.subf %cst_1, %5 : f64
    //   affine.store %6, %alloc[%arg0] : memref<4xf64>
    // }

    // }
    // }
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: BitwiseAndOp operations
//===----------------------------------------------------------------------===//

struct BitwiseAndOpLowering : public ConversionPattern {
  BitwiseAndOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::BitwiseAndOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // dsp.bitwiseandop has 2 operands -- both of type tensor f64 , of the same
    // size

    // Get the location of BitwiseAndOp
    auto loc = op->getLoc();

    // Pseudo-code:
    //   y[i] = bitwiseand(lhs[i], rhs[i]) for  0<=i<N
    //

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // construct affine loops for the input
    SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value*/ 0);
    SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);
    BitwiseAndOpAdaptor bitwiseandOpAdaptor(operands);

    DEBUG_PRINT_NO_ARGS();

    // first from 0 <= i < N
    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    DEBUG_PRINT_NO_ARGS();

    // loop from 0 <= i < N
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());

    Value getLhs =
        rewriter.create<AffineLoadOp>(loc, bitwiseandOpAdaptor.getLhs(), ivY);
    Value getRhs =
        rewriter.create<AffineLoadOp>(loc, bitwiseandOpAdaptor.getRhs(), ivY);
    Value lhsInt =
        rewriter.create<arith::FPToSIOp>(loc, rewriter.getI64Type(), getLhs);
    Value rhsInt =
        rewriter.create<arith::FPToSIOp>(loc, rewriter.getI64Type(), getRhs);
    Value andiResult = rewriter.create<arith::AndIOp>(loc, lhsInt, rhsInt);
    Value resultFp = rewriter.create<arith::SIToFPOp>(
        loc, rewriter.getF64Type(), andiResult);

    rewriter.create<AffineStoreOp>(loc, resultFp, alloc, ValueRange{ivY});
    rewriter.setInsertionPointAfter(forOpY);

    // debug
    forOpY->dump();

    rewriter.replaceOp(op, alloc);

    return success();
  };
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: BitwiseAndOp operations
//===----------------------------------------------------------------------===//

struct zeroCrossCountOpLowering : public ConversionPattern {
  zeroCrossCountOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::zeroCrossCountOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    // dsp.zerocrosscount has 1 operand -- of type tensor f64

    // Get the location of zeroCrossCountOp
    auto loc = op->getLoc();

    // Pseudo-code is based on the C++ implementation here:
    // https://toto-share.com/2011/05/cc-zero-crossing-code/
    //   for 1<=i<N
    //      if sign of operand[i] is not equal to sign of operand[i-1]
    //         increment zero-cross count

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));
    Type integerType = rewriter.getI64Type();

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    // Force the result to be a tensor of size 1
    auto alloc = insertAllocAndDealloc(
        MemRefType::get(ArrayRef<int64_t>(1), tensorType.getElementType()), loc,
        rewriter);
    zeroCrossCountOpAdaptor zeroCrossCountOpAdaptor(operands);
    DEBUG_PRINT_NO_ARGS();

    // Define constants
    Value constant0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(),
        rewriter.getIntegerAttr(rewriter.getI64Type(), 0));
    Value constant1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64Type(),
        rewriter.getIntegerAttr(rewriter.getI64Type(), 1));
    Value Indx0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);

    // Define bounds
    Value lb = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(), 1));
    Value ub = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIntegerAttr(rewriter.getIndexType(),
                                tensorType.getShape()[0]));
    Value step = rewriter.create<arith::ConstantIndexOp>(loc, 1);

    // Set up for loop
    auto forOpY =
        rewriter.create<scf::ForOp>(loc, lb, ub, step, ValueRange{constant0});
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());
    auto countArg = forOpY.getRegionIterArgs()[0];

    // Get the current and previous elements
    Value ivYPrev = rewriter.create<arith::SubIOp>(loc, ivY, step);
    Value getLhsPrev = rewriter.create<memref::LoadOp>(
        loc, zeroCrossCountOpAdaptor.getLhs(), ivYPrev);
    Value getLhs = rewriter.create<memref::LoadOp>(
        loc, zeroCrossCountOpAdaptor.getLhs(), ivY);

    // Convert from float to integer
    Value lhsPrevInt = rewriter.create<arith::FPToSIOp>(
        loc, rewriter.getI64Type(), getLhsPrev);
    Value lhsInt =
        rewriter.create<arith::FPToSIOp>(loc, rewriter.getI64Type(), getLhs);

    // Check whether the elements are less than zero
    Value signLhsPrev = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, lhsPrevInt, constant0);
    Value signLhs = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, lhsInt, constant0);
    Value equal = rewriter.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                                 signLhsPrev, signLhs);

    // If the signs aren't the same, increment the zero cross counter
    auto ifOp =
        rewriter.create<scf::IfOp>(loc, TypeRange{integerType}, equal, true);

    // If block
    rewriter.setInsertionPointToStart(ifOp.thenBlock());
    rewriter.create<scf::YieldOp>(loc, ValueRange{countArg});

    // Else block
    rewriter.setInsertionPointToStart(ifOp.elseBlock());
    auto countPlusOne =
        rewriter.create<arith::AddIOp>(loc, countArg, constant1);
    rewriter.create<scf::YieldOp>(loc, ValueRange{countPlusOne});

    rewriter.setInsertionPointAfter(ifOp);
    auto countRes = ifOp.getResults()[0];
    rewriter.create<scf::YieldOp>(loc, ValueRange{countRes});

    rewriter.setInsertionPointAfter(forOpY);

    // debug
    // forOpY->dump();
    // %15 = "scf.for"(%12, %13, %14, %9) ({
    //     ^bb0(%arg0: index, %arg1: i64):
    //     %17 = "arith.subi"(%arg0, %14) <{overflowFlags =
    //     #arith.overflow<none>}>
    // : (index, index) -> index %18 = "memref.load"(%1, %17) <{nontemporal =
    // false}> : (memref<3xf64>, index) -> f64 %19 = "memref.load"(%1, %arg0)
    // <{nontemporal = false}> : (memref<3xf64>, index) -> f64 %20 =
    // "arith.fptosi"(%18) : (f64) -> i64 %21 = "arith.fptosi"(%19) : (f64) ->
    // i64
    //     %22 = "arith.cmpi"(%20, %9) <{predicate = 2 : i64}> : (i64, i64) ->
    //     i1 %23 = "arith.cmpi"(%21, %9) <{predicate = 2 : i64}> : (i64, i64)
    //     -> i1 %24 = "arith.cmpi"(%22, %23) <{predicate = 0 : i64}> : (i1, i1)
    //     -> i1 %25 = "scf.if"(%24) ({
    //         "scf.yield"(%arg1) : (i64) -> ()
    //     }, {
    //         %26 = "arith.addi"(%arg1, %10) <{overflowFlags =
    // #arith.overflow<none>}> : (i64, i64) -> i64 "scf.yield"(%26) : (i64) ->
    // ()
    //     }) : (i1) -> i64
    //     "scf.yield"(%25) : (i64) -> ()
    // }) : (index, index, index, i64) -> i64

    auto finalCountArg = forOpY.getResults()[0];
    Value finalCountArgFloat = rewriter.create<arith::SIToFPOp>(
        loc, rewriter.getF64Type(), finalCountArg);

    rewriter.create<AffineStoreOp>(loc, finalCountArgFloat, alloc, Indx0);
    rewriter.replaceOp(op, alloc);

    return success();
  };
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Binary operations
//===----------------------------------------------------------------------===//

template <typename BinaryOp, typename LoweredBinaryOp>
struct BinaryOpLowering : public ConversionPattern {
  BinaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder &builder, ValueRange memRefOperands,
                         ValueRange loopIvs) {
                     // Generate an adaptor for the remapped operands of the
                     // BinaryOp. This allows for using the nice named accessors
                     // that are generated by the ODS.
                     typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

                     // Generate loads for the element of 'lhs' and 'rhs' at the
                     // inner loop.
                     auto loadedLhs = builder.create<affine::AffineLoadOp>(
                         loc, binaryAdaptor.getLhs(), loopIvs);
                     auto loadedRhs = builder.create<affine::AffineLoadOp>(
                         loc, binaryAdaptor.getRhs(), loopIvs);

                     // Create the binary operation performed on the loaded
                     // values.
                     return builder.create<LoweredBinaryOp>(loc, loadedLhs,
                                                            loadedRhs);
                   });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine AdditionalPatterns: Shift operations
//===----------------------------------------------------------------------===//

struct ShiftRightOpLowering : public ConversionPattern {
  ShiftRightOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::ShiftRightOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    // Get the location of GainOp
    auto loc = op->getLoc();

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // first from 1 <= i < N
    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    typename dsp::ShiftRightOp::Adaptor binaryAdaptor(operands);

    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());

    auto loadedLhs =
        rewriter.create<affine::AffineLoadOp>(loc, binaryAdaptor.getLhs(), ivY);
    Value IntegerLhs =
        rewriter.create<arith::FPToSIOp>(loc, rewriter.getI64Type(), loadedLhs);

    auto loadedRhs =
        rewriter.create<affine::AffineLoadOp>(loc, binaryAdaptor.getRhs(), ivY);
    Value IntegerRhs =
        rewriter.create<arith::FPToSIOp>(loc, rewriter.getI64Type(), loadedRhs);

    auto LoweredOp =
        rewriter.create<arith::ShRSIOp>(loc, IntegerLhs, IntegerRhs);

    Value FloatOp =
        rewriter.create<arith::SIToFPOp>(loc, rewriter.getF64Type(), LoweredOp);

    rewriter.create<AffineStoreOp>(loc, FloatOp, alloc, ValueRange{ivY});

    rewriter.setInsertionPointAfter(forOpY);

    DEBUG_PRINT_NO_ARGS();

    // rewriter.replaceOp(op, FloatOp);
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine AdditionalPatterns: Matmul operations
//===----------------------------------------------------------------------===//

// template <typename BinaryOp>

struct MatmulOpLowering : public ConversionPattern {
  MatmulOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::MatmulOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    // Get the location of GainOp
    auto loc = op->getLoc();

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc_output = insertAllocAndDealloc(memRefType, loc, rewriter);

    typename dsp::MatmulOp::Adaptor binaryAdaptor(operands);

    auto lhsType =
        llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    // auto rhsType =
    // llvm::dyn_cast<RankedTensorType>(op->getOperand(1).getType());

    // first from 1 <= i < N
    int64_t lb = 0;
    int64_t ub_0 = lhsType.getShape()[0];
    int64_t ub_1 = lhsType.getShape()[1];
    int64_t step = 1;

    Value constantZero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

    // NOTE: matrix [y, x] --> y means row, x means column
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub_0, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());

    affine::AffineForOp forOpX =
        rewriter.create<AffineForOp>(loc, lb, ub_1, step);
    auto ivX = forOpX.getInductionVar();
    // auto getIterArg =  forOpX.getBody()->getArgument(1); //HWISOO: Find this
    // to check how previous codes did
    rewriter.setInsertionPointToStart(forOpX.getBody());

    rewriter.create<AffineStoreOp>(loc, constantZero, alloc_output,
                                   ValueRange{ivY, ivX});

    affine::AffineForOp forOpIndex =
        rewriter.create<AffineForOp>(loc, lb, ub_1, step);
    auto ivIndex = forOpIndex.getInductionVar();
    rewriter.setInsertionPointToStart(forOpIndex.getBody());

    auto loadedLhs = rewriter.create<affine::AffineLoadOp>(
        loc, binaryAdaptor.getLhs(), ValueRange{ivY, ivIndex});

    auto loadedRhs = rewriter.create<affine::AffineLoadOp>(
        loc, binaryAdaptor.getRhs(), ValueRange{ivIndex, ivX});

    Value mulLhsRhs = rewriter.create<arith::MulFOp>(loc, loadedLhs, loadedRhs);

    auto loadedResult = rewriter.create<affine::AffineLoadOp>(
        loc, alloc_output, ValueRange{ivY, ivX});

    Value addResultAndMul =
        rewriter.create<arith::AddFOp>(loc, loadedResult, mulLhsRhs);

    rewriter.create<AffineStoreOp>(loc, addResultAndMul, alloc_output,
                                   ValueRange{ivY, ivX});

    /*
    auto loadedLhs = rewriter.create<affine::AffineLoadOp>(loc,
binaryAdaptor.getLhs(), ivY); Value IntegerLhs =
rewriter.create<arith::FPToSIOp>(loc, rewriter.getI64Type(),  loadedLhs);

auto loadedRhs = rewriter.create<affine::AffineLoadOp>(loc,
binaryAdaptor.getRhs(), ivY); Value IntegerRhs =
rewriter.create<arith::FPToSIOp>(loc, rewriter.getI64Type(),  loadedRhs);

    auto LoweredOp = rewriter.create<LoweredBinaryOp>(loc, IntegerLhs,
IntegerRhs);

    Value FloatOp = rewriter.create<arith::SIToFPOp>(loc, rewriter.getF64Type(),
LoweredOp);

    rewriter.create<AffineStoreOp>(loc, FloatOp, alloc, ValueRange{ivY});

    */

    rewriter.setInsertionPointAfter(forOpY);

    DEBUG_PRINT_NO_ARGS();

    // rewriter.replaceOp(op, FloatOp);
    rewriter.replaceOp(op, alloc_output);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine AdditionalPatterns: Find peaks operations
//===----------------------------------------------------------------------===//

// template <typename BinaryOp>

struct FindPeaksOpLowering : public ConversionPattern {
  FindPeaksOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::FindPeaksOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    // Get the location of GainOp
    auto loc = op->getLoc();

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc_output = insertAllocAndDealloc(memRefType, loc, rewriter);

    auto countMemRefType = MemRefType::get({}, rewriter.getIndexType());
    auto alloc_peaks_count =
        insertAllocAndDealloc(countMemRefType, loc, rewriter);

    typename dsp::FindPeaksOp::Adaptor findPeaksOpAdaptor(operands);

    Value constant_minus_one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(-1));

    Value constant_index_zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(), rewriter.getIndexAttr(0));
    Value constant_index_one = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(), rewriter.getIndexAttr(1));

    rewriter.create<AffineStoreOp>(loc, constant_index_zero, alloc_peaks_count,
                                   ValueRange{});

    auto signalType =
        llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    int64_t lb = 1;
    int64_t ub = signalType.getShape()[0] - 1;
    int64_t step = 1;

    Value constant_len_singal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIndexType(),
        rewriter.getIndexAttr(signalType.getShape()[0]));

    //%distance = affine.load %alloc_distance[] : memref<index>
    auto distance_fp = rewriter.create<affine::AffineLoadOp>(
        loc, findPeaksOpAdaptor.getDistance(), ValueRange{});
    // f64 to index
    Value distance_ui = rewriter.create<arith::FPToUIOp>(
        loc, rewriter.getIntegerType(32), distance_fp);
    Value distance = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), distance_ui);

    affine::AffineForOp forOpInit =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto init_iter = forOpInit.getInductionVar();
    rewriter.setInsertionPointToStart(forOpInit.getBody());

    rewriter.create<AffineStoreOp>(loc, constant_minus_one, alloc_output,
                                   ValueRange{init_iter});

    rewriter.setInsertionPointAfter(forOpInit);

    affine::AffineForOp forOpSignal =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto current_index = forOpSignal.getInductionVar();
    rewriter.setInsertionPointToStart(forOpSignal.getBody());

    // %prev_index = arith.subi %current_index, %cst_one_index : index
    // %signal_prev = memref.load %alloc_signal[%prev_index] : memref<10xf64>
    // %signal_current = affine.load %alloc_signal[%current_index] :
    // memref<10xf64> %signal_next = affine.load %alloc_signal[%current_index+1]
    // : memref<10xf64> Q. How can I do this? %height = affine.load
    // %alloc_height[] : memref<f64>

    AffineExpr ExprForPrev =
        rewriter.getAffineDimExpr(0) - rewriter.getAffineConstantExpr(1);
    AffineMap addMapForPrev = AffineMap::get(1, 0, ExprForPrev);

    AffineExpr ExprForNext =
        rewriter.getAffineDimExpr(0) + rewriter.getAffineConstantExpr(1);
    AffineMap addMapForNext = AffineMap::get(1, 0, ExprForNext);

    auto signal_prev =
        rewriter.create<AffineLoadOp>(loc, findPeaksOpAdaptor.getSignal(),
                                      addMapForPrev, ValueRange{current_index});
    auto signal_current = rewriter.create<affine::AffineLoadOp>(
        loc, findPeaksOpAdaptor.getSignal(), current_index);
    auto signal_next =
        rewriter.create<AffineLoadOp>(loc, findPeaksOpAdaptor.getSignal(),
                                      addMapForNext, ValueRange{current_index});
    auto height = rewriter.create<affine::AffineLoadOp>(
        loc, findPeaksOpAdaptor.getHeight(), ValueRange{});

    //%cmp_current_prev = arith.cmpf ogt, %signal_current, %signal_prev : f64
    //%cmp_current_next = arith.cmpf ogt, %signal_current, %signal_next : f64
    //%cmp_current_height = arith.cmpf oge, %signal_current, %signal_next : f64
    auto cmp_current_prev = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGT, signal_current, signal_prev);
    auto cmp_current_next = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGT, signal_current, signal_next);
    auto cmp_current_height = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OGE, signal_current, height);

    //%and_two_cmps = arith.andi %cmp_current_prev, %cmp_current_next : index
    //%and_three_cmps = arith.andi %and_two_cmps, cmp_current_height : index
    auto and_two_cmps =
        rewriter.create<arith::AndIOp>(loc, cmp_current_prev, cmp_current_next);
    auto and_three_cmps =
        rewriter.create<arith::AndIOp>(loc, and_two_cmps, cmp_current_height);

    // scf.if %and_three_cmps {
    auto firstIfOp =
        rewriter.create<scf::IfOp>(loc, and_three_cmps, false /* else=1 */);
    rewriter.setInsertionPointToStart(firstIfOp.thenBlock());

    //%peaks_count = affine.load %alloc_peaks_count[] : memref<index>
    //%cmp_new_peak = arith.cmpi eq, %peaks_count, %cst_zero_index : index
    auto peaks_count = rewriter.create<affine::AffineLoadOp>(
        loc, alloc_peaks_count, ValueRange{});
    auto cmp_new_peak = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, peaks_count, constant_index_zero);

    // scf.if %cmp_new_peak {
    //     memref.store %current_index, %alloc_peaks[%peaks_count] :
    //     memref<10xindex> %peaks_count_inc = arith.addi %peaks_count,
    //     %cst_one_index : index affine.store %peaks_count_inc,
    //     %alloc_peaks_count[] : memref<index>
    // }
    auto secondIfOp =
        rewriter.create<scf::IfOp>(loc, cmp_new_peak, true /* else=1 */);
    rewriter.setInsertionPointToStart(secondIfOp.thenBlock());
    // index to f64
    Value current_index_to_ui = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), current_index);
    Value current_index_to_f64 = rewriter.create<arith::UIToFPOp>(
        loc, rewriter.getF64Type(), current_index_to_ui);
    rewriter.create<memref::StoreOp>(loc, current_index_to_f64, alloc_output,
                                     ValueRange{peaks_count});
    auto peaks_count_inc =
        rewriter.create<arith::AddIOp>(loc, peaks_count, constant_index_one);
    rewriter.create<AffineStoreOp>(loc, peaks_count_inc, alloc_peaks_count,
                                   ValueRange{});

    /*
    else {
        %last_peaks_count = arith.subi %peaks_count, %cst_one_index : index
        %last_peak_index = memref.load %alloc_peaks[%last_peaks_count] :
    memref<10xindex> %subtract_current_index_last_peak = arith.subi
    %current_index, %last_peak_index : index %cmp_sub_distance = arith.cmpi sge,
    %subtract_current_index_last_peak, %distance : index
        */
    rewriter.setInsertionPointToStart(secondIfOp.elseBlock());
    // auto last_peak_index = rewriter.create<AffineLoadOp>(loc, alloc_output,
    // addMapForPrev, ValueRange{peaks_count}); HWISOO: It does not work since
    // it gives "error: 'affine.load' op index must be a valid dimension or
    // symbol identifier" here.
    Value last_peaks_count =
        rewriter.create<arith::SubIOp>(loc, peaks_count, constant_index_one);
    auto last_peak_index_fp = rewriter.create<memref::LoadOp>(
        loc, alloc_output, ValueRange{last_peaks_count});
    // f64 to index
    Value last_peak_index_ui = rewriter.create<arith::FPToUIOp>(
        loc, rewriter.getIntegerType(32), last_peak_index_fp);
    Value last_peak_index = rewriter.create<arith::IndexCastOp>(
        loc, rewriter.getIndexType(), last_peak_index_ui);
    Value subtract_current_index_last_peak =
        rewriter.create<arith::SubIOp>(loc, current_index, last_peak_index);
    auto cmp_sub_distance = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, subtract_current_index_last_peak,
        distance);

    /*
        scf.if %cmp_sub_distance {
    memref.store %current_index, %alloc_peaks[%peaks_count] : memref<10xindex>
    %peaks_count_inc = arith.addi %peaks_count, %cst_one_index : index
    affine.store %peaks_count_inc, %alloc_peaks_count[] : memref<index>
            }
    }
    */
    auto thirdIfOp =
        rewriter.create<scf::IfOp>(loc, cmp_sub_distance, true /* else=1 */);
    rewriter.setInsertionPointToStart(thirdIfOp.thenBlock());
    // index to f64
    Value current_index_to_ui_2 = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), current_index);
    Value current_index_to_f64_2 = rewriter.create<arith::UIToFPOp>(
        loc, rewriter.getF64Type(), current_index_to_ui_2);
    rewriter.create<memref::StoreOp>(loc, current_index_to_f64_2, alloc_output,
                                     ValueRange{peaks_count});
    auto peaks_count_inc_2 =
        rewriter.create<arith::AddIOp>(loc, peaks_count, constant_index_one);
    rewriter.create<AffineStoreOp>(loc, peaks_count_inc_2, alloc_peaks_count,
                                   ValueRange{});

    rewriter.setInsertionPointAfter(forOpSignal);

    /* Setting last element of the output as the count of peaks.
    Note that last-last ([-2]) should be always -1. */
    auto peaks_count_final = rewriter.create<affine::AffineLoadOp>(
        loc, alloc_peaks_count, ValueRange{});
    // index to f64
    Value peaks_count_final_to_ui = rewriter.create<arith::IndexCastUIOp>(
        loc, rewriter.getIntegerType(32), peaks_count_final);
    Value peaks_count_final_to_f64 = rewriter.create<arith::UIToFPOp>(
        loc, rewriter.getF64Type(), peaks_count_final_to_ui);
    rewriter.create<AffineStoreOp>(loc, peaks_count_final_to_f64, alloc_output,
                                   addMapForPrev,
                                   ValueRange{constant_len_singal});

    rewriter.replaceOp(op, alloc_output);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Unary operations
//===----------------------------------------------------------------------===//

template <typename UnaryOp, typename LoweredUnaryOp>
struct UnaryOpLowering : public ConversionPattern {
  UnaryOpLowering(MLIRContext *ctx)
      : ConversionPattern(UnaryOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder &builder, ValueRange memRefOperands,
                         ValueRange loopIvs) {
                     // Generate an adaptor for the remapped operands of the
                     // UnaryOp. This allows for using the nice named accessors
                     // that are generated by the ODS.
                     typename UnaryOp::Adaptor unaryAdaptor(memRefOperands);

                     // Generate loads for the element of 'lhs' and 'rhs' at the
                     // inner loop.
                     auto loadedInput = builder.create<affine::AffineLoadOp>(
                         loc, unaryAdaptor.getInput(), loopIvs);

                     // Create the unary operation performed on the loaded
                     // values.
                     return builder.create<LoweredUnaryOp>(loc, loadedInput);
                   });
    return success();
  }
};

using AddOpLowering = BinaryOpLowering<dsp::AddOp, arith::AddFOp>;
using ModuloOpLowering = BinaryOpLowering<dsp::ModuloOp, arith::RemFOp>;
using SubOpLowering = BinaryOpLowering<dsp::SubOp, arith::SubFOp>;
using MulOpLowering = BinaryOpLowering<dsp::MulOp, arith::MulFOp>;
using DivOpLowering = BinaryOpLowering<dsp::DivOp, arith::DivFOp>;
using SinOpLowering = UnaryOpLowering<dsp::SinOp, math::SinOp>;
using CosOpLowering = UnaryOpLowering<dsp::CosOp, math::CosOp>;
//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Constant operations
//===----------------------------------------------------------------------===//

struct ConstantOpLowering : public OpRewritePattern<dsp::ConstantOp> {
  using OpRewritePattern<dsp::ConstantOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(dsp::ConstantOp op,
                                PatternRewriter &rewriter) const final {
    DenseElementsAttr constantValue = op.getValue();
    Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = llvm::cast<RankedTensorType>(op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
               0, *std::max_element(valueShape.begin(), valueShape.end())))
        constantIndices.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.value_begin<FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<affine::AffineStoreOp>(
            loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
            llvm::ArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Func operations
//===----------------------------------------------------------------------===//

struct FuncOpLowering : public OpConversionPattern<dsp::FuncOp> {
  using OpConversionPattern<dsp::FuncOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dsp::FuncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We only lower the main function as we expect that all other functions
    // have been inlined.
    if (op.getName() != "main")
      return failure();

    // Verify that the given main has no inputs and results.
    if (op.getNumArguments() || op.getFunctionType().getNumResults()) {
      return rewriter.notifyMatchFailure(op, [](Diagnostic &diag) {
        diag << "expected 'main' to have 0 inputs and 0 results";
      });
    }

    // Create a new non-dsp function, with the same region.
    auto func = rewriter.create<mlir::func::FuncOp>(op.getLoc(), op.getName(),
                                                    op.getFunctionType());
    rewriter.inlineRegionBefore(op.getRegion(), func.getBody(), func.end());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Print operations
//===----------------------------------------------------------------------===//

struct PrintOpLowering : public OpConversionPattern<dsp::PrintOp> {
  using OpConversionPattern<dsp::PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(dsp::PrintOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // We don't lower "dsp.print" in this pass, but we need to update its
    // operands.
    rewriter.modifyOpInPlace(op,
                             [&] { op->setOperands(adaptor.getOperands()); });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Return operations
//===----------------------------------------------------------------------===//

struct ReturnOpLowering : public OpRewritePattern<dsp::ReturnOp> {
  using OpRewritePattern<dsp::ReturnOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(dsp::ReturnOp op,
                                PatternRewriter &rewriter) const final {
    // During this lowering, we expect that all function calls have been
    // inlined.
    if (op.hasOperand())
      return failure();

    // We lower "dsp.return" directly to "func.return".
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Transpose operations
//===----------------------------------------------------------------------===//

struct TransposeOpLowering : public ConversionPattern {
  TransposeOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::TransposeOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](OpBuilder &builder, ValueRange memRefOperands,
                         ValueRange loopIvs) {
                     // Generate an adaptor for the remapped operands of the
                     // TransposeOp. This allows for using the nice named
                     // accessors that are generated by the ODS.
                     dsp::TransposeOpAdaptor transposeAdaptor(memRefOperands);
                     Value input = transposeAdaptor.getInput();

                     // Transpose the elements by generating a load from the
                     // reverse indices.
                     SmallVector<Value, 2> reverseIvs(llvm::reverse(loopIvs));
                     return builder.create<affine::AffineLoadOp>(loc, input,
                                                                 reverseIvs);
                   });
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: Transpose operations
//===----------------------------------------------------------------------===//

struct Conv2DOpLowering : public ConversionPattern {
  Conv2DOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::Conv2DOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    // output mem alloc and dealloc
    auto output = llvm::dyn_cast<RankedTensorType>((*op->result_type_begin()));
    auto outputMem = convertTensorToMemRef(output);
    auto alloc = insertAllocAndDealloc(outputMem, loc, rewriter);

    Conv2DOpAdaptor conv2dAdaptor(operands);
    Value input = conv2dAdaptor.getInput();
    Value kernel = conv2dAdaptor.getKernel();

    // ranked tensor type
    auto inputType =
        llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    auto kernelType =
        llvm::dyn_cast<RankedTensorType>(op->getOperand(1).getType());

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> kernelShape = kernelType.getShape();

    // input layout
    int64_t IH = inputShape[0];
    int64_t IW = inputShape[1];

    // kernel layout
    int64_t KH = kernelShape[0];
    int64_t KW = kernelShape[1];

    // output layout
    ArrayRef<int64_t> outputShape = output.getShape();
    int64_t OH = outputShape[0];
    int64_t OW = outputShape[1];

    AffineExpr d0, d1, d2, d3; // declare affine expression: i, j, p, q
    bindDims(
        rewriter.getContext(), d0, d1, d2,
        d3); // bind affine expr d0, d1 to current input dimension i, j, p, q

    // input affine map
    AffineMap inputMap = AffineMap::get(
        4, 0, ArrayRef<AffineExpr>{d0 + d2, d1 + d3}, rewriter.getContext());
    // kernel affine map
    AffineMap kernelMap = AffineMap::get(4, 0, ArrayRef<AffineExpr>{d2, d3},
                                         rewriter.getContext());

    // loops
    int64_t lb = 0, step = 1;
    /* looping i*/
    AffineForOp forOpI = rewriter.create<AffineForOp>(loc, lb, OH, step);
    rewriter.setInsertionPointToStart(forOpI.getBody());
    auto ivI = forOpI.getInductionVar();

    /* looping j*/
    AffineForOp forOpJ = rewriter.create<AffineForOp>(loc, lb, OW, step);
    rewriter.setInsertionPointToStart(forOpJ.getBody());
    auto ivJ = forOpJ.getInductionVar();

    // initilize output val
    Value zeroVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
    rewriter.create<AffineStoreOp>(loc, zeroVal, alloc, ValueRange{ivI, ivJ});

    /* looping p*/
    AffineForOp forOpP = rewriter.create<AffineForOp>(loc, lb, KH, step);
    rewriter.setInsertionPointToStart(forOpP.getBody());
    auto ivP = forOpP.getInductionVar();

    /* looping q*/
    AffineForOp forOpQ = rewriter.create<AffineForOp>(loc, lb, KW, step);
    rewriter.setInsertionPointToStart(forOpQ.getBody());
    auto ivQ = forOpQ.getInductionVar();

    // input bound check
    Value inputRow = rewriter.create<AffineApplyOp>(
        loc, inputMap.getSubMap(0), ValueRange{ivI, ivJ, ivP, ivQ});
    Value inputCol = rewriter.create<AffineApplyOp>(
        loc, inputMap.getSubMap(1), ValueRange{ivI, ivJ, ivP, ivQ});
    Value rowUB = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, inputRow,
        rewriter.create<arith::ConstantIndexOp>(loc, IH));
    Value colUB = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, inputCol,
        rewriter.create<arith::ConstantIndexOp>(loc, IW));
    Value bound = rewriter.create<arith::AndIOp>(loc, rowUB, colUB);

    // bound condition
    rewriter.create<scf::IfOp>(
        loc, bound, [&](OpBuilder &builder, Location loc) {
          // load input
          Value inputVal = builder.create<AffineLoadOp>(
              loc, input, inputMap, ValueRange{ivI, ivJ, ivP, ivQ});
          Value kernelVal = builder.create<AffineLoadOp>(
              loc, kernel, kernelMap, ValueRange{ivI, ivJ, ivP, ivQ});
          // mul
          Value prod = builder.create<arith::MulFOp>(loc, inputVal, kernelVal);
          Value outputVal =
              builder.create<AffineLoadOp>(loc, alloc, ValueRange{ivI, ivJ});
          Value sum = builder.create<arith::AddFOp>(loc, prod, outputVal);

          // store the computed output
          builder.create<AffineStoreOp>(loc, sum, alloc, ValueRange{ivI, ivJ});

          builder.create<scf::YieldOp>(loc);
        });

    rewriter.replaceOp(op, alloc);

    return success();
  }
}; // conv2d

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: ThresholdUpOpLowering operations
//===----------------------------------------------------------------------===//

struct ThresholdUpOpLowering : public ConversionPattern {
  ThresholdUpOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::ThresholdUpOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    // Pseudo-code:
    //   y[n] = 1 , if a[i] >= threshld
    //     = 0 , else

    // output for result type
    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);

    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    Value constant0 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));

    Value constant1 = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(1));

    // y[n] = a[n] , if a[i] >= threshld
    // loop from 0 to len

    // load from X,
    ThresholdUpOpAdaptor thresholdUpAdaptor(operands);

    int64_t lb = 0;
    int64_t ub = tensorType.getShape()[0];
    int64_t step = 1;

    // for loop from 0 to len(Output)
    affine::AffineForOp forOpY =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivY = forOpY.getInductionVar();
    rewriter.setInsertionPointToStart(forOpY.getBody());

    Value inputX =
        rewriter.create<AffineLoadOp>(loc, thresholdUpAdaptor.getInput(), ivY);

    // Load the threshold value from the memref
    auto thresholdMemRef = thresholdUpAdaptor.getThreshold();
    auto returnOriginalMemRef = thresholdUpAdaptor.getReturnoriginal();
    auto threshold =
        rewriter.create<AffineLoadOp>(loc, thresholdMemRef, ValueRange{});
    auto returnOriginal =
        rewriter.create<AffineLoadOp>(loc, returnOriginalMemRef, ValueRange{});

    // Compare a[i] >= threshold
    auto cmp1 = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGE,
                                               inputX, threshold);
    // Compare if return original is true or false and return 1 or original
    // value
    auto cmpro = rewriter.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ,
                                                constant1, returnOriginal);

    // Use select to choose between inputX and 1
    auto selectreturn =
        rewriter.create<arith::SelectOp>(loc, cmpro, inputX, constant1);

    // Use select to choose between 0 and selectreturn
    auto selectOp =
        rewriter.create<arith::SelectOp>(loc, cmp1, selectreturn, constant0);

    // Store the result
    rewriter.create<AffineStoreOp>(loc, selectOp, alloc, ivY);

    rewriter.setInsertionPointAfter(forOpY);
    // debug
    //  forOpY->dump();
    //  affine.store %cst, %alloc_10[] : memref<f64>
    //  %0 = affine.load %alloc_11[4] : memref<10xf64>
    //  affine.store %0, %alloc[0] : memref<1xf64>

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: GenerateDTMFOpLowering operations
//===----------------------------------------------------------------------===//

struct GenerateDTMFOpLowering : public ConversionPattern {
  GenerateDTMFOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::GenerateDTMFOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    auto tensorType = llvm::cast<RankedTensorType>((*op->result_type_begin()));

    // allocation & deallocation for the result of this operation
    auto memRefType = convertTensorToMemRef(tensorType);

    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    GenerateDTMFOpAdaptor generatedtmfAdaptor(operands);
    std::vector<std::vector<int64_t>> freqPairs = {
        {697, 1209}, {697, 1336}, {697, 1477}, {770, 1209},
        {770, 1336}, {770, 1477}, {852, 1209}, {852, 1336},
        {852, 1477}, {941, 1209}, {941, 1336}, {941, 1477}};

    //     auto inputval = generatedtmfAdaptor.getInput();

    //     Value GetDigitInput = op->getOperand(0);
    //     dsp::ConstantOp inputvl = inputval.getDefiningOp<dsp::ConstantOp>();
    //     DenseElementsAttr inputvalue = inputvl.getValue();
    //     auto elements1 = inputvalue.getValues<FloatAttr>();
    //     float input = elements1[0].getValueAsDouble();

    //     Value GetDurationOp = op->getOperand(1);
    //     dsp::ConstantOp constantOp2ndArg =
    //         GetDurationOp.getDefiningOp<dsp::ConstantOp>();
    //     DenseElementsAttr constant2ndValue = constantOp2ndArg.getValue();
    //     auto elements2 = constant2ndValue.getValues<FloatAttr>();
    //     float duration = elements2[0].getValueAsDouble();

    //     Value GetFreqOp = op->getOperand(2);
    //     dsp::ConstantOp constantOp3rdArg =
    //         GetFreqOp.getDefiningOp<dsp::ConstantOp>();
    //     DenseElementsAttr constant3rdValue = constantOp3rdArg.getValue();
    //     auto elements3 = constant3rdValue.getValues<FloatAttr>();
    //     float freq = elements3[0].getValueAsDouble();

    //     const std::vector<int64_t> &pair = freqPairs[input];
    //     int64_t f1 = pair[0];
    //     int64_t f2 = pair[1];
    // int64_t lb = 0;
    //     int64_t ub = tensorType.getShape()[0];
    //     int64_t step = 1;

    //     Value const2pi = rewriter.create<arith::ConstantOp>(
    //         loc, rewriter.getF64Type(),
    //         rewriter.getF64FloatAttr(6.28318530718));

    //     Value mulProd = rewriter.create<arith::MulFOp>(loc, getLhs, getRhs);
    // auto yMemRefType = MemRefType::get({ub}, rewriter.getF64Type());
    //     auto tAlloc = rewriter.create<memref::AllocOp>(loc, yMemRefType);

    // // for loop from 0 to len(Output)
    //     affine::AffineForOp forOpY =
    //         rewriter.create<AffineForOp>(loc, lb, ub, step);
    //     auto ivY = forOpY.getInductionVar();
    //     rewriter.setInsertionPointToStart(forOpY.getBody());

    //     Value inputX =
    //         rewriter.create<AffineLoadOp>(loc, thresholdUpAdaptor.getInput(),
    //         ivY);

    //     // Store the result
    //     rewriter.create<AffineStoreOp>(loc, selectOp, alloc, ivY);

    //     rewriter.setInsertionPointAfter(forOpY);

    //     rewriter.replaceOp(op, alloc);

    return success();
  }
};

struct QamModulateRealOpLowering : public ConversionPattern {
  QamModulateRealOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::QamModulateRealOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    auto output = llvm::dyn_cast<RankedTensorType>((*op->result_type_begin()));
    auto outputMem = convertTensorToMemRef(output);
    auto alloc = insertAllocAndDealloc(outputMem, loc, rewriter);

    QamModulateRealOpAdaptor adaptor(operands);
    Value signal = adaptor.getSignal();

    llvm::ArrayRef<int64_t> outputShape = output.getShape();

    // constant vals;
    Value negOneVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(-1));
    Value zeroVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
    Value oneVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(1));

    AffineExpr d0; // declare one dimension affine expr
    bindDims(rewriter.getContext(),
             d0); // bind affine expr d0 to current input (real array) dimension

    // real affine map
    AffineMap signalMap = AffineMap::get(1, 0, ArrayRef<AffineExpr>{d0 * 2},
                                         rewriter.getContext());
    // output affine map
    AffineMap outputRealMap =
        AffineMap::get(1, 0, ArrayRef<AffineExpr>{d0}, rewriter.getContext());

    // loops
    int64_t lb = 0, step = 1, ub = outputShape[0];
    /* looping i*/
    AffineForOp forOpI = rewriter.create<AffineForOp>(loc, lb, ub, step);
    rewriter.setInsertionPointToStart(forOpI.getBody());
    auto ivI = forOpI.getInductionVar();

    // input bound check
    Value signalNum =
        rewriter.create<AffineLoadOp>(loc, signal, signalMap, ValueRange{ivI});

    Value zeroReal = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, signalNum, zeroVal);

    Value out =
        rewriter.create<arith::SelectOp>(loc, zeroReal, negOneVal, oneVal);

    rewriter.create<AffineStoreOp>(loc, out, alloc, outputRealMap,
                                   ValueRange{ivI});
    rewriter.setInsertionPointAfter(forOpI);
    rewriter.replaceOp(op, alloc);

    return success();
  }
};

struct QamModulateImgOpLowering : public ConversionPattern {
  QamModulateImgOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::QamModulateImgOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();

    auto output = llvm::dyn_cast<RankedTensorType>((*op->result_type_begin()));
    auto outputMem = convertTensorToMemRef(output);
    auto alloc = insertAllocAndDealloc(outputMem, loc, rewriter);

    QamModulateRealOpAdaptor adaptor(operands);
    Value signal = adaptor.getSignal();

    llvm::ArrayRef<int64_t> outputShape = output.getShape();

    // constant vals;
    Value negOneVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(-1));
    Value zeroVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
    Value oneVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(1));

    AffineExpr d0; // declare one dimension affine expr
    bindDims(rewriter.getContext(),
             d0); // bind affine expr d0 to current input (real array) dimension

    // real affine map
    AffineMap signalMap = AffineMap::get(1, 0, ArrayRef<AffineExpr>{d0 * 2 + 1},
                                         rewriter.getContext());
    // output affine map
    AffineMap outputImgMap =
        AffineMap::get(1, 0, ArrayRef<AffineExpr>{d0}, rewriter.getContext());

    // loops
    int64_t lb = 0, step = 1, ub = outputShape[0];
    /* looping i*/
    AffineForOp forOpI = rewriter.create<AffineForOp>(loc, lb, ub, step);
    rewriter.setInsertionPointToStart(forOpI.getBody());
    auto ivI = forOpI.getInductionVar();

    // input bound check
    Value signalNum =
        rewriter.create<AffineLoadOp>(loc, signal, signalMap, ValueRange{ivI});

    Value zeroReal = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, signalNum, zeroVal);

    Value out =
        rewriter.create<arith::SelectOp>(loc, zeroReal, negOneVal, oneVal);

    rewriter.create<AffineStoreOp>(loc, out, alloc, outputImgMap,
                                   ValueRange{ivI});

    rewriter.setInsertionPointAfter(forOpI);
    rewriter.replaceOp(op, alloc);

    return success();
  }
};
//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: QAM demodulate operations
//===----------------------------------------------------------------------===//

struct QamDemodulateOpLowering : public ConversionPattern {
  QamDemodulateOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::QamDemodulateOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {

    auto loc = op->getLoc();
    // output mem alloc and dealloc
    auto output = llvm::dyn_cast<RankedTensorType>((*op->result_type_begin()));
    auto outputMem = convertTensorToMemRef(output);
    auto alloc = insertAllocAndDealloc(outputMem, loc, rewriter);

    QamDemodulateOpAdaptor qamDemodualteAdaptor(operands);
    Value realVal = qamDemodualteAdaptor.getReal();
    Value imgVal = qamDemodualteAdaptor.getImagine();

    // ranked tensor type
    auto realType =
        llvm::dyn_cast<RankedTensorType>(op->getOperand(0).getType());

    llvm::ArrayRef<int64_t> realShape = realType.getShape();
    llvm::ArrayRef<int64_t> outputShape = output.getShape();

    // constant vals;
    Value negOneVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(-1));
    Value zeroVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(0));
    Value oneVal = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(1));

    AffineExpr d0; // declare one dimension affine expr
    bindDims(rewriter.getContext(),
             d0); // bind affine expr d0 to current input (real array) dimension

    // real affine map
    AffineMap realMap =
        AffineMap::get(1, 0, ArrayRef<AffineExpr>{d0}, rewriter.getContext());
    // imagine affine map
    AffineMap imgMap =
        AffineMap::get(1, 0, ArrayRef<AffineExpr>{d0}, rewriter.getContext());
    // output affine map
    AffineMap outputMapReal = AffineMap::get(1, 0, ArrayRef<AffineExpr>{d0 * 2},
                                             rewriter.getContext());
    AffineMap outputMapImagine = AffineMap::get(
        1, 0, ArrayRef<AffineExpr>{d0 * 2 + 1}, rewriter.getContext());

    // loops
    int64_t lb = 0, step = 1, ub = realShape[0];
    /* looping i*/
    AffineForOp forOpI = rewriter.create<AffineForOp>(loc, lb, ub, step);
    rewriter.setInsertionPointToStart(forOpI.getBody());
    auto ivI = forOpI.getInductionVar();

    // input bound check
    Value realNum =
        rewriter.create<AffineLoadOp>(loc, realVal, realMap, ValueRange{ivI});
    Value imgNum =
        rewriter.create<AffineLoadOp>(loc, imgVal, imgMap, ValueRange{ivI});

    Value negReal = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, realNum, negOneVal);
    Value negImagine = rewriter.create<arith::CmpFOp>(
        loc, arith::CmpFPredicate::OEQ, imgNum, negOneVal);

    Value out1 =
        rewriter.create<arith::SelectOp>(loc, negReal, zeroVal, oneVal);
    Value out2 =
        rewriter.create<arith::SelectOp>(loc, negImagine, zeroVal, oneVal);

    rewriter.create<AffineStoreOp>(loc, out1, alloc, outputMapReal,
                                   ValueRange{ivI});
    rewriter.create<AffineStoreOp>(loc, out2, alloc, outputMapImagine,
                                   ValueRange{ivI});

    rewriter.replaceOp(op, alloc);

    return success();
  }
}; // qam_demodulate op

//===----------------------------------------------------------------------===//
// ToyToAffine RewritePatterns: BeamForm operations
//===----------------------------------------------------------------------===//

struct BeamFormOpLowering : public ConversionPattern {
  BeamFormOpLowering(MLIRContext *ctx)
      : ConversionPattern(dsp::BeamFormOp::getOperationName(), 1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    auto beamFormOp = llvm::dyn_cast<mlir::dsp::BeamFormOp>(op);

    // allocating space for output
    auto output = llvm::dyn_cast<RankedTensorType>((*op->result_type_begin()));
    llvm::errs() << "output case.\n";
    auto outputMemRefType = convertTensorToMemRef(output);
    auto alloc = insertAllocAndDealloc(outputMemRefType, loc, rewriter);

    BeamFormOpAdaptor beamFormAdaptor(operands);
    auto time = beamFormAdaptor.getTime();
    auto weights = beamFormAdaptor.getWeights();

    // allocating space for internal generated signals
    auto timeDim = output.getShape()[0]; // dry run: 9
    int64_t antennas = beamFormOp.getAntennas();
    int64_t frequency = beamFormOp.getFreq();

    llvm::ArrayRef<int64_t> signalShape{antennas, timeDim};
    auto signalType = output.clone(signalShape);

    auto signalMemRefType = convertTensorToMemRef(signalType);
    auto allocSignal = insertAllocAndDealloc(signalMemRefType, loc, rewriter);

    AffineExpr d0, d1; // i, j for generated signal dimension
    bindDims(rewriter.getContext(), d0, d1);

    // generated input map
    AffineMap genInputMap =
        AffineMap::get(2 /* dim */, 0 /* sym */, ArrayRef<AffineExpr>{d0, d1},
                       rewriter.getContext());
    // time affine map
    AffineMap timeMap =
        AffineMap::get(2 /* dim */, 0 /* sym */, ArrayRef<AffineExpr>{d1},
                       rewriter.getContext());

    // output map
    AffineMap outputMap =
        AffineMap::get(2, 0, ArrayRef<AffineExpr>{d0}, rewriter.getContext());

    auto pi = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(3.1415926));
    auto zero = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(),
                                                   rewriter.getF64FloatAttr(0));
    auto two = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(),
                                                  rewriter.getF64FloatAttr(2));
    auto four = rewriter.create<arith::ConstantOp>(loc, rewriter.getF64Type(),
                                                   rewriter.getF64FloatAttr(4));
    auto two_pi = rewriter.create<arith::MulFOp>(loc, pi, two); // 2 * pi
    auto freq_val = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getF64Type(), rewriter.getF64FloatAttr(frequency));
    auto phase_var =
        rewriter.create<arith::MulFOp>(loc, two_pi, freq_val); // 2*pi*freq

    // for loop from 0 to phase
    int64_t lb = 0, ub = antennas, step = 1;
    affine::AffineForOp forOpI =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivI = forOpI.getInductionVar(); // i : phase
    rewriter.setInsertionPointToStart(forOpI.getBody());

    // get the induction var to phase variable
    auto intType = rewriter.getI64Type();
    auto intI = rewriter.create<arith::IndexCastOp>(loc, intType, ivI);
    auto floatType = rewriter.getF64Type();
    auto floatI = rewriter.create<arith::SIToFPOp>(loc, floatType, intI);

    auto iter_tmp = rewriter.create<arith::MulFOp>(loc, floatI, pi); // i * pi
    auto iter_args =
        rewriter.create<arith::DivFOp>(loc, iter_tmp, four); // i*pi/4

    // for loop from 0 to timeDim
    ub = timeDim;
    affine::AffineForOp forOpJ =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivJ = forOpJ.getInductionVar();
    rewriter.setInsertionPointToStart(forOpJ.getBody());

    rewriter.create<AffineStoreOp>(loc, zero, alloc, ValueRange{ivJ});

    // loop body
    auto time_var = rewriter.create<AffineLoadOp>(
        loc, time, timeMap, ValueRange{ivI, ivJ}); // load from time[j]
    auto mul_var = rewriter.create<arith::MulFOp>(
        loc, time_var, phase_var); // time[j] * (2*pi*freq)
    auto sin_body = rewriter.create<arith::AddFOp>(
        loc, mul_var, iter_args); // time[j] * (2*pi*freq) + i*pi/4
    auto result = rewriter.create<math::SinOp>(loc, sin_body);
    rewriter.create<AffineStoreOp>(loc, result, allocSignal,
                                   ValueRange{ivI, ivJ});

    rewriter.setInsertionPointAfter(forOpJ);
    rewriter.setInsertionPointAfter(forOpI);

    ub = antennas;
    affine::AffineForOp forOpIOut =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivIoutput = forOpIOut.getInductionVar();
    rewriter.setInsertionPointToStart(forOpIOut.getBody());

    ub = timeDim;
    affine::AffineForOp forOpJOut =
        rewriter.create<AffineForOp>(loc, lb, ub, step);
    auto ivJoutput = forOpJOut.getInductionVar();
    rewriter.setInsertionPointToStart(forOpJOut.getBody());

    // load from signal input
    auto signalInput = rewriter.create<AffineLoadOp>(
        loc, allocSignal, genInputMap, ValueRange{ivIoutput, ivJoutput});
    auto weight = rewriter.create<AffineLoadOp>(
        loc, weights, outputMap, ValueRange{ivIoutput, ivJoutput});
    auto intermediateVal =
        rewriter.create<arith::MulFOp>(loc, signalInput, weight);

    // load from output
    auto outputVal =
        rewriter.create<AffineLoadOp>(loc, alloc, ValueRange{ivJoutput});
    auto beamOut =
        rewriter.create<arith::AddFOp>(loc, intermediateVal, outputVal);

    rewriter.create<AffineStoreOp>(loc, beamOut, alloc, ValueRange{ivJoutput});

    rewriter.setInsertionPointAfter(forOpJOut);
    rewriter.setInsertionPointAfter(forOpIOut);

    rewriter.replaceOp(op, alloc);

    return mlir::success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// ToyToAffineLoweringPass
//===----------------------------------------------------------------------===//

/// This is a partial lowering to affine loops of the dsp operations that are
/// computationally intensive (like matmul for example...) while keeping the
/// rest of the code in the Toy dialect.
namespace {
struct ToyToAffineLoweringPass
    : public PassWrapper<ToyToAffineLoweringPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ToyToAffineLoweringPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<affine::AffineDialect, func::FuncDialect, memref::MemRefDialect,
                math::MathDialect, scf::SCFDialect>();
  }
  void runOnOperation() final;
};
} // namespace

void ToyToAffineLoweringPass::runOnOperation() {
  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering. In our case, we are lowering to a combination of the
  // `Affine`, `Arith`, `Func`, and `MemRef` dialects.
  target.addLegalDialect<affine::AffineDialect, BuiltinDialect,
                         arith::ArithDialect, func::FuncDialect,
                         memref::MemRefDialect, math::MathDialect,
                         scf::SCFDialect, linalg::LinalgDialect>();

  // We also define the Toy dialect as Illegal so that the conversion will fail
  // if any of these operations are *not* converted. Given that we actually want
  // a partial lowering, we explicitly mark the Toy operations that don't want
  // to lower, `dsp.print`, as `legal`. `dsp.print` will still need its operands
  // to be updated though (as we convert from TensorType to MemRefType), so we
  // only treat it as `legal` if its operands are legal.
  target.addIllegalDialect<dsp::DspDialect>();
  target.addDynamicallyLegalOp<dsp::PrintOp>([](dsp::PrintOp op) {
    return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return llvm::isa<TensorType>(type); });
  });

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  RewritePatternSet patterns(&getContext());
  patterns.add<
      AddOpLowering, ModuloOpLowering, ConstantOpLowering, FuncOpLowering,
      MulOpLowering, PrintOpLowering, ReturnOpLowering, TransposeOpLowering,
      DelayOpLowering, GainOpLowering, SubOpLowering,
      FIRFilterResponseOpLowering, SlidingWindowAvgOpLowering,
      DownSamplingOpLowering, UpSamplingOpLowering,
      LowPassFilter1stOrderOpLowering, HighPassFilterOpLowering,
      FFT1DOpLowering, IFFT1DOpLowering, HammingWindowOpLowering, DCTOpLowering,
      filterOpLowering, DivOpLowering, BitwiseAndOpLowering,
      zeroCrossCountOpLowering, SumOpLowering, SinOpLowering, CosOpLowering,
      SquareOpLowering, FFT1DRealOpLowering, FFT1DImgOpLowering, SincOpLowering,
      GetElemAtIndxOpLowering, SetElemAtIndxOpLowering,
      LowPassFIRFilterOpLowering, HighPassFIRFilterOpLowering,
      GetRangeOfVectorOpLowering, FIRFilterHammingOptimizedOpLowering,
      HighPassFIRHammingOptimizedOpLowering, LMSFilterOpLowering,
      ThresholdOpLowering, QuantizationOpLowering, LMSFilterResponseOpLowering,
      RunLenEncodingOpLowering, FIRFilterResSymmOptimizedOpLowering,
      LengthOpLowering, ReverseInputOpLowering, PaddingOpLowering,
      FIRFilterYSymmOptimizedOpLowering, FFT1DRealSymmOpLowering,
      FFT1DImgConjSymmOpLowering, FFTRealOpLowering, FFTImagOpLowering,
      Conv2DOpLowering, ShiftRightOpLowering, MatmulOpLowering,
      ThresholdUpOpLowering, QamModulateRealOpLowering,
      QamModulateImgOpLowering, QamDemodulateOpLowering, FindPeaksOpLowering,
      BeamFormOpLowering, MedianFilterOpLowering>(&getContext());

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

/// Create a pass for lowering operations in the `Affine` and `Std` dialects,
/// for a subset of the Toy IR (e.g. matmul).
std::unique_ptr<Pass> mlir::dsp::createLowerToAffinePass() {
  return std::make_unique<ToyToAffineLoweringPass>();
}

#pragma GCC diagnostic pop
// #pragma warning(pop)
