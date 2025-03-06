//===-- ShapedOpInterfaces.td - Interfaces for Shaped Ops --*- tablegen -*-===//
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

#ifndef MLIR_INTERFACES_SHAPEINFERENCEOPINTERFACES
#define MLIR_INTERFACES_SHAPEINFERENCEOPINTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// ShapedDimOpInterface
//===----------------------------------------------------------------------===//

// Ops that return the dimension of a shaped value.
// def ShapeInferenceOpInterface : OpInterface<"ShapeInterface"> {
//   let description = [{
//     An interface for ops that return the dimension of a shaped value (such as a
//     tensor or a memref).  It provides access to the source shaped value and to
//     the dimension.
//   }];
//   let cppNamespace = "::mlir";

//   let methods = [
//     InterfaceMethod<
//       /*desc=*/[{
//         Return the shaped value operand. This is the value that the dimension
//         is taken from.
//       }],
//       /*retTy=*/"::mlir::Value",
//       /*methodName=*/"getShapeValue",
//       /*args=*/(ins)
//     >,
//     InterfaceMethod<
//       /*desc=*/[{
//         Return the dimension operand. This can be a constant or an SSA value.
//       }],
//       /*retTy=*/"::mlir::OpFoldResult",
//       /*methodName=*/"getDimension",
//       /*args=*/(ins)
//     >
//   ];

//   let verify = [{
//     return verifyShapedDimOpInterface($_op);
//   }];
// }


def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "void", "inferShapes">
  ];
}

#endif // MLIR_INTERFACES_SHAPEDOPINTERFACES
