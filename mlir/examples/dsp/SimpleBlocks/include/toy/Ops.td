//===- Ops.td - Toy dialect operation definitions ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines the operations of the Toy dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TOY_OPS
#define TOY_OPS

// include "mlir/IR/FunctionInterfaces.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "toy/ShapeInferenceInterface.td"


// Provide a definition of the 'toy' dialect in the ODS framework so that we
// can define our operations.
def Dsp_Dialect : Dialect {
  let name = "dsp";
  let cppNamespace = "::mlir::dsp";
}

// Base class for toy dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class Dsp_Op<string mnemonic, list<Trait> traits = []> :
    Op<Dsp_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Toy Operations
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

// We define a toy operation by inheriting from our base 'Dsp_Op' class above.
// Here we provide the mnemonic and a list of traits for the operation. The
// constant operation is marked as 'Pure' as it is a pure operation
// and may be removed if dead.
def ConstantOp : Dsp_Op<"constant", [Pure]> {
  // Provide a summary and description for this operation. This can be used to
  // auto-generate documentation of the operations within our dialect.
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = dsp.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]>
                        : tensor<2x3xf64>
    ```
  }];

  // The constant operation takes an attribute as the only input.
  let arguments = (ins F64ElementsAttr:$value);

  // The constant operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Add custom build methods for the constant operation. These method populates
  // the `state` that MLIR uses to create operations, i.e. these are used when
  // using `builder.create<ConstantOp>(...)`.
  let builders = [
    // Build a constant with a given constant tensor value.
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,

    // Build a constant with a given constant floating-point value.
    OpBuilder<(ins "double":$value)>,

    // Build a constant with a given constant floating-point value.
    // OpBuilder<(ins "int":$value)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ModuloOp
//===----------------------------------------------------------------------===//

def ModuloOp : Dsp_Op<"modulo",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise modulo operation";
  let description = [{
    The "modulo" operation performs element-wise modulo op between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
  
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// AddOp
//===----------------------------------------------------------------------===//

def AddOp : Dsp_Op<"add",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise addition operation";
  let description = [{
    The "add" operation performs element-wise addition between two tensors.
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building an AddOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// CastOp
//===----------------------------------------------------------------------===//

def CastOp : Dsp_Op<"cast", [
     DeclareOpInterfaceMethods<CastOpInterface>,
     DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
     Pure,
     SameOperandsAndResultShape
  ]> {
  let summary = "shape cast operation";
  let description = [{
    The "cast" operation converts a tensor from one type to an equivalent type
    without changing any data elements. The source and destination types must
    both be tensor types with the same element type. If both are ranked, then
    shape is required to match. The operation is invalid if converting to a
    mismatching constant dimension.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input) `to` type($output)";
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

def FuncOp : Dsp_Op<"func", [
     FunctionOpInterface,IsolatedFromAbove
  ]> {
  let summary = "user defined function operation";
  let description = [{
    The "dsp.func" operation represents a user defined function. These are
    callable SSA-region operations that contain toy computations.

    Example:

    ```mlir
    dsp.func @main() {
      %0 = dsp.constant dense<5.500000e+00> : tensor<f64>
      %1 = dsp.reshape(%0 : tensor<f64>) to tensor<2x2xf64>
      dsp.print %1 : tensor<2x2xf64>
      dsp.return
    }
    ```
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    TypeAttrOf<FunctionType>:$function_type,
    OptionalAttr<DictArrayAttr>:$arg_attrs,
    OptionalAttr<DictArrayAttr>:$res_attrs
  );
  let regions = (region AnyRegion:$body);

  let builders = [OpBuilder<(ins
    "StringRef":$name, "FunctionType":$type,
    CArg<"ArrayRef<NamedAttribute>", "{}">:$attrs)
  >];
  let extraClassDeclaration = [{
    //===------------------------------------------------------------------===//
    // FunctionOpInterface Methods
    //===------------------------------------------------------------------===//

    /// Returns the argument types of this function.
    ArrayRef<Type> getArgumentTypes() { return getFunctionType().getInputs(); }

    /// Returns the result types of this function.
    ArrayRef<Type> getResultTypes() { return getFunctionType().getResults(); }

    /// Returns the region on the function operation that is callable.
    Region *getCallableRegion() { return &getBody(); }
  }];
  let hasCustomAssemblyFormat = 1;
  let skipDefaultBuilders = 1;
}

//===----------------------------------------------------------------------===//
// GenericCallOp
//===----------------------------------------------------------------------===//

def GenericCallOp : Dsp_Op<"generic_call",
    [DeclareOpInterfaceMethods<CallOpInterface>]> {
  let summary = "generic call operation";
  let description = [{
    Generic calls represent calls to a user defined function that needs to
    be specialized for the shape of its arguments. The callee name is attached
    as a symbol reference via an attribute. The arguments list must match the
    arguments expected by the callee. For example:

    ```mlir
     %4 = dsp.generic_call @my_func(%1, %3)
           : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<*xf64>
    ```

    This is only valid if a function named "my_func" exists and takes two
    arguments.
  }];

  // The generic call operation takes a symbol reference attribute as the
  // callee, and inputs for the call.
  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<F64Tensor>:$inputs);

  // The generic call operation returns a single value of TensorType.
  let results = (outs F64Tensor);

  // Specialize assembly printing and parsing using a declarative format.
  let assemblyFormat = [{
    $callee `(` $inputs `)` attr-dict `:` functional-type($inputs, results)
  }];

  // Add custom build methods for the generic call operation.
  let builders = [
    OpBuilder<(ins "StringRef":$callee, "ArrayRef<Value>":$arguments)>
  ];
}

//===----------------------------------------------------------------------===//
// MulOp
//===----------------------------------------------------------------------===//

def MulOp : Dsp_Op<"mul",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise multiplication operation";
  let description = [{
    The "mul" operation performs element-wise multiplication between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];

  let hasCanonicalizer = 1;
}


//===----------------------------------------------------------------------===//
// DivOp
//===----------------------------------------------------------------------===//

def DivOp : Dsp_Op<"div",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "element-wise division operation";
  let description = [{
    The "div" operation performs element-wise division between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building a DivOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// BitwiseAndOp
//===----------------------------------------------------------------------===//

def BitwiseAndOp : Dsp_Op<"bitwiseand",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "bit-wise and operation";
  let description = [{
    The "bitwiseand" operation performs bit-wise and between two
    tensors. The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  let hasCustomAssemblyFormat = 1;

  // Allow building a BitwiseAndOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];
}

//===----------------------------------------------------------------------===//
// PowerOp
//===----------------------------------------------------------------------===//

def PowOp : Dsp_Op<"pow",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]>{
        let summary = "element-wise power operation for tensor";
        let description = [{
            The "pow" operation performs element-wise power for base tensor.
                The accepted operand is restrict to a scaler constant.
        }];

        let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);
        let results = (outs F64Tensor);

        // has custom parser and printer for method
        // FIXME: pow op should have custom assembly format
        // let hasCustomAssemblyFormat = 1;

        // Allow building a PowOp from two operands.
        let builders = [
            OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
        ];
        let hasVerifier = 1;
    }


//===----------------------------------------------------------------------===//
// PrintOp
//===----------------------------------------------------------------------===//

def PrintOp : Dsp_Op<"print"> {
  let summary = "print operation";
  let description = [{
    The "print" builtin operation prints a given input tensor, and produces
    no results.
  }];

  // The print operation takes an input tensor to print.
  // We also allow a F64MemRef to enable interop during partial lowering.
  let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef]>:$input);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

//===----------------------------------------------------------------------===//
// ReshapeOp
//===----------------------------------------------------------------------===//

def ReshapeOp : Dsp_Op<"reshape", [Pure]> {
  let summary = "tensor reshape operation";
  let description = [{
    Reshape operation is transforming its input tensor into a new tensor with
    the same number of elements but different shapes. For example:

    ```mlir
       %0 = dsp.reshape (%arg1 : tensor<10xf64>) to tensor<5x2xf64>
    ```
  }];

  let arguments = (ins F64Tensor:$input);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;

  // We expect that the reshape operation returns a statically shaped tensor.
  let results = (outs StaticShapeTensorOf<[F64]>);
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

def ReturnOp : Dsp_Op<"return", [Pure, HasParent<"FuncOp">,
                                 Terminator]> {
  let summary = "return operation";
  let description = [{
    The "return" operation represents a return operation within a function.
    The operation takes an optional tensor operand and produces no results.
    The operand type must match the signature of the function that contains
    the operation. For example:

    ```mlir
      dsp.func @foo() -> tensor<2xf64> {
        ...
        dsp.return %0 : tensor<2xf64>
      }
    ```
  }];

  // The return operation takes an optional input operand to return. This
  // value must match the return type of the enclosing function.
  let arguments = (ins Variadic<F64Tensor>:$input);

  // The return operation only emits the input in the format if it is present.
  let assemblyFormat = "($input^ `:` type($input))? attr-dict ";

  // Allow building a ReturnOp with no return operand.
  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  // Provide extra utility definitions on the c++ operation class definition.
  let extraClassDeclaration = [{
    bool hasOperand() { return getNumOperands() != 0; }
  }];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

def TransposeOp : Dsp_Op<"transpose",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "transpose operation";

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;

  // Allow building a TransposeOp with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// DelayOp
//===----------------------------------------------------------------------===//

def DelayOp : Dsp_Op<"delay" , 
          [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "shifting tensor by given number";
  let description = [{
    The "delay" operation performs shift in array/tensor by given arg -- 
    ex: shift by 1 -- 1 element will become 0.
    The shapes of the tensor operands are expected to match.
  }];

  // let arguments = (ins F64Tensor:$lhs, UI32:$rhs); //
  // let results = (outs F64Tensor);

  // let arguments = (ins F64Tensor:$lhs, UI32:$rhs); 
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);//working
  let results = (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  // let assemblyFormat = [{
  //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
  // }];
  // Allow building a MulOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;

  // Allow building an Dealy2ArgsOp with from the two input operands.
  // let builders = [
  //   OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  // ];
  
  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;

  let hasVerifier = 1;

}


//===----------------------------------------------------------------------===//
// GainOp
//===----------------------------------------------------------------------===//
def GainOp : Dsp_Op<"gain" ,
    [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "shifting tensor by given number";
  let description = [{
    The "Gain" operation performs multiplication of each element with given gain -- 
    ex: [1, 2, 4, 5] & gain= 2 : Output : [2, 4, 8, 10]
    The shapes of the tensor operands are expected to match.
  }];

  // let arguments = (ins F64Tensor:$lhs, UI32:$rhs); //
  // let results = (outs F64Tensor);

  // let arguments = (ins F64Tensor:$lhs, UI32:$rhs); 
  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);//working -- F64
  // let arguments = (ins F64Tensor:$lhs, F64:$rhs);
  // let arguments = (ins F64Tensor:$lhs, F64Attr:$rhs);
  let results = (outs F64Tensor); //I64
  

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  // let assemblyFormat = [{
  //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
  // }];
  // Allow building a GainOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  
  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;

  // let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SubOp
//===----------------------------------------------------------------------===//

def SubOp : Dsp_Op<"sub", [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
   let summary = "Subtract b from a";
   let description = [{
     The subtract block subtracts each element of a vector with each other.
   }];

   let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);//working
   let results = (outs F64Tensor);

   // Indicate that the operation has a custom parser and printer method.
   // let hasCustomAssemblyFormat = 1;
   // let assemblyFormat = [{
   //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
   // }];
   // Allow building a MulOp with from the two input operands.
   let builders = [
     OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
   ];

   // Indicate that the operation has a custom parser and printer method.
   // let hasCustomAssemblyFormat = 1;

   // let hasVerifier = 1;
 }

//===----------------------------------------------------------------------===//
// FFTRealOp
//===----------------------------------------------------------------------===//

def FFTRealOp : Dsp_Op<"fftReal", [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Performs FFT Operation on the input";
  let description = [{
      This function accepts a 1D input array of size 2^n and returns the real part of its Fourier transform, 
      producing an output array of the same size. The function is designed to work exclusively with input sizes that are powers of 2. 
      Providing an array of any other size will result in a segmentation fault.
  }];

  let arguments = (ins F64Tensor:$lhs);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "Value":$lhs)>
  ]; 
}

//===----------------------------------------------------------------------===//
// FFTImagOp
//===----------------------------------------------------------------------===//

def FFTImagOp : Dsp_Op<"fftImag", [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Performs FFT Operation on the input";
  let description = [{
      This function accepts a 1D input array of size 2^n and returns the imaginary part of its Fourier transform, 
      producing an output array of the same size. The function is designed to work exclusively with input sizes that are powers of 2. 
      Providing an array of any other size will result in a segmentation fault.
  }];

  let arguments = (ins F64Tensor:$lhs);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "Value":$lhs)>
  ]; 
}

//===----------------------------------------------------------------------===//
// zeroCrossCountOp
//===----------------------------------------------------------------------===//
def zeroCrossCountOp : Dsp_Op<"zeroCrossCount" ,
    [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "count the crosses through zero";
  let description = [{
    The "zeroCrossCount" operation detects no of zero crosses in a given array -- 
    ex: [-1 , -2 , 3, 0 , 0, -2] has 2 zero-crosses 
  }];

  let arguments = (ins F64Tensor:$lhs); //working -- F64
  let results =  (outs F64Tensor);
  // let results = (outs I64);

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  // let assemblyFormat = [{
  //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
  // }];
  // Allow building a zeroCrossCountOp with from the one input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs)>
  ];

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  
  // Enable registering canonicalization patterns with this operation.
  //let hasCanonicalizer = 1;

  // let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// FIRFilterResponseOp
//===----------------------------------------------------------------------===//
def FIRFilterResponseOp : Dsp_Op<"FIRFilterResponse" ,
    [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "shifting tensor by given number";
  let description = [{
    The "FIRFilterResponseOp" operation is basically the convolution of input 1-D and filter vector 
    ex: x[n] = [2,1,3,2,4], h[0]=1,h[1]=−1, and h[2]=2 then
      y[n] = sum(h(k) . x(n-k)) k=0 to N-1 
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs); //working -- F64
  let results =  (outs F64Tensor);
  // let results = (outs I64);

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  // let assemblyFormat = [{
  //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
  // }];
  // Allow building a FIRFilterResponseOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  
  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// MedianFilterOp
//===----------------------------------------------------------------------===//

def MedianFilterOp : Dsp_Op<"medianFilter", 
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  
  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];
}

//===----------------------------------------------------------------------===//
// SlidingWindowAvg
//===----------------------------------------------------------------------===//

def SlidingWindowAvgOp : Dsp_Op<"slidingWindowAvg",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "sliding window average operation";

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a SlidingWindowAvg with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}


//===----------------------------------------------------------------------===//
// DownsamplingOp
//===----------------------------------------------------------------------===//
def DownsamplingOp : Dsp_Op<"downsampling" ,
    [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "downsampling a input tensor by given rate";
  let description = [{
    y[m] = x[M . n] ie, m = len(x) / M 
    The "Downsampling" operation gives output with less no of samples:  -- 
    ex: [1, 2, 4, 5] & DownsamplingRate= 2 : Output : [1, 4]
    The shapes of the tensor operands are expected to match.
  }];


  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);//working -- F64
  let results = (outs F64Tensor); //I64
  

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  // let assemblyFormat = [{
  //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
  // }];
  // Allow building a DownsamplingOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  
  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;

  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// UpsamplingOp
//===----------------------------------------------------------------------===//

def UpsamplingOp : Dsp_Op<"upsampling" ,
    [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "upsampling a input tensor by given rate";
  let description = [{
    y[L . m] = x[n] ie, o/p len, m = len(x) * L 
    The "upsampling" operation gives output with less no of samples:  -- 
    ex: [1, 2, 4, 5] & UpsamplingRate= 2 : Output : [1, 0, 2, 0, 4, 0,5 , 0]
    The shapes of the tensor operands are expected to match.
  }];


  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);//working -- F64
  let results = (outs F64Tensor); //I64
  

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  // let assemblyFormat = [{
  //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
  // }];
  // Allow building a UpsamplingOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  
  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// LowPassFilter1stOrderOp
//===----------------------------------------------------------------------===//

def LowPassFilter1stOrderOp : Dsp_Op<"lowPassFilter" ,
    [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "lowPassFilter a input tensor by given rate";
  let description = [{
    y[n] = (1-alpha) * y[n-1] + alpha * x[n]  
    The "lowPassFilter" operation filters and gives low freq components based on alpha :  -- 
    ex: [1, 2, 4, 5] & UpsamplingRate= 2 : Output : [1, 0, 2, 0, 4, 0,5 , 0]
    The shapes of the tensor operands are expected to match.
  }];


  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs);//working -- F64
  let results = (outs F64Tensor); //I64
  

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  // let assemblyFormat = [{
  //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
  // }];
  // Allow building a LowPassFilter1stOrderOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  
  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// HighPassFilter
//===----------------------------------------------------------------------===//


def HighPassFilterOp : Dsp_Op<"highPassFilter",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "high pass filter operation";
  let description = [{
    y[n] = x[n] - x[n-1] ie, H(z) = 1 - pow(z,-1) 
    The "HighPassFilter" operation gives output with low freq components removed:  -- 
    ex: x=[0,1,0,−1,0] &  Output : [0, 1,-1,-1,1]
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` type(results)
  // }];

  // let assemblyFormat = [{
  //   `(` $input `)` `:` `(` type($input) `)` attr-dict `to` type(results)
  // }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a HighPassFilter with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FFT1D
//===----------------------------------------------------------------------===//


def FFT1DOp : Dsp_Op<"fft1d",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "fft1d operation";
  let description = [{
    y[k] = y_real[k] + j *y_img[k] 
    y_real = sumOver_n(x[n]*cos[2*pi * k *n/N ] 
    y_img = sumOver_n(x[n]*sin[2*pi * k *n/N ] * -1
    The "FFT1D" operation gives output in freq domain  -- 
    ex: x=[10,0,0,0] &  Output : [10,10,10,10] , [10,10,]
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$real, F64Tensor:$img); //(ins F64Tensor:$lhs, F64Tensor:$rhs);
  // let results = (outs F64Tensor:$real);
  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` type(results)
  // }];

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` `(` type($real) `,` type($img) `)`
  }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a FFT1D with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$lhs)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// IFFT1D
//===----------------------------------------------------------------------===//


def IFFT1DOp : Dsp_Op<"ifft1d",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ifft1d operation";
  let description = [{
    y[k] = y_real[k] + j *y_img[k] 
    y_real = sumOver_n(x[n]*cos[2*pi * k *n/N ] 
    y_img = sumOver_n(x[n]*sin[2*pi * k *n/N ] 
    The "IFFT1D" operation gives output in freq domain  -- 
    ex: x=[10,0,0,0] &  Output : real= [10,10,10,10] , img= j[0,0,0,0]
    The shapes of the tensor operands are expected to match.
  }];

  // let arguments = (ins F64Tensor:$input ,  F64Tensor:$input);
  let arguments = (ins F64Tensor:$real ,  F64Tensor:$img);
  // let results = (outs F64Tensor:$real, F64Tensor:$img); //(ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
  // let results = (outs F64Tensor:$real);
  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` type(results)
  // }];

  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` `(` type($real) `,` type($img) `)`
  // }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a IFFT1D with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$real, "Value":$img)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// HammingWindow
//===----------------------------------------------------------------------===//


def HammingWindowOp : Dsp_Op<"hamming",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "hamming operation";
  let description = [{
    y[n] = 0.54 - 0.46 cos(2 *pi * n/N-1) , 0<=n<N : Taking length as N 0 to N-1
    
    The "HammingWindow" operation calculates the 
    ex: N = 5 &  Output : y= [0.080000 0.540000 1.000000 0.540000 0.0800004] 
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$input);
  // let results = (outs F64Tensor:$real, F64Tensor:$img); //(ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` `(` type($real) `,` type($img) `)`
  // }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a HammingWindow with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$lhs)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}



//===----------------------------------------------------------------------===//
// DCT-II
//===----------------------------------------------------------------------===//


def DCTOp : Dsp_Op<"dct",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "dct-II operation";
  let description = [{
    y[k] = sqrt(2/N) * SumOverAllN( x[n] cos(pi * k * (n +0.5)/N)) , 0<=n<=N-1 :
    
    for y[0] , the answer will be multiplied by 1/sqrt(2)
    The "DCT" operation calculates the dct 2 
    ex: N = 4 &  Output : y= [0.08,0.54,1,0.54] 
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$input);
  // let results = (outs F64Tensor:$real, F64Tensor:$img); //(ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` `(` type($real) `,` type($img) `)`
  // }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a DCT with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$lhs)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// filterOp
//===----------------------------------------------------------------------===//

def filterOp : Dsp_Op<"filter" ,
    [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "get filter response given b,a and x";
  let description = [{
    The "filterOp" operation is basically the filter response of input x and filter represented as b/a rational transfer function form
    ex: x[n] = [2,1,3,2,4], b[0,1,2,3,4] and a[1,2,3,4,5] then
      y[i] = sum(b[j] * x(i-j) - a[j] *x[i-j] ) j=1 to i and  i=1 to len(x)
      also, y[0] = b[0] * x[0]
  }];

  let arguments = (ins F64Tensor:$b, F64Tensor:$a, F64Tensor:$x); //working -- F64
  let results =  (outs F64Tensor);
  
  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  // let assemblyFormat = [{
  //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
  // }];
  // Allow building a filterOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$b, "Value":$a, "Value":$x)>
  ];

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  
  // Enable registering canonicalization patterns with this operation.
  //let hasCanonicalizer = 1;

  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// SumOp
//===----------------------------------------------------------------------===//

def SumOp : Dsp_Op<"sum",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "sum of all the elements operation";

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a Sum with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SinOp
//===----------------------------------------------------------------------===//

 def SinOp : Dsp_Op<"sin",
     [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
   let summary = "sin operation";
   let description = [{
     The sin operation computes the sine of a given value. It takes one operand of floating point type (i.e., scalar, tensor or vector) and returns one result of the same type. It has no standard attributes.
   }];

   let arguments = (ins F64Tensor:$input);
   // let results = (outs F64Tensor:$real, F64Tensor:$img); //(ins F64Tensor:$lhs, F64Tensor:$rhs);
   let results = (outs F64Tensor);
   let assemblyFormat = [{
     `(` $input `:` type($input) `)` attr-dict `to` type(results)
   }];

   // let assemblyFormat = [{
   //   `(` $input `:` type($input) `)` attr-dict `to` `(` type($real) `,` type($img) `)`
   // }];

   // Enable registering canonicalization patterns with this operation.
   // let hasCanonicalizer = 1;

   // Allow building a DCT with from the input operand.
   let builders = [
     OpBuilder<(ins "Value":$lhs)>
   ];

   // Indicate that additional verification for this operation is necessary.
   let hasVerifier = 1;
 }

//===----------------------------------------------------------------------===//
// CosOp
//===----------------------------------------------------------------------===//

 def CosOp : Dsp_Op<"cos",
     [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
   let summary = "cos operation";
   let description = [{
     The cos operation computes the cosine of a given value. It takes one operand of floating point type (i.e., scalar, tensor or vector) and returns one result of the same type. It has no standard attributes.
   }];

   let arguments = (ins F64Tensor:$input);
   // let results = (outs F64Tensor:$real, F64Tensor:$img); //(ins F64Tensor:$lhs, F64Tensor:$rhs);
   let results = (outs F64Tensor);
   let assemblyFormat = [{
     `(` $input `:` type($input) `)` attr-dict `to` type(results)
   }];

   // let assemblyFormat = [{
   //   `(` $input `:` type($input) `)` attr-dict `to` `(` type($real) `,` type($img) `)`
   // }];

   // Enable registering canonicalization patterns with this operation.
   // let hasCanonicalizer = 1;

   // Allow building a DCT with from the input operand.
   let builders = [
     OpBuilder<(ins "Value":$lhs)>
   ];

   // Indicate that additional verification for this operation is necessary.
   let hasVerifier = 1;
 }


 //===----------------------------------------------------------------------===//
// SquareOp
//===----------------------------------------------------------------------===//

def SquareOp : Dsp_Op<"square",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "square of the input vector";

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  let assemblyFormat = [{
    `(` $input `:` type($input) `)` attr-dict `to` type(results)
  }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a Square with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// FFT1DRealOp
//===----------------------------------------------------------------------===//
def FFT1DRealOp : Dsp_Op<"fft1dreal",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "fft1dreal operation";
  let description = [{
    y[k] = y_real[k]
    y_real = sumOver_n(x[n]*cos[2*pi * k *n/N ] 
    The "FFT1DReal" operation gives output in freq domain  -- 
    ex: x=[10,0,0,0] &  Output : [10,10,10,10] , [10,10,]
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$real); //(ins F64Tensor:$lhs, F64Tensor:$rhs);
  // let results = (outs F64Tensor:$real);
  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` type(results)
  // }];

  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` `(` type($real) `,` type($img) `)`
  // }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a FFT1DReal with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$lhs)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;

  let hasCanonicalizer = 1;
}


//===----------------------------------------------------------------------===//
// FFT1DImgOp
//===----------------------------------------------------------------------===//
def FFT1DImgOp : Dsp_Op<"fft1dimg",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "fft1dimg operation";
  let description = [{
    y[k] = y_real[k]
    y_real = sumOver_n(x[n]*sin[2*pi * k *n/N ] * -1
    The "FFT1DImg" operation gives output in freq domain  -- 
    ex: x=[10,0,0,0] &  Output : [10,10,10,10] , [10,10,]
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$real); //(ins F64Tensor:$lhs, F64Tensor:$rhs);
  // let results = (outs F64Tensor:$real);
  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` type(results)
  // }];

  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` `(` type($real) `,` type($img) `)`
  // }];

  // Enable registering canonicalization patterns with this operation.
  let hasCanonicalizer = 1;

  // Allow building a FFT1DImg with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$lhs)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// SincOp
//===----------------------------------------------------------------------===//


def SincOp : Dsp_Op<"sinc",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "sinc operation";
  let description = [{
    y = sinc(wc * n) = [1, sin(wc)/pi , sin(2* wc)/2*pi , ... sin(n * wc)/n*pi]
    The "SINc" operation gives sin(wc *n)/n*pi 
    ex: x=[10,0,0,0] &  Output : [10,10,10,10] , [10,10,]
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$wc , F64Tensor:$n);
  let results = (outs F64Tensor); //(ins F64Tensor:$lhs, F64Tensor:$rhs);
  // let results = (outs F64Tensor:$real);
  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` type(results)
  // }];

  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` `(` type($real) `,` type($img) `)`
  // }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a SINc with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$wc, "Value":$n)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GetElemAtIndxOp
//===----------------------------------------------------------------------===//

def GetElemAtIndxOp : Dsp_Op<"getElemAtIndx",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "get elem at given indx of the vector";

  let arguments = (ins F64Tensor:$input, F64Tensor:$indx);
  let results = (outs F64Tensor);

  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` type(results)
  // }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a GetElemAtIndx with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$indx)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}






//===----------------------------------------------------------------------===//
// GetSingleElemAtIdxOp
//===----------------------------------------------------------------------===//

def GetSingleElemAtIdxOp : Dsp_Op<"getSingleElemAtIndx",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "This one access ranked tensor at idx and returns signle tensor without dimension.";

  let arguments = (ins F64Tensor:$input, F64Tensor:$indx);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$indx)>
  ];

  // Indicate that additional verification for this operation is necessary.
  //let hasVerifier = 1;
}




//===----------------------------------------------------------------------===//
// Diff2MeanOptimizedOp
//===----------------------------------------------------------------------===//

def Diff2MeanOptimizedOp : Dsp_Op<"diff2meanOpt",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "This one implemnets mean(diff(input)) as (input[-1] - input[0])/len(input). Note that mean uses length of diff, this operation consider input[-1] == input[length paramter of mean], not input[length paramter of mean-1]";

  let arguments = (ins F64Tensor:$input, F64Tensor:$length);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$length)>
  ];
}



//===----------------------------------------------------------------------===//
// FindPeaks2Diff2MeanOptimizedOp
//===----------------------------------------------------------------------===//

def FindPeaks2Diff2MeanOptimizedOp : Dsp_Op<"findpeaks2diff2meanOpt",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "This one implemnets mean(diff(find_peaks(input))) as (peak[-1] - peak[0])/(len(peaks)-1).";

   let arguments = (ins F64Tensor:$signal, F64Tensor:$height, F64Tensor:$distance);
   let results = (outs F64Tensor);
   
   let builders = [
     OpBuilder<(ins "Value":$signal, "Value":$height, "Value":$distance)>
   ];
}







//===----------------------------------------------------------------------===//
// LMS2FindPeaksOptimizedOp
//===----------------------------------------------------------------------===//

def LMS2FindPeaksOptimizedOp : Dsp_Op<"lms2findPeaks",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "Fusing loop for LMSFilterResponseOp and FindPeaksOp";

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs, F64Tensor:$mu, F64Tensor:$filterLen, F64Tensor:$height, F64Tensor:$distance);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$mu, "Value":$filterLen, "Value":$height, "Value":$distance)>
  ];
}






//===----------------------------------------------------------------------===//
// SetElemAtIndxOp
//===----------------------------------------------------------------------===//

def SetElemAtIndxOp : Dsp_Op<"setElemAtIndx",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "set elem at given indx of the vector";

  let arguments = (ins F64Tensor:$input, F64Tensor:$indx , F64Tensor:$val);
  let results = (outs F64Tensor);

  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` type(results)
  // }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a SetElemAtIndx with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$indx, "Value":$val)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LowPassFilter
//===----------------------------------------------------------------------===//


def LowPassFIRFilterOp : Dsp_Op<"lowPassFIRFilter",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "lowPassFilter operation";
  let description = [{
    y_lpf[n] = wc/pi * sinc(wc * (n- (N-1)/2)) , n!= (N-1)/2 : 
             = wc/pi , n = (N-1)/2
    
    The "LowpassFilter" operation is the base filter 
    ex: N = odd for symmetry &  Output : y= [0.08,0.54,1,0.54] 
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$wc, F64Tensor:$n);
  // let results = (outs F64Tensor:$real, F64Tensor:$img); //(ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` type(results)
  // }];

  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` `(` type($real) `,` type($img) `)`
  // }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a LowpassFilter with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$wc,"Value":$n)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// LMSFilterOp
//===----------------------------------------------------------------------===//


def LMSFilterOp : Dsp_Op<"lmsFilter",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "LMS filter";
  let description = [{
    When you select LMS for the Algorithm parameter, the block calculates the filter weights by using the least mean-square (LMS) algorithm. This algorithm is defined by these equations.
    y(n)=wT(n−1)u(n)
    e(n)=d(n)−y(n)
    w(n)=αw(n−1)+f(u(n),e(n),μ)
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs, F64Tensor:$mu, F64Tensor:$filterLen, F64Tensor:$iters);
   let results = (outs F64Tensor);
  /// Indicate that the operation has a custom parser and printer method.
   // let hasCustomAssemblyFormat = 1;
   // let assemblyFormat = [{
   //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
   // }];
   // Allow building a LMSFilterOp with from the two input operands.
   let builders = [
     OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$mu, "Value":$filterLen, "Value":$iters)>
   ];

   // Indicate that the operation has a custom parser and printer method.
   // let hasCustomAssemblyFormat = 1;

   // let hasVerifier = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// HighPassFilter
//===----------------------------------------------------------------------===//


def HighPassFIRFilterOp : Dsp_Op<"highPassFIRFilter",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "highPassFIRFilter operation";
  let description = [{
   y_hpf[n] = dirac(n-(N-1)/2) - y_lpf[n] = -1 * wc/pi * sinc(wc * (n- (N-1)/2)) , n!= (N-1)/2 : 
             = 1 - wc/pi , n = (N-1)/2
    
    The "highPassFIRFilter" operation gives the coefficients of high pass filter  for 
    cut-off freq, wc= 2*pi*fc and filter order, N .. choose N=odd for symmetry
    ex: N = odd for symmetry &  Output : y= [0.08,0.54,1,0.54] 
  }];

  let arguments = (ins F64Tensor:$wc, F64Tensor:$N);
  let results = (outs F64Tensor);
  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` type(results)
  // }];

  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` `(` type($real) `,` type($img) `)`
  // }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a LowpassFilter with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$wc,"Value":$N)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// GetRangeOfVector
//===----------------------------------------------------------------------===//


def GetRangeOfVectorOp : Dsp_Op<"getRangeOfVector",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "getRangeOfVector operation";
  let description = [{
   y[n] = [first to N elements using step]  -- similar to np.arange(first,last,step)
   But here, the difference is we are mentioning NoOfElements and not the last element
    
    The "getRangeOfVector" operation gives the vector from first to last --
    Ex: y = getRangeOfVector(0,5,2) => y = [0 2 4 6 8]
    
  }];

  let arguments = (ins F64Tensor:$first, F64Tensor:$N, F64Tensor:$step);
  let results = (outs F64Tensor);
  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` type(results)
  // }];

  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` `(` type($real) `,` type($img) `)`
  // }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a LowpassFilter with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$first,"Value":$N, "Value":$step)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// FIRFilterHammingOptimizedOp
//===----------------------------------------------------------------------===//


def FIRFilterHammingOptimizedOp : Dsp_Op<"FIRFilterHammingOptimized",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "FIRFilterHammingOptimized operation";
  let description = [{
    y_FIRHamming[n] = [wc/pi * sinc(wc * (n- (N-1)/2))] * [0.54 - 0.46 cos(2 *pi * n/N-1)], 0<= n < (N-1)/2 : 
             = wc/pi * 1 , n = (N-1)/2

             and also, y_FIRHamming[N-1-n] = y[n]
    
    The "FIRFilterHammingOptimized" operation is the fusion of 2 operations and symmetry behaviour
    ex: N = odd for symmetry &  Output for wc=2*pi*1/8 , N = 7 : y= 0.006002 0.049338 0.173311 0.250000 0.173311 0.049338 0.006002]
    
  }];

  let arguments = (ins F64Tensor:$wc, F64Tensor:$n);
  // let results = (outs F64Tensor:$real, F64Tensor:$img); //(ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` type(results)
  // }];

  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` `(` type($real) `,` type($img) `)`
  // }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a LowpassFilter with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$wc,"Value":$n)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// HighPassFIRHammingOptimizedOp
//===----------------------------------------------------------------------===//


def HighPassFIRHammingOptimizedOp : Dsp_Op<"highPassFIRHammingOptimizedOp",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "highPassFIRHammingOptimizedOp operation";
  let description = [{
    y_highFIRHamming[n] = -1 * [wc/pi * sinc(wc * (n- (N-1)/2))] * [0.54 - 0.46 cos(2 *pi * n/N-1)], 0<= n < (N-1)/2 : 
             = 1 - wc/pi , n = (N-1)/2

             and also, y_FIRHamming[N-1-n] = y[n]
    
    The "highPassFIRHammingOptimizedOp" operation is the fusion of 2 operations and symmetry behaviour
    ex: N = odd for symmetry &  Output for wc=2*pi*1/8 , N = 7 : y= [-0.006002 -0.049338 -0.173311 0.750000 -0.173311 -0.049338 -0.00600]
    
  }];

  let arguments = (ins F64Tensor:$wc, F64Tensor:$n);
  // let results = (outs F64Tensor:$real, F64Tensor:$img); //(ins F64Tensor:$lhs, F64Tensor:$rhs);
  let results = (outs F64Tensor);
  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` type(results)
  // }];

  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` `(` type($real) `,` type($img) `)`
  // }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a LowpassFilter with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$wc,"Value":$n)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ThresholdOp
//===----------------------------------------------------------------------===//

def ThresholdOp : Dsp_Op<"threshold",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "limit the array values";
  let description = [{
   y_threshold[n] = a[i]  if a[i] >= threshld or, a[i] <= -threshld
                  = 0 , else
    
    The "threshold" operation returns an array with only those values retained 
    ex: Input x= [-0.08,-5.4,-1,3.4] and threshld = 2 , then y[n] = [ 0, -5.4, 0, 3.4]  
  }];

  let arguments = (ins F64Tensor:$input, F64Tensor:$threshld);
  let results = (outs F64Tensor);

  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` type(results)
  // }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a GetElemAtIndx with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$threshld)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// QuantizationOp
//===----------------------------------------------------------------------===//


def QuantizationOp : Dsp_Op<"quantization",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "quantization operation";
  let description = [{
    y_quantized[i] = Round(a[i] - min) / step) * step + min
        where, step = (max-min)/NoOfLevels 
    
    The "quantization" operation steps:
    1) calulate NoOfLevels = 2^NoOfBits -- separate
    2) Then calculate stepSize = (Max-Min)/NoOfLevels
    3) iterate for all the elements and calculate quantizedCoeff

    GetLevelForVal =  (a[i] - min)/step
    RoundedVal = arith.FPToSI(GetLevelForVal)
    QuantVal = RoundedVal * step + min_val

    ex: Original coefficients: [ 3.2 -1.5  0.8 -2.9  4.5] & numLevels = 16, max=5, min=-5
        Quantized coefficients: [ 3.125 -1.25   0.625 -3.125  4.375]
  }];

  let arguments = (ins F64Tensor:$input, F64Tensor:$Nlevels, F64Tensor:$max, F64Tensor:$min);
  let results = (outs F64Tensor);

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a LowpassFilter with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$input, "Value":$Nlevels, "Value":$max, "Value":$min)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LMSFilterReponseOp
//===----------------------------------------------------------------------===//


def LMSFilterResponseOp : Dsp_Op<"lmsFilterResponse",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "LMS filter Response";
  let description = [{
    When you select LMS for the Algorithm parameter, the block calculates the filter weights by using the least mean-square (LMS) algorithm and returns the predicted output i.e y(n). This algorithm is defined by these equations.
    y(n)=wT(n−1)u(n)
    e(n)=d(n)−y(n)
    w(n)=αw(n−1)+f(u(n),e(n),μ)
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs, F64Tensor:$mu, F64Tensor:$filterLen);
   let results = (outs F64Tensor);
  /// Indicate that the operation has a custom parser and printer method.
   // let hasCustomAssemblyFormat = 1;
   // let assemblyFormat = [{
   //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
   // }];
   // Allow building a LMSFilterOp with from the two input operands.
   let builders = [
     OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$mu, "Value":$filterLen)>
   ];

   // Indicate that the operation has a custom parser and printer method.
   // let hasCustomAssemblyFormat = 1;

   // let hasVerifier = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// RunLenEncodingOp
//===----------------------------------------------------------------------===//


def RunLenEncodingOp : Dsp_Op<"runLenEncoding",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "runLenEncoding operation";
  let description = [{
    y_rle[i] =  x[i] , if x[i] != x[i-1] , 1<=i<n
                CountOfXi , at n<=i < 2n -1

     ex: if input is [4,4,2,-1,-1,2,2,2] inputLen = 8
    then output is [4,2,-1,2, 0,0,0,0 , 2,1,2,3,0,0,0,0]
    }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a LowpassFilter with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FIRFilterResSymmOptimizedOp
//===----------------------------------------------------------------------===//
def FIRFilterResSymmOptimizedOp : Dsp_Op<"FIRFilterResSymmOptimized" ,
    [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "FIRFilterResSymmOptimizedOp";
  let description = [{
    The "FIRFilterResSymmOptimizedOp" operation is basically the convolution of input 1-D and filter vector when the filter is symmetrical ie,
    h[0] = h[L-1] , h[1] = h[L-2] .. h[middle ie, (L-1)/2] is single element & filter length is odd always
    ex: x[n] = [2,1,3,2,4], h[l] = [1, -1 ,1]  then
      y[n] = sum(h(k) . x(n-k)) k=0 to N-1 can be rewritten as 
      y[n] = sum(h[k] .{ x[n-k] + x[n-(L-1-k)]}) + h[L-1/2].x[n-(L-1)/2] , k=0 to L-1/2

      Basically, we are trying to reduce the number of load/store operations by half -- so that we can reduce the operations
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs); //working -- F64
  let results =  (outs F64Tensor);
  // let results = (outs I64);

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  // let assemblyFormat = [{
  //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
  // }];
  // Allow building a FIRFilterOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  
  // Enable registering canonicalization patterns with this operation.
  //let hasCanonicalizer = 1;

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// LengthOp
//===----------------------------------------------------------------------===//
def LengthOp : Dsp_Op<"len" ,
    [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "LengthOp";
  let description = [{
    The "LengthOp" operation gives the length of vector
    ex: x[n] = [2,1,3,2,4], len(x) will be 5
    }];

  let arguments = (ins F64Tensor:$input); //working -- F64
  let results =  (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  // let assemblyFormat = [{
  //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
  // }];
  // Allow building a FIRFilterOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  
  // Enable registering canonicalization patterns with this operation.
  //let hasCanonicalizer = 1;

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ReverseInputOp
//===----------------------------------------------------------------------===//
def ReverseInputOp : Dsp_Op<"reverseInput" ,
    [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ReverseInputOp";
  let description = [{
    The "ReverseInputOp" operation gives the reverse of vector
    y[n] = x[N-1-n] ie, where N is length of input
    ex: x[n] = [1,2,3,4], y[n] = [4,3,2,1]
    }];

  let arguments = (ins F64Tensor:$input); //working -- F64
  let results =  (outs F64Tensor);

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  // let assemblyFormat = [{
  //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
  // }];
  // Allow building a ReverseInputOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  
  // Enable registering canonicalization patterns with this operation.
  //let hasCanonicalizer = 1;

  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// PaddingOp
//===----------------------------------------------------------------------===//


def PaddingOp : Dsp_Op<"padding",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "PaddingOp(input , ValueToBePadded , PadLen) ";
  let description = [{ 
    PaddingOp : y[n] = x[n] for 0<=n< N
                y[n] = val  for N<=n< N+PadLen 
    ex: x[n] = [1,2,3,4] & padding(x, value=0, len=3) then 
        y[n] = [1,2,3,4,0,0,0]
  }];

  let arguments = (ins F64Tensor:$input, F64Tensor:$PadValue, F64Tensor:$PadLen);
   let results = (outs F64Tensor);
  /// Indicate that the operation has a custom parser and printer method.
   // let hasCustomAssemblyFormat = 1;
   // let assemblyFormat = [{
   //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
   // }];
   // Allow building a LMSFilterOp with from the two input operands.
   let builders = [
     OpBuilder<(ins "Value":$input, "Value":$PadValue, "Value":$PadLen)>
   ];

   // Indicate that the operation has a custom parser and printer method.
   // let hasCustomAssemblyFormat = 1;

   // let hasVerifier = 1;
  let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// FIRFilterYSymmOptimizedOp
//===----------------------------------------------------------------------===//
def FIRFilterYSymmOptimizedOp : Dsp_Op<"FIRFilterYSymmOptimized" ,
    [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "FIRFilterYSymmOptimizedOp";
  let description = [{
    The "FIRFilterYSymmOptimizedOp" operation is basically the convolution of input 1-D and filter vector when filter is reverse of input then output is symmetrical ie,
    say, x[n] convolution x[-n] ie, x[-n] = x[N-1-n]
    ex: x[n] = [1,2], x[-n] = x[N-1-n] = [2,1]  then
        y[n] is always of odd length & is symmetrical about middle
        ie, y[n] = y[N-1-n] : we will reduce half the computations
      y[n] = sum(h(k) . x(n-k)) k=0 to N-1 can be rewritten as 
      
      So, we are reducing the number of outer loops -- 
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs); //working -- F64
  let results =  (outs F64Tensor);
  // let results = (outs I64);

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  // let assemblyFormat = [{
  //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
  // }];
  // Allow building a FIRFilterYSymmOptimizedOp with from the two input operands.
  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
  ];

  // Indicate that the operation has a custom parser and printer method.
  // let hasCustomAssemblyFormat = 1;
  
  // Enable registering canonicalization patterns with this operation.
  //let hasCanonicalizer = 1;

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FFT1DRealSymmOp
//===----------------------------------------------------------------------===//
def FFT1DRealSymmOp : Dsp_Op<"fft1DRealSymm",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "fft1DRealSymm operation";
  let description = [{
    Actual Def: y[k] = y_real[k] & y_real = sumOver_n(x[n]*cos[2*pi * k *n/N ] 
    Here: y[k] = sumOver_n(x[n]*cos[2*pi * k *n/N ] , 0<=k < (N+1)/2
            & y[N-k] = y[k]  (N+1)/2<= k< N
            ie,  y[1] = y[N-1] , y[2] = y[N-2] , 
    The "fft1DRealSymm" operation gives symmetric output when input is symmetric & real 
    ie: when x = [1,2,3,2,1] then y_real will be symm: except the first elem
    ie, y = [9, -2.11 , 0.118, 0.118 , -2.11] 

    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$real); //(ins F64Tensor:$lhs, F64Tensor:$rhs);
  // let results = (outs F64Tensor:$real);
  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` type(results)
  // }];

  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` `(` type($real) `,` type($img) `)`
  // }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a fft1DRealSymm with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$lhs)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;

  // let hasCanonicalizer = 1;
}


//===----------------------------------------------------------------------===//
// FFT1DImgConjSymmOp
//===----------------------------------------------------------------------===//
def FFT1DImgConjSymmOp : Dsp_Op<"fft1DimgConjSymm",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "fft1DimgConjSymm operation";
  let description = [{
    y[k] = y_img[k]
    y_img = sumOver_n(x[n]*sin[2*pi * k *n/N ] * -1
    & when the input is real & symmetric then output is symmetric ie,
    y[k] = sumOver_n(x[n]*sin[2*pi * k *n/N ] * -1 0<=k<(N+1)/2
        & y[N-k] = -y[k] , (N+1)/2 <=k < N
    The "fft1DimgConjSymm" operation gives output in freq domain  -- 
    ex: x=[1,2,3,2,1] &  Output : [0,-1.53,0.36,-0.36,1.53] , 
    The shapes of the tensor operands are expected to match.
  }];

  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor:$real); //(ins F64Tensor:$lhs, F64Tensor:$rhs);
  // let results = (outs F64Tensor:$real);
  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` type(results)
  // }];

  // let assemblyFormat = [{
  //   `(` $input `:` type($input) `)` attr-dict `to` `(` type($real) `,` type($img) `)`
  // }];

  // Enable registering canonicalization patterns with this operation.
  // let hasCanonicalizer = 1;

  // Allow building a fft1DimgConjSymm with from the input operand.
  let builders = [
    OpBuilder<(ins "Value":$lhs)>
  ];

  // Indicate that additional verification for this operation is necessary.
  let hasVerifier = 1;
}



//===----------------------------------------------------------------------===//
// ShiftRightOp
//===----------------------------------------------------------------------===//

def ShiftRightOp : Dsp_Op<"shiftRight", [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
   let summary = "Bit-wise shift right a by b";
   let description = [{
     The shift right block shifts each element of a vector by right-hand side integer. 
   }]; 

   let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs); //Problem: how can we do logical shift with floating point tensor?
   let results = (outs F64Tensor);

   // Indicate that the operation has a custom parser and printer method.
   // let hasCustomAssemblyFormat = 1;
   // let assemblyFormat = [{
   //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
   // }];
   // Allow building a MulOp with from the two input operands.
   let builders = [
     OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
   ];

   // Indicate that the operation has a custom parser and printer method.
   // let hasCustomAssemblyFormat = 1;

   // let hasVerifier = 1;
 }


//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

def MatmulOp : Dsp_Op<"matmul", [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
   let summary = "Matrix multiplication a * b";
   let description = [{
     Matrix multiplication between the left-hand side and right-hand side.
   }]; 

   let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs); //Problem: how can we do logical shift with floating point tensor?
   let results = (outs F64Tensor);

   // Indicate that the operation has a custom parser and printer method.
   // let hasCustomAssemblyFormat = 1;
   // let assemblyFormat = [{
   //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
   // }];
   // Allow building a MulOp with from the two input operands.
   let builders = [
     OpBuilder<(ins "Value":$lhs, "Value":$rhs)>
   ];

   // Indicate that the operation has a custom parser and printer method.
   // let hasCustomAssemblyFormat = 1;

   let hasVerifier = 1;
 }




//===----------------------------------------------------------------------===//
// Conv2DOp
//===----------------------------------------------------------------------===//

def Conv2DOp : Dsp_Op<"conv2d", 
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "Dsp dialect convolution 2d operation";
    let description = [{
        Performs a 2D convolution on the input tensor using specified kernel.
    }];

    let arguments = (ins F64Tensor:$input, F64Tensor:$kernel, F64Tensor:$bias);

    let results = (outs F64Tensor:$output);

    let builders = [
        OpBuilder<(ins "Value":$input, "Value":$kernel, "Value":$bias)>
    ];

    let extraClassDeclaration = [{
        static StringRef getStrideName() { return "stride"; }
        static StringRef getPaddingName() { return "padding"; }
    }];

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// ThresholdUpOp
//===----------------------------------------------------------------------===//

def ThresholdUpOp : Dsp_Op<"thresholdUp", 
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "Converts all the values above threhold to 1 else 0";
    let description = [{
        Converts all the values above threhold to 1 if returnoriginal is false, returns original value if returnoriginal is true else 0
    }];

    let arguments = (ins F64Tensor:$input, F64Tensor:$threshold, F64Tensor:$returnoriginal);

    let results = (outs F64Tensor:$output);

    let builders = [
        OpBuilder<(ins "Value":$input, "Value":$threshold, "Value":$returnoriginal)>
    ];
    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// GenerateDTMFOp
//===----------------------------------------------------------------------===//

def GenerateDTMFOp : Dsp_Op<"generateDtmf", 
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "Generates signal for the character input";
    let description = [{
        Converts the characterr into signal
    }];

    let arguments = (ins F64Tensor:$input, F64Tensor:$duration, F64Tensor:$freq);

    let results = (outs F64Tensor:$output);

    let builders = [
        OpBuilder<(ins "Value":$input, "Value":$duration, "Value":$freq)>
    ];
    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// QamModulateRealOp real
//===----------------------------------------------------------------------===//

def QamModulateRealOp : Dsp_Op<"qam_modulate_real", 
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "Dsp dialect qam modulation real operation";
    let description = [{
        Performs a digital modulation on input tensor.
    }];

    let arguments = (ins F64Tensor:$signal);

    let results = (outs F64Tensor:$real);


    let builders = [
        OpBuilder<(ins "Value":$signal)>
    ];

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// QamModulateImgOp imagine
//===----------------------------------------------------------------------===//

def QamModulateImgOp : Dsp_Op<"qam_modulate_imagine", 
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "Dsp dialect qam modulation imagine operation";
    let description = [{
        Performs a digital modulation on input tensor.
    }];

    let arguments = (ins F64Tensor:$signal);

    let results = (outs F64Tensor:$imagine);


    let builders = [
        OpBuilder<(ins "Value":$signal)>
    ];

    let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// QamDemodulateOp
//===----------------------------------------------------------------------===//

def QamDemodulateOp : Dsp_Op<"qam_demodulate", 
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "Dsp dialect qam demodulation operation";
    let description = [{
        Takes in 2 arrays, one is the real part of a signal the other is the imaginary part of a signal.
        Returns the decoded binary output.
    }];

    let arguments = (ins F64Tensor:$real, F64Tensor:$imagine);

    let results = (outs F64Tensor:$output);

    let builders = [
        OpBuilder<(ins "Value":$real, "Value":$imagine)>
    ];


    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// FindPeaksOp
//===----------------------------------------------------------------------===//

def FindPeaksOp : Dsp_Op<"find_peaks", [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
   let summary = "Find peaks from the signal. Since the number of peaks vary, the output is initialized as -1.";
   let description = [{
		Input: signal, height, distance
		Output: indices of peaks. All of none-used values are initialized as -1, so the length can be measured by this.

		Functionality: check the below original python-level code.
		
		def manual_find_peaks(signal, height, distance):
			peaks = []
			for i in range(1, len(signal) - 1):
				# Check if the current point is higher than its neighbors
				if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
					# Check if it meets the height criterion
					if signal[i] >= height:
						# Check if it's far enough from the previously detected peak
						if not peaks or i - peaks[-1] >= distance:
							peaks.append(i)
			return np.array(peaks)

   }]; 

   let arguments = (ins F64Tensor:$signal, F64Tensor:$height, F64Tensor:$distance);
   let results = (outs F64Tensor);

   // Indicate that the operation has a custom parser and printer method.
   // let hasCustomAssemblyFormat = 1;
   // let assemblyFormat = [{
   //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
   // }];
   // Allow building a MulOp with from the two input operands.
   
   let builders = [
     OpBuilder<(ins "Value":$signal, "Value":$height, "Value":$distance)>
   ];

   // Indicate that the operation has a custom parser and printer method.
   // let hasCustomAssemblyFormat = 1;

   // let hasVerifier = 1;
   let hasCanonicalizer = 1;

 }


//===----------------------------------------------------------------------===//
// BeamFormOp
//===----------------------------------------------------------------------===//

def BeamFormOp : Dsp_Op<"beam_form", 
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "Dsp dialect Beam forming operation";
    let description = [{
        Performs a beam forming signal encoding on the input tensor using specified weights.
    }];

    let arguments = (ins I64Attr:$antennas, I64Attr:$freq, F64Tensor:$time, F64Tensor:$weights);

    let results = (outs F64Tensor:$output);

    let builders = [
        OpBuilder<(ins "int64_t":$antennas, "int64_t":$freq, "Value":$time, "Value":$weights)>
    ];

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SpaceModulateOp
//===----------------------------------------------------------------------===//

def SpaceModulateOp : Dsp_Op<"space_modulate", 
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "Dsp dialect space modulation operation";
    let description = [{
        Takes in string input and convert it to binary.
    }];

    let arguments = (ins F64Tensor:$signal);

    let results = (outs F64Tensor:$output);

    let builders = [
        OpBuilder<(ins "Value":$signal)>
    ];

    let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// SpaceDemodulateOp
//===----------------------------------------------------------------------===//

def SpaceDemodulateOp : Dsp_Op<"space_demodulate", 
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "Dsp dialect space demodulation operation";
    let description = [{
        Takes in binary input and convert it to string.
    }];

    let arguments = (ins F64Tensor:$binary);

    let results = (outs F64Tensor:$output);

    let builders = [
        OpBuilder<(ins "Value":$binary)>
    ];

    let hasVerifier = 1;
    let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// SpaceErrCorrectionOp
//===----------------------------------------------------------------------===//

def SpaceErrCorrectionOp : Dsp_Op<"space_err_correction", 
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
    let summary = "Dsp dialect space error correction operation";
    let description = [{
        Remove noise operation for signal transmission in space.
    }];

    let arguments = (ins F64Tensor:$signal);

    let results = (outs F64Tensor:$output);

    let builders = [
        OpBuilder<(ins "Value":$signal)>
    ];

    let hasVerifier = 1;
}


//===----------------------------------------------------------------------===//
// MaxOp
//===----------------------------------------------------------------------===//

def MaxOp : Dsp_Op<"max", [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
   let summary = "Find maximum value in tensor";
   let description = [{
		This operation finds and returns the maximum value of the tensor.
   }]; 

   let arguments = (ins F64Tensor:$input);
   let results = (outs F64Tensor);

   // Indicate that the operation has a custom parser and printer method.
   // let hasCustomAssemblyFormat = 1;
   // let assemblyFormat = [{
   //   `(` $input `:` type($input1 , $input2) `)` attr-dict `to` type(results)
   // }];
   // Allow building a MulOp with from the two input operands.
   
   let builders = [
     OpBuilder<(ins "Value":$input)>
   ];

   // Indicate that the operation has a custom parser and printer method.
   // let hasCustomAssemblyFormat = 1;

   // let hasVerifier = 1;
 }
 

//===----------------------------------------------------------------------===//
// MeanOp
//===----------------------------------------------------------------------===//

def MeanOp : Dsp_Op<"mean", [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
   let summary = "Find mean value of tensor. Requires (input, length).";
   let description = [{
		This operation finds and returns the mean value of the tensor.
        Note that it requires length.
		It would be better if we can implement both versions 
		- no length argument -> automatically use the length of tensor
		- with length argument -> use the provided length
   }]; 

   let arguments = (ins F64Tensor:$input, F64Tensor:$length);
   let results = (outs F64Tensor);
   
   let builders = [
     OpBuilder<(ins "Value":$input, "Value":$length)>
   ];

   // Indicate that the operation has a custom parser and printer method.
   // let hasCustomAssemblyFormat = 1;

   // let hasVerifier = 1;
   let hasCanonicalizer = 1;
 }
 


//===----------------------------------------------------------------------===//
// DiffOp
//===----------------------------------------------------------------------===//

def DiffOp : Dsp_Op<"diff", [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
   let summary = "np.diff (out[i] = a[i+1] - a[i]). It receives second argument as length";
   let description = [{
		This operation returns a tensor that contains diff (out[i] = a[i+1] - a[i]).
		The length of the output tensor is len(input)-1, regardless of length parameter.
		Note that it requires length.
		It would be better if we can implement both versions 
		- no length argument -> automatically use the length of tensor
		- with length argument -> use the provided length
   }]; 

   let arguments = (ins F64Tensor:$input, F64Tensor:$length);
   let results = (outs F64Tensor);
   
   let builders = [
     OpBuilder<(ins "Value":$input, "Value":$length)>
   ];

   // Indicate that the operation has a custom parser and printer method.
   // let hasCustomAssemblyFormat = 1;

   // let hasVerifier = 1;
 }
 
//===----------------------------------------------------------------------===//
// AbsOp
//===----------------------------------------------------------------------===//

def AbsOp : Dsp_Op<"abs", [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
   let summary = "np.abs -> calculate the absolute value element-wise";
   let description = [{
       This operation calculates the absolute value element-wise.
   }]; 

   let arguments = (ins F64Tensor:$input);
   let results = (outs F64Tensor:$output);
   
   let builders = [
     OpBuilder<(ins "Value":$input)>
   ];
 }


//===----------------------------------------------------------------------===//
// ArgMaxOp
//===----------------------------------------------------------------------===//

def ArgMaxOp : Dsp_Op<"argmax", [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
   let summary = "np.argmax -> find the indices of the maximum values along a specifies axis in an array.";
   let description = [{
       This operation find the indices of the maximum values along a specifies axis in an array.
   }]; 

   let arguments = (ins F64Tensor:$input, I64Attr:$axis);
   let results = (outs F64Tensor:$output);
   
   let builders = [
     OpBuilder<(ins "Value":$input, "int64_t":$axis)>
   ];
 }
//===----------------------------------------------------------------------===//
// NormalizeOp
//===----------------------------------------------------------------------===//

def NormalizeOp : Dsp_Op<"normalize", [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
   let summary = "normalize operation.";
   let description = [{
       normalization dsp operation.
   }]; 

   let arguments = (ins F64Tensor:$signal);
   let results = (outs F64Tensor);
   
   let builders = [
     OpBuilder<(ins "Value":$signal)>
   ];

    let hasCanonicalizer = 1;
 }

//===----------------------------------------------------------------------===//
// NormLMSFilterResponseOptimizeOp
//===----------------------------------------------------------------------===//


def NormLMSFilterResponseOptimizeOp : Dsp_Op<"norm_LMSFilterResponse_opt",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "LMS filter Response + norm optimize";
  let description = [{
      norm + lmsfilter
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs, F64Tensor:$mu, F64Tensor:$filterLen);

   let results = (outs F64Tensor);

   let builders = [
     OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$mu, "Value":$filterLen)>
   ];

   let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Median2SlidingOptimizedOp
//===----------------------------------------------------------------------===//

def Median2SlidingOptimizedOp : Dsp_Op<"median2slidingOp", 
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  
  let arguments = (ins F64Tensor:$input);
  let results = (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "Value":$input)>
  ];
}


//===----------------------------------------------------------------------===//
// FIRFilterResSymmThresholdUpOptimizedOp
//===----------------------------------------------------------------------===//
def FIRFilterResSymmThresholdUpOptimizedOp : Dsp_Op<"FIRFilterResSymmThresholdUpOptimizedOp" ,
    [Pure , DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "FIRFilterResSymmThresholdUpOptimizedOp";
  let description = [{
    The "FIRFilterResSymmThresholdUpOptimizedOp" operation is basically the convolution of input 1-D and filter vector when the filter is symmetrical ie,
    h[0] = h[L-1] , h[1] = h[L-2] .. h[middle ie, (L-1)/2] is single element & filter length is odd always
    ex: x[n] = [2,1,3,2,4], h[l] = [1, -1 ,1]  then
      y[n] = sum(h(k) . x(n-k)) k=0 to N-1 can be rewritten as 
      y[n] = sum(h[k] .{ x[n-k] + x[n-(L-1-k)]}) + h[L-1/2].x[n-(L-1)/2] , k=0 to L-1/2

      Basically, we are trying to reduce the number of load/store operations by half -- so that we can reduce the operations
  }];

  let arguments = (ins F64Tensor:$lhs, F64Tensor:$rhs, F64Tensor:$threshold, F64Tensor:$returnoriginal); 
  let results =  (outs F64Tensor);

  let builders = [
    OpBuilder<(ins "Value":$lhs, "Value":$rhs, "Value":$threshold, "Value":$returnoriginal)>
  ];

}

#endif // TOY_OPS



