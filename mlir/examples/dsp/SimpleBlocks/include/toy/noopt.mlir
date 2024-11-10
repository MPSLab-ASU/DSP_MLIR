module {
  func.func @main() {
    %c2047_i64 = arith.constant 2047 : i64
    %cst = arith.constant 0.49971199035644531 : f64
    %c11 = arith.constant 11 : index
    %c10 = arith.constant 10 : index
    %cst_0 = arith.constant 3.000000e+00 : f64
    %c-1 = arith.constant -1 : index
    %cst_1 = arith.constant 7.700000e+02 : f64
    %cst_2 = arith.constant 1.209000e+03 : f64
    %cst_3 = arith.constant 6.970000e+02 : f64
    %cst_4 = arith.constant 1.336000e+03 : f64
    %cst_5 = arith.constant 9.410000e+02 : f64
    %cst_6 = arith.constant 1.220000e-04 : f64
    %cst_7 = arith.constant 4.096000e+03 : f64
    %cst_8 = arith.constant -2.000000e+00 : f64
    %cst_9 = arith.constant 3.1415926535897931 : f64
    %c12 = arith.constant 12 : index
    %cst_10 = arith.constant 0.000000e+00 : f64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c4096 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    %cst_11 = arith.constant 1.477000e+03 : f64
    %cst_12 = arith.constant 8.520000e+02 : f64
    %cst_13 = arith.constant 1.000000e+01 : f64
    %cst_14 = arith.constant 6.2831853071800001 : f64
    %cst_15 = arith.constant 8.192000e+03 : f64
    %cst_16 = arith.constant 5.000000e-01 : f64
    %cst_17 = arith.constant 9.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<index>
    %alloc_18 = memref.alloc() : memref<1xf64>
    %alloc_19 = memref.alloc() : memref<10x2xf64>
    %alloc_20 = memref.alloc() : memref<2xf64>
    %alloc_21 = memref.alloc() : memref<4096xf64>
    %alloc_22 = memref.alloc() : memref<f64>
    %alloc_23 = memref.alloc() : memref<f64>
    %alloc_24 = memref.alloc() : memref<4096xf64>
    %alloc_25 = memref.alloc() : memref<4096xf64>
    %alloc_26 = memref.alloc() : memref<4096xf64>
    %alloc_27 = memref.alloc() : memref<4096xf64>
    %alloc_28 = memref.alloc() : memref<f64>
    %alloc_29 = memref.alloc() : memref<f64>
    %alloc_30 = memref.alloc() : memref<f64>
    affine.store %cst_17, %alloc_30[] : memref<f64>
    affine.store %cst_16, %alloc_29[] : memref<f64>
    affine.store %cst_15, %alloc_28[] : memref<f64>
    scf.for %arg0 = %c0 to %c4096 step %c1 {
      %6 = arith.index_cast %arg0 : index to i64
      %7 = arith.sitofp %6 : i64 to f64
      %8 = arith.divf %7, %cst_15 : f64
      %9 = arith.mulf %8, %cst_12 : f64
      %10 = arith.mulf %9, %cst_14 : f64
      %11 = math.sin %10 : f64
      %12 = arith.mulf %8, %cst_11 : f64
      %13 = arith.mulf %12, %cst_14 : f64
      %14 = math.sin %13 : f64
      %15 = arith.addf %11, %14 : f64
      %16 = arith.mulf %15, %cst_13 : f64
      memref.store %16, %alloc_27[%arg0] : memref<4096xf64>
    }
    scf.for %arg0 = %c0 to %c4096 step %c1 {
      %6 = arith.index_cast %arg0 : index to i64
      %7 = scf.for %arg1 = %c0 to %c12 step %c1 iter_args(%arg2 = %c0_i64) -> (i64) {
        %10 = arith.index_cast %arg1 : index to i64
        %11 = arith.shli %c1_i64, %10 : i64
        %12 = arith.andi %6, %11 : i64
        %13 = arith.cmpi ne, %12, %c0_i64 : i64
        %14 = arith.subi %c11, %arg1 : index
        %15 = arith.index_cast %14 : index to i64
        %16 = arith.shli %c1_i64, %15 : i64
        %17 = arith.select %13, %16, %c0_i64 : i64
        %18 = arith.ori %arg2, %17 : i64
        scf.yield %18 : i64
      }
      %8 = arith.index_cast %7 : i64 to index
      %9 = memref.load %alloc_27[%arg0] : memref<4096xf64>
      memref.store %9, %alloc_26[%8] : memref<4096xf64>
      memref.store %cst_10, %alloc_25[%8] : memref<4096xf64>
    }
    scf.for %arg0 = %c0 to %c12 step %c1 {
      %6 = arith.shli %c1, %arg0 : index
      %7 = arith.shli %6, %c1 : index
      scf.for %arg1 = %c0 to %c4096 step %7 {
        scf.for %arg2 = %c0 to %6 step %c1 {
          %8 = arith.addi %arg1, %arg2 : index
          %9 = arith.addi %8, %6 : index
          %10 = arith.index_cast %arg2 : index to i64
          %11 = arith.sitofp %10 : i64 to f64
          %12 = arith.index_cast %7 : index to i64
          %13 = arith.sitofp %12 : i64 to f64
          %14 = arith.divf %11, %13 : f64
          %15 = arith.mulf %14, %cst_8 : f64
          %16 = arith.mulf %15, %cst_9 : f64
          %17 = math.cos %16 : f64
          %18 = math.sin %16 : f64
          %19 = memref.load %alloc_26[%9] : memref<4096xf64>
          %20 = memref.load %alloc_25[%9] : memref<4096xf64>
          %21 = arith.mulf %19, %17 : f64
          %22 = arith.mulf %20, %18 : f64
          %23 = arith.subf %21, %22 : f64
          %24 = arith.mulf %19, %18 : f64
          %25 = arith.mulf %20, %17 : f64
          %26 = arith.addf %24, %25 : f64
          %27 = memref.load %alloc_26[%8] : memref<4096xf64>
          %28 = memref.load %alloc_25[%8] : memref<4096xf64>
          %29 = arith.addf %27, %23 : f64
          %30 = arith.addf %28, %26 : f64
          %31 = arith.subf %27, %23 : f64
          %32 = arith.subf %28, %26 : f64
          %33 = arith.mulf %29, %29 : f64
          %34 = arith.mulf %30, %30 : f64
          %35 = arith.addf %33, %34 : f64
          %36 = math.sqrt %35 : f64
          %37 = arith.mulf %31, %31 : f64
          %38 = arith.mulf %32, %32 : f64
          %39 = arith.addf %37, %38 : f64
          %40 = math.sqrt %39 : f64
          memref.store %29, %alloc_26[%8] : memref<4096xf64>
          memref.store %30, %alloc_25[%8] : memref<4096xf64>
          memref.store %31, %alloc_26[%9] : memref<4096xf64>
          memref.store %32, %alloc_25[%9] : memref<4096xf64>
          memref.store %36, %alloc_24[%8] : memref<4096xf64>
          memref.store %40, %alloc_24[%9] : memref<4096xf64>
        }
      }
    }
    affine.store %cst_7, %alloc_23[] : memref<f64>
    affine.store %cst_6, %alloc_22[] : memref<f64>
    scf.for %arg0 = %c0 to %c4096 step %c1 {
      %6 = arith.index_cast %arg0 : index to i64
      %7 = arith.sitofp %6 : i64 to f64
      %8 = arith.cmpi sle, %6, %c2047_i64 : i64
      scf.if %8 {
        %9 = arith.divf %7, %cst : f64
        memref.store %9, %alloc_21[%arg0] : memref<4096xf64>
      } else {
        %9 = arith.subf %7, %cst_7 : f64
        %10 = arith.divf %9, %cst : f64
        memref.store %10, %alloc_21[%arg0] : memref<4096xf64>
      }
    }
    %0:4 = scf.for %arg0 = %c0 to %c4096 step %c1 iter_args(%arg1 = %cst_10, %arg2 = %cst_10, %arg3 = %cst_10, %arg4 = %cst_10) -> (f64, f64, f64, f64) {
      %6 = memref.load %alloc_21[%arg0] : memref<4096xf64>
      %7 = memref.load %alloc_24[%arg0] : memref<4096xf64>
      %8 = arith.cmpf ogt, %6, %cst_10 : f64
      %9:4 = scf.if %8 -> (f64, f64, f64, f64) {
        %10 = arith.cmpf ogt, %7, %arg1 : f64
        %11 = arith.select %10, %7, %arg1 : f64
        %12 = arith.select %10, %6, %arg3 : f64
        %13:2 = scf.if %10 -> (f64, f64) {
          scf.yield %arg1, %arg3 : f64, f64
        } else {
          %14 = arith.cmpf ogt, %7, %arg2 : f64
          %15 = arith.select %14, %7, %arg2 : f64
          %16 = arith.select %14, %6, %arg4 : f64
          scf.yield %15, %16 : f64, f64
        }
        scf.yield %11, %13#0, %12, %13#1 : f64, f64, f64, f64
      } else {
        scf.yield %arg1, %arg2, %arg3, %arg4 : f64, f64, f64, f64
      }
      scf.yield %9#0, %9#1, %9#2, %9#3 : f64, f64, f64, f64
    }
    memref.store %0#2, %alloc_20[%c0] : memref<2xf64>
    memref.store %0#3, %alloc_20[%c1] : memref<2xf64>
    affine.store %cst_5, %alloc_19[0, 0] : memref<10x2xf64>
    affine.store %cst_4, %alloc_19[0, 1] : memref<10x2xf64>
    affine.store %cst_3, %alloc_19[1, 0] : memref<10x2xf64>
    affine.store %cst_2, %alloc_19[1, 1] : memref<10x2xf64>
    affine.store %cst_3, %alloc_19[2, 0] : memref<10x2xf64>
    affine.store %cst_4, %alloc_19[2, 1] : memref<10x2xf64>
    affine.store %cst_3, %alloc_19[3, 0] : memref<10x2xf64>
    affine.store %cst_11, %alloc_19[3, 1] : memref<10x2xf64>
    affine.store %cst_1, %alloc_19[4, 0] : memref<10x2xf64>
    affine.store %cst_2, %alloc_19[4, 1] : memref<10x2xf64>
    affine.store %cst_1, %alloc_19[5, 0] : memref<10x2xf64>
    affine.store %cst_4, %alloc_19[5, 1] : memref<10x2xf64>
    affine.store %cst_1, %alloc_19[6, 0] : memref<10x2xf64>
    affine.store %cst_11, %alloc_19[6, 1] : memref<10x2xf64>
    affine.store %cst_12, %alloc_19[7, 0] : memref<10x2xf64>
    affine.store %cst_2, %alloc_19[7, 1] : memref<10x2xf64>
    affine.store %cst_12, %alloc_19[8, 0] : memref<10x2xf64>
    affine.store %cst_4, %alloc_19[8, 1] : memref<10x2xf64>
    affine.store %cst_12, %alloc_19[9, 0] : memref<10x2xf64>
    affine.store %cst_11, %alloc_19[9, 1] : memref<10x2xf64>
    %1 = memref.load %alloc_20[%c0] : memref<2xf64>
    %2 = memref.load %alloc_20[%c1] : memref<2xf64>
    affine.store %c-1, %alloc[] : memref<index>
    scf.for %arg0 = %c0 to %c10 step %c1 {
      %6 = memref.load %alloc[] : memref<index>
      %7 = memref.load %alloc_19[%arg0, %c0] : memref<10x2xf64>
      %8 = memref.load %alloc_19[%arg0, %c1] : memref<10x2xf64>
      %9 = arith.subf %7, %1 : f64
      %10 = arith.subf %8, %2 : f64
      %11 = math.absf %9 : f64
      %12 = math.absf %10 : f64
      %13 = arith.cmpf ole, %11, %cst_0 : f64
      %14 = arith.cmpf ole, %12, %cst_0 : f64
      %15 = arith.andi %13, %14 : i1
      %16 = arith.select %15, %arg0, %6 : index
      memref.store %16, %alloc[] : memref<index>
    }
    %3 = memref.load %alloc[] : memref<index>
    %4 = arith.index_cast %3 : index to i64
    %5 = arith.sitofp %4 : i64 to f64
    memref.store %5, %alloc_18[%c0] : memref<1xf64>
root@f68572e75858:/home/DSP_MLIR# /home/DSP_MLIR/build/bin/dsp1 /home/DSP_MLIR/mlir/test/Examples/DspExample/full_dtmf.py -emit=mlir-affine -affineOpt -canonOpt -opt
module {
  func.func @main() {
    %c2047_i64 = arith.constant 2047 : i64
    %cst = arith.constant 0.49971199035644531 : f64
    %c11 = arith.constant 11 : index
    %c10 = arith.constant 10 : index
    %cst_0 = arith.constant 3.000000e+00 : f64
    %c-1 = arith.constant -1 : index
    %cst_1 = arith.constant 7.700000e+02 : f64
    %cst_2 = arith.constant 1.209000e+03 : f64
    %cst_3 = arith.constant 6.970000e+02 : f64
    %cst_4 = arith.constant 1.336000e+03 : f64
    %cst_5 = arith.constant 9.410000e+02 : f64
    %cst_6 = arith.constant 1.220000e-04 : f64
    %cst_7 = arith.constant 4.096000e+03 : f64
    %cst_8 = arith.constant -2.000000e+00 : f64
    %cst_9 = arith.constant 3.1415926535897931 : f64
    %c12 = arith.constant 12 : index
    %cst_10 = arith.constant 0.000000e+00 : f64
    %c1_i64 = arith.constant 1 : i64
    %c0_i64 = arith.constant 0 : i64
    %c4096 = arith.constant 4096 : index
    %c1 = arith.constant 1 : index
    %cst_11 = arith.constant 1.477000e+03 : f64
    %cst_12 = arith.constant 8.520000e+02 : f64
    %cst_13 = arith.constant 1.000000e+01 : f64
    %cst_14 = arith.constant 6.2831853071800001 : f64
    %cst_15 = arith.constant 8.192000e+03 : f64
    %cst_16 = arith.constant 5.000000e-01 : f64
    %cst_17 = arith.constant 9.000000e+00 : f64
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<index>
    %alloc_18 = memref.alloc() : memref<1xf64>
    %alloc_19 = memref.alloc() : memref<10x2xf64>
    %alloc_20 = memref.alloc() : memref<2xf64>
    %alloc_21 = memref.alloc() : memref<4096xf64>
    %alloc_22 = memref.alloc() : memref<f64>
    %alloc_23 = memref.alloc() : memref<f64>
    %alloc_24 = memref.alloc() : memref<4096xf64>
    %alloc_25 = memref.alloc() : memref<4096xf64>
    %alloc_26 = memref.alloc() : memref<4096xf64>
    %alloc_27 = memref.alloc() : memref<4096xf64>
    %alloc_28 = memref.alloc() : memref<f64>
    %alloc_29 = memref.alloc() : memref<f64>
    %alloc_30 = memref.alloc() : memref<f64>
    affine.store %cst_17, %alloc_30[] : memref<f64>
    affine.store %cst_16, %alloc_29[] : memref<f64>
    affine.store %cst_15, %alloc_28[] : memref<f64>
    scf.for %arg0 = %c0 to %c4096 step %c1 {
      %6 = arith.index_cast %arg0 : index to i64
      %7 = arith.sitofp %6 : i64 to f64
      %8 = arith.divf %7, %cst_15 : f64
      %9 = arith.mulf %8, %cst_12 : f64
      %10 = arith.mulf %9, %cst_14 : f64
      %11 = math.sin %10 : f64
      %12 = arith.mulf %8, %cst_11 : f64
      %13 = arith.mulf %12, %cst_14 : f64
      %14 = math.sin %13 : f64
      %15 = arith.addf %11, %14 : f64
      %16 = arith.mulf %15, %cst_13 : f64
      memref.store %16, %alloc_27[%arg0] : memref<4096xf64>
    }
    scf.for %arg0 = %c0 to %c4096 step %c1 {
      %6 = arith.index_cast %arg0 : index to i64
      %7 = scf.for %arg1 = %c0 to %c12 step %c1 iter_args(%arg2 = %c0_i64) -> (i64) {
        %10 = arith.index_cast %arg1 : index to i64
        %11 = arith.shli %c1_i64, %10 : i64
        %12 = arith.andi %6, %11 : i64
        %13 = arith.cmpi ne, %12, %c0_i64 : i64
        %14 = arith.subi %c11, %arg1 : index
        %15 = arith.index_cast %14 : index to i64
        %16 = arith.shli %c1_i64, %15 : i64
        %17 = arith.select %13, %16, %c0_i64 : i64
        %18 = arith.ori %arg2, %17 : i64
        scf.yield %18 : i64
      }
      %8 = arith.index_cast %7 : i64 to index
      %9 = memref.load %alloc_27[%arg0] : memref<4096xf64>
      memref.store %9, %alloc_26[%8] : memref<4096xf64>
      memref.store %cst_10, %alloc_25[%8] : memref<4096xf64>
    }
    scf.for %arg0 = %c0 to %c12 step %c1 {
      %6 = arith.shli %c1, %arg0 : index
      %7 = arith.shli %6, %c1 : index
      scf.for %arg1 = %c0 to %c4096 step %7 {
        scf.for %arg2 = %c0 to %6 step %c1 {
          %8 = arith.addi %arg1, %arg2 : index
          %9 = arith.addi %8, %6 : index
          %10 = arith.index_cast %arg2 : index to i64
          %11 = arith.sitofp %10 : i64 to f64
          %12 = arith.index_cast %7 : index to i64
          %13 = arith.sitofp %12 : i64 to f64
          %14 = arith.divf %11, %13 : f64
          %15 = arith.mulf %14, %cst_8 : f64
          %16 = arith.mulf %15, %cst_9 : f64
          %17 = math.cos %16 : f64
          %18 = math.sin %16 : f64
          %19 = memref.load %alloc_26[%9] : memref<4096xf64>
          %20 = memref.load %alloc_25[%9] : memref<4096xf64>
          %21 = arith.mulf %19, %17 : f64
          %22 = arith.mulf %20, %18 : f64
          %23 = arith.subf %21, %22 : f64
          %24 = arith.mulf %19, %18 : f64
          %25 = arith.mulf %20, %17 : f64
          %26 = arith.addf %24, %25 : f64
          %27 = memref.load %alloc_26[%8] : memref<4096xf64>
          %28 = memref.load %alloc_25[%8] : memref<4096xf64>
          %29 = arith.addf %27, %23 : f64
          %30 = arith.addf %28, %26 : f64
          %31 = arith.subf %27, %23 : f64
          %32 = arith.subf %28, %26 : f64
          %33 = arith.mulf %29, %29 : f64
          %34 = arith.mulf %30, %30 : f64
          %35 = arith.addf %33, %34 : f64
          %36 = math.sqrt %35 : f64
          %37 = arith.mulf %31, %31 : f64
          %38 = arith.mulf %32, %32 : f64
          %39 = arith.addf %37, %38 : f64
          %40 = math.sqrt %39 : f64
          memref.store %29, %alloc_26[%8] : memref<4096xf64>
          memref.store %30, %alloc_25[%8] : memref<4096xf64>
          memref.store %31, %alloc_26[%9] : memref<4096xf64>
          memref.store %32, %alloc_25[%9] : memref<4096xf64>
          memref.store %36, %alloc_24[%8] : memref<4096xf64>
          memref.store %40, %alloc_24[%9] : memref<4096xf64>
        }
      }
    }
    affine.store %cst_7, %alloc_23[] : memref<f64>
    affine.store %cst_6, %alloc_22[] : memref<f64>
    scf.for %arg0 = %c0 to %c4096 step %c1 {
      %6 = arith.index_cast %arg0 : index to i64
      %7 = arith.sitofp %6 : i64 to f64
      %8 = arith.cmpi sle, %6, %c2047_i64 : i64
      scf.if %8 {
        %9 = arith.divf %7, %cst : f64
        memref.store %9, %alloc_21[%arg0] : memref<4096xf64>
      } else {
        %9 = arith.subf %7, %cst_7 : f64
        %10 = arith.divf %9, %cst : f64
        memref.store %10, %alloc_21[%arg0] : memref<4096xf64>
      }
    }
    %0:4 = scf.for %arg0 = %c0 to %c4096 step %c1 iter_args(%arg1 = %cst_10, %arg2 = %cst_10, %arg3 = %cst_10, %arg4 = %cst_10) -> (f64, f64, f64, f64) {
      %6 = memref.load %alloc_21[%arg0] : memref<4096xf64>
      %7 = memref.load %alloc_24[%arg0] : memref<4096xf64>
      %8 = arith.cmpf ogt, %6, %cst_10 : f64
      %9:4 = scf.if %8 -> (f64, f64, f64, f64) {
        %10 = arith.cmpf ogt, %7, %arg1 : f64
        %11 = arith.select %10, %7, %arg1 : f64
        %12 = arith.select %10, %6, %arg3 : f64
        %13:2 = scf.if %10 -> (f64, f64) {
          scf.yield %arg1, %arg3 : f64, f64
        } else {
          %14 = arith.cmpf ogt, %7, %arg2 : f64
          %15 = arith.select %14, %7, %arg2 : f64
          %16 = arith.select %14, %6, %arg4 : f64
          scf.yield %15, %16 : f64, f64
        }
        scf.yield %11, %13#0, %12, %13#1 : f64, f64, f64, f64
      } else {
        scf.yield %arg1, %arg2, %arg3, %arg4 : f64, f64, f64, f64
      }
      scf.yield %9#0, %9#1, %9#2, %9#3 : f64, f64, f64, f64
    }
    memref.store %0#2, %alloc_20[%c0] : memref<2xf64>
    memref.store %0#3, %alloc_20[%c1] : memref<2xf64>
    affine.store %cst_5, %alloc_19[0, 0] : memref<10x2xf64>
    affine.store %cst_4, %alloc_19[0, 1] : memref<10x2xf64>
    affine.store %cst_3, %alloc_19[1, 0] : memref<10x2xf64>
    affine.store %cst_2, %alloc_19[1, 1] : memref<10x2xf64>
    affine.store %cst_3, %alloc_19[2, 0] : memref<10x2xf64>
    affine.store %cst_4, %alloc_19[2, 1] : memref<10x2xf64>
    affine.store %cst_3, %alloc_19[3, 0] : memref<10x2xf64>
    affine.store %cst_11, %alloc_19[3, 1] : memref<10x2xf64>
    affine.store %cst_1, %alloc_19[4, 0] : memref<10x2xf64>
    affine.store %cst_2, %alloc_19[4, 1] : memref<10x2xf64>
    affine.store %cst_1, %alloc_19[5, 0] : memref<10x2xf64>
    affine.store %cst_4, %alloc_19[5, 1] : memref<10x2xf64>
    affine.store %cst_1, %alloc_19[6, 0] : memref<10x2xf64>
    affine.store %cst_11, %alloc_19[6, 1] : memref<10x2xf64>
    affine.store %cst_12, %alloc_19[7, 0] : memref<10x2xf64>
    affine.store %cst_2, %alloc_19[7, 1] : memref<10x2xf64>
    affine.store %cst_12, %alloc_19[8, 0] : memref<10x2xf64>
    affine.store %cst_4, %alloc_19[8, 1] : memref<10x2xf64>
    affine.store %cst_12, %alloc_19[9, 0] : memref<10x2xf64>
    affine.store %cst_11, %alloc_19[9, 1] : memref<10x2xf64>
    %1 = memref.load %alloc_20[%c0] : memref<2xf64>
    %2 = memref.load %alloc_20[%c1] : memref<2xf64>
    affine.store %c-1, %alloc[] : memref<index>
    scf.for %arg0 = %c0 to %c10 step %c1 {
      %6 = memref.load %alloc[] : memref<index>
      %7 = memref.load %alloc_19[%arg0, %c0] : memref<10x2xf64>
      %8 = memref.load %alloc_19[%arg0, %c1] : memref<10x2xf64>
      %9 = arith.subf %7, %1 : f64
      %10 = arith.subf %8, %2 : f64
      %11 = math.absf %9 : f64
      %12 = math.absf %10 : f64
      %13 = arith.cmpf ole, %11, %cst_0 : f64
      %14 = arith.cmpf ole, %12, %cst_0 : f64
      %15 = arith.andi %13, %14 : i1
      %16 = arith.select %15, %arg0, %6 : index
      memref.store %16, %alloc[] : memref<index>
    }
    %3 = memref.load %alloc[] : memref<index>
    %4 = arith.index_cast %3 : index to i64
    %5 = arith.sitofp %4 : i64 to f64
    memref.store %5, %alloc_18[%c0] : memref<1xf64>
root@f68572e75858:/home/DSP_MLIR# /home/DSP_MLIR/build/bin/dsp1 /home/DSP_MLIR/mlir/test/Examples/DspExample/full_dtmf.py -emit=mlir-affine 
module {
  func.func @main() {
    %alloc = memref.alloc() : memref<index>
    %alloc_0 = memref.alloc() : memref<1xf64>
    %alloc_1 = memref.alloc() : memref<10x2xf64>
    %alloc_2 = memref.alloc() : memref<2xf64>
    %alloc_3 = memref.alloc() : memref<4096xf64>
    %alloc_4 = memref.alloc() : memref<f64>
    %alloc_5 = memref.alloc() : memref<f64>
    %alloc_6 = memref.alloc() : memref<4096xf64>
    %alloc_7 = memref.alloc() : memref<4096xf64>
    %alloc_8 = memref.alloc() : memref<4096xf64>
    %alloc_9 = memref.alloc() : memref<4096xf64>
    %alloc_10 = memref.alloc() : memref<4096xf64>
    %alloc_11 = memref.alloc() : memref<4096xf64>
    %alloc_12 = memref.alloc() : memref<4096xf64>
    %alloc_13 = memref.alloc() : memref<4096xf64>
    %alloc_14 = memref.alloc() : memref<4096xf64>
    %alloc_15 = memref.alloc() : memref<4096xf64>
    %alloc_16 = memref.alloc() : memref<4096xf64>
    %alloc_17 = memref.alloc() : memref<4096xf64>
    %alloc_18 = memref.alloc() : memref<4096xf64>
    %alloc_19 = memref.alloc() : memref<f64>
    %alloc_20 = memref.alloc() : memref<f64>
    %alloc_21 = memref.alloc() : memref<f64>
    %c0 = arith.constant 0 : index
    %cst = arith.constant 9.000000e+00 : f64
    affine.store %cst, %alloc_21[] : memref<f64>
    %c0_22 = arith.constant 0 : index
    %cst_23 = arith.constant 5.000000e-01 : f64
    affine.store %cst_23, %alloc_20[] : memref<f64>
    %c0_24 = arith.constant 0 : index
    %cst_25 = arith.constant 8.192000e+03 : f64
    affine.store %cst_25, %alloc_19[] : memref<f64>
    %cst_26 = arith.constant 6.2831853071800001 : f64
    %cst_27 = arith.constant 1.000000e+01 : f64
    %cst_28 = arith.constant 8.192000e+03 : f64
    %cst_29 = arith.constant 8.520000e+02 : f64
    %cst_30 = arith.constant 1.477000e+03 : f64
    %c1 = arith.constant 1 : index
    %c4096 = arith.constant 4096 : index
    %c0_31 = arith.constant 0 : index
    scf.for %arg0 = %c0_31 to %c4096 step %c1 {
      %19 = arith.index_cast %arg0 : index to i64
      %20 = arith.sitofp %19 : i64 to f64
      %21 = arith.divf %20, %cst_28 : f64
      %22 = arith.mulf %cst_29, %21 : f64
      %23 = arith.mulf %cst_26, %22 : f64
      %24 = math.sin %23 : f64
      %25 = arith.mulf %cst_30, %21 : f64
      %26 = arith.mulf %cst_26, %25 : f64
      %27 = math.sin %26 : f64
      %28 = arith.addf %24, %27 : f64
      %29 = arith.mulf %cst_27, %28 : f64
      memref.store %29, %alloc_18[%arg0] : memref<4096xf64>
    }
    %c0_32 = arith.constant 0 : index
    %c4096_33 = arith.constant 4096 : index
    %c1_34 = arith.constant 1 : index
    %0 = arith.index_cast %c4096_33 : index to i64
    %1 = arith.sitofp %0 : i64 to f64
    %2 = math.log2 %1 : f64
    %3 = arith.fptosi %2 : f64 to i64
    %4 = arith.index_cast %3 : i64 to index
    scf.for %arg0 = %c0_32 to %c4096_33 step %c1_34 {
      %19 = arith.index_cast %arg0 : index to i64
      %c0_i64 = arith.constant 0 : i64
      %20 = scf.for %arg1 = %c0_32 to %4 step %c1_34 iter_args(%arg2 = %c0_i64) -> (i64) {
        %23 = arith.index_cast %arg1 : index to i64
        %c1_i64 = arith.constant 1 : i64
        %24 = arith.shli %c1_i64, %23 : i64
        %25 = arith.andi %19, %24 : i64
        %c0_i64_92 = arith.constant 0 : i64
        %26 = arith.cmpi ne, %25, %c0_i64_92 : i64
        %c1_93 = arith.constant 1 : index
        %27 = arith.subi %4, %arg1 : index
        %28 = arith.subi %27, %c1_93 : index
        %29 = arith.index_cast %28 : index to i64
        %c1_i64_94 = arith.constant 1 : i64
        %30 = arith.shli %c1_i64_94, %29 : i64
        %c0_i64_95 = arith.constant 0 : i64
        %31 = arith.select %26, %30, %c0_i64_95 : i64
        %32 = arith.ori %arg2, %31 : i64
        scf.yield %32 : i64
      }
      %21 = arith.index_cast %20 : i64 to index
      %22 = memref.load %alloc_18[%arg0] : memref<4096xf64>
      %cst_91 = arith.constant 0.000000e+00 : f64
      memref.store %22, %alloc_15[%21] : memref<4096xf64>
      memref.store %cst_91, %alloc_14[%21] : memref<4096xf64>
    }
    %c12 = arith.constant 12 : index
    %cst_35 = arith.constant 3.1415926535897931 : f64
    %cst_36 = arith.constant -2.000000e+00 : f64
    scf.for %arg0 = %c0_32 to %c12 step %c1_34 {
      %c1_91 = arith.constant 1 : index
      %19 = arith.shli %c1_91, %arg0 : index
      %c1_92 = arith.constant 1 : index
      %20 = arith.shli %19, %c1_92 : index
      scf.for %arg1 = %c0_32 to %c4096_33 step %20 {
        scf.for %arg2 = %c0_32 to %19 step %c1_34 {
          %21 = arith.addi %arg1, %arg2 : index
          %22 = arith.addi %21, %19 : index
          %23 = arith.index_cast %arg2 : index to i64
          %24 = arith.sitofp %23 : i64 to f64
          %25 = arith.index_cast %20 : index to i64
          %26 = arith.sitofp %25 : i64 to f64
          %27 = arith.divf %24, %26 : f64
          %28 = arith.mulf %cst_36, %27 : f64
          %29 = arith.mulf %cst_35, %28 : f64
          %30 = math.cos %29 : f64
          %31 = math.sin %29 : f64
          %32 = memref.load %alloc_15[%22] : memref<4096xf64>
          %33 = memref.load %alloc_14[%22] : memref<4096xf64>
          %34 = arith.mulf %32, %30 : f64
          %35 = arith.mulf %33, %31 : f64
          %36 = arith.subf %34, %35 : f64
          %37 = arith.mulf %32, %31 : f64
          %38 = arith.mulf %33, %30 : f64
          %39 = arith.addf %37, %38 : f64
          %40 = memref.load %alloc_15[%21] : memref<4096xf64>
          %41 = memref.load %alloc_14[%21] : memref<4096xf64>
          %42 = arith.addf %40, %36 : f64
          %43 = arith.addf %41, %39 : f64
          %44 = arith.subf %40, %36 : f64
          %45 = arith.subf %41, %39 : f64
          memref.store %42, %alloc_15[%21] : memref<4096xf64>
          memref.store %43, %alloc_14[%21] : memref<4096xf64>
          memref.store %44, %alloc_15[%22] : memref<4096xf64>
          memref.store %45, %alloc_14[%22] : memref<4096xf64>
        }
      }
    }
    %c0_37 = arith.constant 0 : index
    %c4096_38 = arith.constant 4096 : index
    %c1_39 = arith.constant 1 : index
    %5 = arith.index_cast %c4096_38 : index to i64
    %6 = arith.sitofp %5 : i64 to f64
    %7 = math.log2 %6 : f64
    %8 = arith.fptosi %7 : f64 to i64
    %9 = arith.index_cast %8 : i64 to index
    scf.for %arg0 = %c0_37 to %c4096_38 step %c1_39 {
      %19 = arith.index_cast %arg0 : index to i64
      %c0_i64 = arith.constant 0 : i64
      %20 = scf.for %arg1 = %c0_37 to %9 step %c1_39 iter_args(%arg2 = %c0_i64) -> (i64) {
        %23 = arith.index_cast %arg1 : index to i64
        %c1_i64 = arith.constant 1 : i64
        %24 = arith.shli %c1_i64, %23 : i64
        %25 = arith.andi %19, %24 : i64
        %c0_i64_92 = arith.constant 0 : i64
        %26 = arith.cmpi ne, %25, %c0_i64_92 : i64
        %c1_93 = arith.constant 1 : index
        %27 = arith.subi %9, %arg1 : index
        %28 = arith.subi %27, %c1_93 : index
        %29 = arith.index_cast %28 : index to i64
        %c1_i64_94 = arith.constant 1 : i64
        %30 = arith.shli %c1_i64_94, %29 : i64
        %c0_i64_95 = arith.constant 0 : i64
        %31 = arith.select %26, %30, %c0_i64_95 : i64
        %32 = arith.ori %arg2, %31 : i64
        scf.yield %32 : i64
      }
      %21 = arith.index_cast %20 : i64 to index
      %22 = memref.load %alloc_18[%arg0] : memref<4096xf64>
      %cst_91 = arith.constant 0.000000e+00 : f64
      memref.store %22, %alloc_11[%21] : memref<4096xf64>
      memref.store %cst_91, %alloc_10[%21] : memref<4096xf64>
    }
    %c12_40 = arith.constant 12 : index
    %cst_41 = arith.constant 3.1415926535897931 : f64
    %cst_42 = arith.constant -2.000000e+00 : f64
    scf.for %arg0 = %c0_37 to %c12_40 step %c1_39 {
      %c1_91 = arith.constant 1 : index
      %19 = arith.shli %c1_91, %arg0 : index
      %c1_92 = arith.constant 1 : index
      %20 = arith.shli %19, %c1_92 : index
      scf.for %arg1 = %c0_37 to %c4096_38 step %20 {
        scf.for %arg2 = %c0_37 to %19 step %c1_39 {
          %21 = arith.addi %arg1, %arg2 : index
          %22 = arith.addi %21, %19 : index
          %23 = arith.index_cast %arg2 : index to i64
          %24 = arith.sitofp %23 : i64 to f64
          %25 = arith.index_cast %20 : index to i64
          %26 = arith.sitofp %25 : i64 to f64
          %27 = arith.divf %24, %26 : f64
          %28 = arith.mulf %cst_42, %27 : f64
          %29 = arith.mulf %cst_41, %28 : f64
          %30 = math.cos %29 : f64
          %31 = math.sin %29 : f64
          %32 = memref.load %alloc_11[%22] : memref<4096xf64>
          %33 = memref.load %alloc_10[%22] : memref<4096xf64>
          %34 = arith.mulf %32, %30 : f64
          %35 = arith.mulf %33, %31 : f64
          %36 = arith.subf %34, %35 : f64
          %37 = arith.mulf %32, %31 : f64
          %38 = arith.mulf %33, %30 : f64
          %39 = arith.addf %37, %38 : f64
          %40 = memref.load %alloc_11[%21] : memref<4096xf64>
          %41 = memref.load %alloc_10[%21] : memref<4096xf64>
          %42 = arith.addf %40, %36 : f64
          %43 = arith.addf %41, %39 : f64
          %44 = arith.subf %40, %36 : f64
          %45 = arith.subf %41, %39 : f64
          memref.store %42, %alloc_11[%21] : memref<4096xf64>
          memref.store %43, %alloc_10[%21] : memref<4096xf64>
          memref.store %44, %alloc_11[%22] : memref<4096xf64>
          memref.store %45, %alloc_10[%22] : memref<4096xf64>
        }
      }
    }
    affine.for %arg0 = 0 to 4096 {
      %19 = affine.load %alloc_15[%arg0] : memref<4096xf64>
      %20 = arith.mulf %19, %19 : f64
      affine.store %20, %alloc_9[%arg0] : memref<4096xf64>
    }
    affine.for %arg0 = 0 to 4096 {
      %19 = affine.load %alloc_10[%arg0] : memref<4096xf64>
      %20 = arith.mulf %19, %19 : f64
      affine.store %20, %alloc_8[%arg0] : memref<4096xf64>
    }
    affine.for %arg0 = 0 to 4096 {
      %19 = affine.load %alloc_9[%arg0] : memref<4096xf64>
      %20 = affine.load %alloc_8[%arg0] : memref<4096xf64>
      %21 = arith.addf %19, %20 : f64
      affine.store %21, %alloc_7[%arg0] : memref<4096xf64>
    }
    affine.for %arg0 = 0 to 4096 {
      %19 = affine.load %alloc_7[%arg0] : memref<4096xf64>
      %20 = math.sqrt %19 : f64
      affine.store %20, %alloc_6[%arg0] : memref<4096xf64>
    }
    %c0_43 = arith.constant 0 : index
    %cst_44 = arith.constant 4.096000e+03 : f64
    affine.store %cst_44, %alloc_5[] : memref<f64>
    %c0_45 = arith.constant 0 : index
    %cst_46 = arith.constant 1.220000e-04 : f64
    affine.store %cst_46, %alloc_4[] : memref<f64>
    %cst_47 = arith.constant 4.096000e+03 : f64
    %cst_48 = arith.constant 1.2199999764561653E-4 : f64
    %c0_49 = arith.constant 0 : index
    %c4096_50 = arith.constant 4096 : index
    %c1_51 = arith.constant 1 : index
    %10 = arith.mulf %cst_47, %cst_48 : f64
    %cst_52 = arith.constant 5.000000e-01 : f64
    %cst_53 = arith.constant 1.000000e+00 : f64
    %11 = arith.subf %cst_47, %cst_53 : f64
    %12 = arith.mulf %11, %cst_52 : f64
    scf.for %arg0 = %c0_49 to %c4096_50 step %c1_51 {
      %19 = arith.index_cast %arg0 : index to i64
      %20 = arith.sitofp %19 : i64 to f64
      %21 = arith.cmpf ole, %20, %12 : f64
      %22 = scf.if %21 -> (f64) {
        %23 = arith.divf %20, %10 : f64
        memref.store %23, %alloc_3[%arg0] : memref<4096xf64>
        scf.yield %23 : f64
      } else {
        %23 = arith.subf %20, %cst_47 : f64
        %24 = arith.divf %23, %10 : f64
        memref.store %24, %alloc_3[%arg0] : memref<4096xf64>
        scf.yield %24 : f64
      }
    }
    %cst_54 = arith.constant 0.000000e+00 : f64
    %cst_55 = arith.constant 0.000000e+00 : f64
    %cst_56 = arith.constant 0.000000e+00 : f64
    %cst_57 = arith.constant 0.000000e+00 : f64
    %c0_58 = arith.constant 0 : index
    %c4096_59 = arith.constant 4096 : index
    %c1_60 = arith.constant 1 : index
    %13:4 = scf.for %arg0 = %c0_58 to %c4096_59 step %c1_60 iter_args(%arg1 = %cst_54, %arg2 = %cst_55, %arg3 = %cst_56, %arg4 = %cst_57) -> (f64, f64, f64, f64) {
      %19 = memref.load %alloc_3[%arg0] : memref<4096xf64>
      %20 = memref.load %alloc_6[%arg0] : memref<4096xf64>
      %cst_91 = arith.constant 0.000000e+00 : f64
      %21 = arith.cmpf ogt, %19, %cst_91 : f64
      %22:4 = scf.if %21 -> (f64, f64, f64, f64) {
        %23 = arith.cmpf ogt, %20, %arg1 : f64
        %24:4 = scf.if %23 -> (f64, f64, f64, f64) {
          scf.yield %20, %arg1, %19, %arg3 : f64, f64, f64, f64
        } else {
          %25 = arith.cmpf ogt, %20, %arg2 : f64
          %26:4 = scf.if %25 -> (f64, f64, f64, f64) {
            scf.yield %arg1, %20, %arg3, %19 : f64, f64, f64, f64
          } else {
            scf.yield %arg1, %arg2, %arg3, %arg4 : f64, f64, f64, f64
          }
          scf.yield %26#0, %26#1, %26#2, %26#3 : f64, f64, f64, f64
        }
        scf.yield %24#0, %24#1, %24#2, %24#3 : f64, f64, f64, f64
      } else {
        scf.yield %arg1, %arg2, %arg3, %arg4 : f64, f64, f64, f64
      }
      scf.yield %22#0, %22#1, %22#2, %22#3 : f64, f64, f64, f64
    }
    %c0_61 = arith.constant 0 : index
    memref.store %13#2, %alloc_2[%c0_61] : memref<2xf64>
    %c1_62 = arith.constant 1 : index
    memref.store %13#3, %alloc_2[%c1_62] : memref<2xf64>
    %c0_63 = arith.constant 0 : index
    %c1_64 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %c8 = arith.constant 8 : index
    %c9 = arith.constant 9 : index
    %cst_65 = arith.constant 9.410000e+02 : f64
    affine.store %cst_65, %alloc_1[%c0_63, %c0_63] : memref<10x2xf64>
    %cst_66 = arith.constant 1.336000e+03 : f64
    affine.store %cst_66, %alloc_1[%c0_63, %c1_64] : memref<10x2xf64>
    %cst_67 = arith.constant 6.970000e+02 : f64
    affine.store %cst_67, %alloc_1[%c1_64, %c0_63] : memref<10x2xf64>
    %cst_68 = arith.constant 1.209000e+03 : f64
    affine.store %cst_68, %alloc_1[%c1_64, %c1_64] : memref<10x2xf64>
    %cst_69 = arith.constant 6.970000e+02 : f64
    affine.store %cst_69, %alloc_1[%c2, %c0_63] : memref<10x2xf64>
    %cst_70 = arith.constant 1.336000e+03 : f64
    affine.store %cst_70, %alloc_1[%c2, %c1_64] : memref<10x2xf64>
    %cst_71 = arith.constant 6.970000e+02 : f64
    affine.store %cst_71, %alloc_1[%c3, %c0_63] : memref<10x2xf64>
    %cst_72 = arith.constant 1.477000e+03 : f64
    affine.store %cst_72, %alloc_1[%c3, %c1_64] : memref<10x2xf64>
    %cst_73 = arith.constant 7.700000e+02 : f64
    affine.store %cst_73, %alloc_1[%c4, %c0_63] : memref<10x2xf64>
    %cst_74 = arith.constant 1.209000e+03 : f64
    affine.store %cst_74, %alloc_1[%c4, %c1_64] : memref<10x2xf64>
    %cst_75 = arith.constant 7.700000e+02 : f64
    affine.store %cst_75, %alloc_1[%c5, %c0_63] : memref<10x2xf64>
    %cst_76 = arith.constant 1.336000e+03 : f64
    affine.store %cst_76, %alloc_1[%c5, %c1_64] : memref<10x2xf64>
    %cst_77 = arith.constant 7.700000e+02 : f64
    affine.store %cst_77, %alloc_1[%c6, %c0_63] : memref<10x2xf64>
    %cst_78 = arith.constant 1.477000e+03 : f64
    affine.store %cst_78, %alloc_1[%c6, %c1_64] : memref<10x2xf64>
    %cst_79 = arith.constant 8.520000e+02 : f64
    affine.store %cst_79, %alloc_1[%c7, %c0_63] : memref<10x2xf64>
    %cst_80 = arith.constant 1.209000e+03 : f64
    affine.store %cst_80, %alloc_1[%c7, %c1_64] : memref<10x2xf64>
    %cst_81 = arith.constant 8.520000e+02 : f64
    affine.store %cst_81, %alloc_1[%c8, %c0_63] : memref<10x2xf64>
    %cst_82 = arith.constant 1.336000e+03 : f64
    affine.store %cst_82, %alloc_1[%c8, %c1_64] : memref<10x2xf64>
    %cst_83 = arith.constant 8.520000e+02 : f64
    affine.store %cst_83, %alloc_1[%c9, %c0_63] : memref<10x2xf64>
    %cst_84 = arith.constant 1.477000e+03 : f64
    affine.store %cst_84, %alloc_1[%c9, %c1_64] : memref<10x2xf64>
    %c0_85 = arith.constant 0 : index
    %c1_86 = arith.constant 1 : index
    %14 = memref.load %alloc_2[%c0_85] : memref<2xf64>
    %15 = memref.load %alloc_2[%c1_86] : memref<2xf64>
    %c-1 = arith.constant -1 : index
    affine.store %c-1, %alloc[] : memref<index>
    %cst_87 = arith.constant 3.000000e+00 : f64
    %c0_88 = arith.constant 0 : index
    %c10 = arith.constant 10 : index
    %c1_89 = arith.constant 1 : index
    scf.for %arg0 = %c0_88 to %c10 step %c1_89 {
      %19 = memref.load %alloc[] : memref<index>
      %20 = memref.load %alloc_1[%arg0, %c0_85] : memref<10x2xf64>
      %21 = memref.load %alloc_1[%arg0, %c1_86] : memref<10x2xf64>
      %22 = arith.subf %20, %14 : f64
      %23 = arith.subf %21, %15 : f64
      %24 = math.absf %22 : f64
      %25 = math.absf %23 : f64
      %26 = arith.cmpf ole, %24, %cst_87 : f64
      %27 = arith.cmpf ole, %25, %cst_87 : f64
      %28 = arith.andi %26, %27 : i1
      %29 = arith.select %28, %arg0, %19 : index
      memref.store %29, %alloc[] : memref<index>
    }
    %16 = memref.load %alloc[] : memref<index>
    %17 = arith.index_cast %16 : index to i64
    %18 = arith.sitofp %17 : i64 to f64
    %c0_90 = arith.constant 0 : index
    memref.store %18, %alloc_0[%c0_90] : memref<1xf64>
    dsp.print %alloc_0 : memref<1xf64>
    memref.dealloc %alloc_21 : memref<f64>
    memref.dealloc %alloc_20 : memref<f64>
    memref.dealloc %alloc_19 : memref<f64>
    memref.dealloc %alloc_18 : memref<4096xf64>
    memref.dealloc %alloc_17 : memref<4096xf64>
    memref.dealloc %alloc_16 : memref<4096xf64>
    memref.dealloc %alloc_15 : memref<4096xf64>
    memref.dealloc %alloc_14 : memref<4096xf64>
    memref.dealloc %alloc_13 : memref<4096xf64>
    memref.dealloc %alloc_12 : memref<4096xf64>
    memref.dealloc %alloc_11 : memref<4096xf64>
    memref.dealloc %alloc_10 : memref<4096xf64>
    memref.dealloc %alloc_9 : memref<4096xf64>
    memref.dealloc %alloc_8 : memref<4096xf64>
    memref.dealloc %alloc_7 : memref<4096xf64>
    memref.dealloc %alloc_6 : memref<4096xf64>
    memref.dealloc %alloc_5 : memref<f64>
    memref.dealloc %alloc_4 : memref<f64>
    memref.dealloc %alloc_3 : memref<4096xf64>
    memref.dealloc %alloc_2 : memref<2xf64>
    memref.dealloc %alloc_1 : memref<10x2xf64>
    memref.dealloc %alloc_0 : memref<1xf64>
    memref.dealloc %alloc : memref<index>
    return
  }
}