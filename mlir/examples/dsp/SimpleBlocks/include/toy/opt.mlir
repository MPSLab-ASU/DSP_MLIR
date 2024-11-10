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
    dsp.print %alloc_18 : memref<1xf64>
    memref.dealloc %alloc_30 : memref<f64>
    memref.dealloc %alloc_29 : memref<f64>
    memref.dealloc %alloc_28 : memref<f64>
    memref.dealloc %alloc_27 : memref<4096xf64>
    memref.dealloc %alloc_26 : memref<4096xf64>
    memref.dealloc %alloc_25 : memref<4096xf64>
    memref.dealloc %alloc_24 : memref<4096xf64>
    memref.dealloc %alloc_23 : memref<f64>
    memref.dealloc %alloc_22 : memref<f64>
    memref.dealloc %alloc_21 : memref<4096xf64>
    memref.dealloc %alloc_20 : memref<2xf64>
    memref.dealloc %alloc_19 : memref<10x2xf64>
    memref.dealloc %alloc_18 : memref<1xf64>
    memref.dealloc %alloc : memref<index>
    return
  }
}