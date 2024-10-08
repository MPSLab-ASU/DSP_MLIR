func.func @main() {
  %alloc = memref.alloc() : memref<3xf64>
  %alloc_1 = memref.alloc() : memref<f64>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %cst = arith.constant 1.000000e+01 : f64
  affine.store %cst, %alloc[%c0] : memref<3xf64>
  %cst_1 = arith.constant -1.000000e+01 : f64
  affine.store %cst_1, %alloc[%c1] : memref<3xf64>
  %cst_2 = arith.constant 1.000000e+01 : f64
  affine.store %cst_2, %alloc[%c2] : memref<3xf64>
  %lb = arith.constant 1 : index
  %ub = arith.constant 3 : index
  %step = arith.constant 1 : index
  %total_0 = arith.constant 0.0 : f64
  %c3 = arith.constant 0 : i64
  %c4 = arith.constant 1.0 : f64
  %total = scf.for %arg0 = %lb to %ub step %step 
    iter_args(%total_iter = %total_0) -> (f64) {
    %prev_idx = arith.subi %arg0, %step : index
    %1 = memref.load %alloc[%prev_idx] : memref<3xf64>
    %int_1 = arith.fptosi %1 : f64 to i64
    %sign_1 = arith.cmpi "slt", %int_1, %c3 : i64    
    %2 = memref.load %alloc[%arg0] : memref<3xf64>
    %int_2 = arith.fptosi %2 : f64 to i64
    %sign_2 = arith.cmpi "slt", %int_2, %c3 : i64    
    %cond = arith.cmpi "eq", %sign_1, %sign_2 : i1    
    %total_next = scf.if %cond -> (f64) {
      scf.yield %total_iter : f64
    } else {
      %new_total = arith.addf %total_iter, %c4 : f64
      scf.yield %new_total : f64
    }
    scf.yield %total_next : f64
  }

  affine.store %total, %alloc_1[] : memref<f64>
  // Print the value held by the buffer.
  // dsp.print %alloc : memref<3xf64>
  // Print the number of crosses through x=0
  dsp.print %alloc_1 : memref<f64>
  memref.dealloc %alloc : memref<3xf64>
  memref.dealloc %alloc_1 : memref<f64>
  return 
}