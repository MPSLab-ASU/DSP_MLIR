func.func @matmul(%A: memref<128x128xf32>, %B: memref<128x128xf32>, %C: memref<128x128xf32>) {
    affine.for %i = 0 to 128 {
        affine.for %j = 0 to 128 {
            affine.for %k = 0 to 128 {
                        %a = affine.load %A[%i, %k] : memref<128x128xf32>
                                %b = affine.load %B[%k, %j] : memref<128x128xf32>
                                        %c = affine.load %C[%i, %j] : memref<128x128xf32>
                                                %mul = arith.mulf %a, %b : f32
                                                        %add = arith.addf %mul, %c : f32
                                                                affine.store %add, %C[%i, %j] : memref<128x128xf32>
                                                                      
            }
                
        }
          
    }
      return
      
}

