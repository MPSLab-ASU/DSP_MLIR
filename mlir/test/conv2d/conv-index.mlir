module {
    func.func @main() {
        %input = memref.alloc() : memref<4x4xf32>
        %kernel = memref.alloc() : memref<3x3xf32>
        %output = memref.alloc() : memref<4x4xf32>

        %c0 = index.constant 0 
        %c1 = index.constant 1
        %c2 = index.constant 2
        %c3 = index.constant 3

        %cst0 = arith.constant 0.000000e+00 : f32
        %cst1 = arith.constant 1.000000e+00 : f32
        %cst2 = arith.constant 2.000000e+00 : f32
        %cst3 = arith.constant 3.000000e+00 : f32
        %cst4 = arith.constant 4.000000e+00 : f32
        %cst5 = arith.constant 5.000000e+00 : f32
        %cst6 = arith.constant 6.000000e+00 : f32
        %cst7 = arith.constant 7.000000e+00 : f32
        %cst8 = arith.constant 8.000000e+00 : f32
        %cstn1 = arith.constant -1.000000e+00 : f32

        // input
        affine.store %cst1, %input[%c0, %c0] : memref<4x4xf32>
        affine.store %cst2, %input[%c0, %c1] : memref<4x4xf32>
        affine.store %cst3, %input[%c0, %c2] : memref<4x4xf32>
        affine.store %cst4, %input[%c0, %c3] : memref<4x4xf32>
        
        affine.store %cst2, %input[%c1, %c0] : memref<4x4xf32>
        affine.store %cst3, %input[%c1, %c1] : memref<4x4xf32>
        affine.store %cst4, %input[%c1, %c2] : memref<4x4xf32>
        affine.store %cst6, %input[%c1, %c3] : memref<4x4xf32>

        affine.store %cst4, %input[%c2, %c0] : memref<4x4xf32>
        affine.store %cst3, %input[%c2, %c1] : memref<4x4xf32>
        affine.store %cst2, %input[%c2, %c2] : memref<4x4xf32>
        affine.store %cst1, %input[%c2, %c3] : memref<4x4xf32>

        affine.store %cst6, %input[%c3, %c0] : memref<4x4xf32>
        affine.store %cst8, %input[%c3, %c1] : memref<4x4xf32>
        affine.store %cst4, %input[%c3, %c2] : memref<4x4xf32>
        affine.store %cst7, %input[%c3, %c3] : memref<4x4xf32>

        // kernel
        affine.store %cst1, %kernel[%c0, %c0] : memref<3x3xf32>
        affine.store %cst0, %kernel[%c0, %c1] : memref<3x3xf32>
        affine.store %cstn1, %kernel[%c0, %c2] : memref<3x3xf32>

        affine.store %cst1, %kernel[%c1, %c0] : memref<3x3xf32>
        affine.store %cst0, %kernel[%c1, %c1] : memref<3x3xf32>
        affine.store %cstn1, %kernel[%c1, %c2] : memref<3x3xf32>

        affine.store %cst1, %kernel[%c2, %c0] : memref<3x3xf32>
        affine.store %cst0, %kernel[%c2, %c1] : memref<3x3xf32>
        affine.store %cstn1, %kernel[%c2, %c2] : memref<3x3xf32>

        
        // delta
        %delta = arith.divf %cst3, %cst2 : f32
        %delta_i = arith.fptoui %delta : f32 to i32    
        %delta_dim = arith.index_cast %delta_i : i32 to index

        memref.dealloc %input : memref<4x4xf32>
        memref.dealloc %kernel : memref<3x3xf32>
        memref.dealloc %output : memref<4x4xf32>
        return
    }
}
