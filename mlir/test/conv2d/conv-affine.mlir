module {
    func.func @main() {
        %input = memref.alloc() : memref<4x4xf64>
        %kernel = memref.alloc() : memref<3x3xf64>
        %output = memref.alloc() : memref<4x4xf64>

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %c4 = arith.constant 4 : index

        %cst0 = arith.constant 0.000000e+00 : f64
        %cst1 = arith.constant 1.000000e+00 : f64
        %cst2 = arith.constant 2.000000e+00 : f64
        %cst3 = arith.constant 3.000000e+00 : f64
        %cst4 = arith.constant 4.000000e+00 : f64
        %cst5 = arith.constant 5.000000e+00 : f64
        %cst6 = arith.constant 6.000000e+00 : f64
        %cst7 = arith.constant 7.000000e+00 : f64
        %cst8 = arith.constant 8.000000e+00 : f64
        %cstn1float = arith.constant -1.000000e+00 : f64
        %cstn1int = arith.constant -1 : i64

        // input
        affine.store %cst1, %input[%c0, %c0] : memref<4x4xf64>
        affine.store %cst2, %input[%c0, %c1] : memref<4x4xf64>
        affine.store %cst3, %input[%c0, %c2] : memref<4x4xf64>
        affine.store %cst4, %input[%c0, %c3] : memref<4x4xf64>
        
        affine.store %cst2, %input[%c1, %c0] : memref<4x4xf64>
        affine.store %cst3, %input[%c1, %c1] : memref<4x4xf64>
        affine.store %cst4, %input[%c1, %c2] : memref<4x4xf64>
        affine.store %cst6, %input[%c1, %c3] : memref<4x4xf64>

        affine.store %cst4, %input[%c2, %c0] : memref<4x4xf64>
        affine.store %cst3, %input[%c2, %c1] : memref<4x4xf64>
        affine.store %cst2, %input[%c2, %c2] : memref<4x4xf64>
        affine.store %cst1, %input[%c2, %c3] : memref<4x4xf64>

        affine.store %cst6, %input[%c3, %c0] : memref<4x4xf64>
        affine.store %cst8, %input[%c3, %c1] : memref<4x4xf64>
        affine.store %cst4, %input[%c3, %c2] : memref<4x4xf64>
        affine.store %cst7, %input[%c3, %c3] : memref<4x4xf64>

        // dsp.print %input : memref<4x4xf64>

        // kernel
        affine.store %cst1, %kernel[%c0, %c0] : memref<3x3xf64>
        affine.store %cst0, %kernel[%c0, %c1] : memref<3x3xf64>
        affine.store %cstn1float, %kernel[%c0, %c2] : memref<3x3xf64>

        affine.store %cst1, %kernel[%c1, %c0] : memref<3x3xf64>
        affine.store %cst0, %kernel[%c1, %c1] : memref<3x3xf64>
        affine.store %cstn1float, %kernel[%c1, %c2] : memref<3x3xf64>

        affine.store %cst1, %kernel[%c2, %c0] : memref<3x3xf64>
        affine.store %cst0, %kernel[%c2, %c1] : memref<3x3xf64>
        affine.store %cstn1float, %kernel[%c2, %c2] : memref<3x3xf64>


        // delta
        %delta_ub = arith.divf %cst3, %cst2 : f64
        %delta_lb = arith.mulf %delta_ub, %cstn1float : f64

        %ub = arith.fptosi %delta_ub : f64 to i64
        %lb = arith.fptosi %delta_lb : f64 to i64
        
        // %delta_dim_ub = arith.index_cast %ub : i64 to index
        // %delta_dim_lb = arith.index_cast %lb : i64 to index
        %delta_dim_lb = arith.constant -1 : index
        %delta_dim_ub = arith.constant 1 : index

    // for debug
        %i = memref.alloc() : memref<1xi64>
        %d = memref.alloc() : memref<1xf64>
        memref.store %cstn1float, %d[%c0] : memref<1xf64>
        memref.store %cstn1int, %i[%c0] : memref<1xi64>
        dsp.print %d : memref<1xf64>
        dsp.print %i : memref<1xi64>

        // x, y iteration
        scf.for %x = %c0 to %c4 step %c1 {
            scf.for %y = %c0 to %c4 step %c1 {
                %mat_sum = scf.for %kx = %delta_dim_lb to %delta_dim_ub step %c1 iter_args(%outer_sum = %cst0) -> ( f64 ) {
                    %ele_sum = scf.for %ky = %delta_dim_lb to %delta_dim_ub step %c1 iter_args(%inner_sum = %outer_sum) -> ( f64 ) {
                        %img_x = arith.addi %x, %kx: index
                        %img_y = arith.addi %y, %ky: index

                        %test = arith.index_cast %kx : index to i64
                        memref.store %test, %i[%c0] : memref<1xi64>
                        // dsp.print %i : memref<1xi64>

                        // sge : predicate 5
                        %cond_x_lb = "arith.cmpi"(%img_x, %c0) {predicate=5: i64} : (index, index) -> i1
                        %cond_y_lb = "arith.cmpi"(%img_y, %c0) {predicate=5: i64} : (index, index) -> i1
                        // slt
                        %cond_x_ub = "arith.cmpi"(%img_x, %c4) {predicate=2: i64} : (index, index) -> i1
                        %cond_y_ub = "arith.cmpi"(%img_y, %c4) {predicate=2: i64} : (index, index) -> i1
                        
                        %img_sum_ = scf.if %cond_x_lb -> (f64) {
                            %sum__ = scf.if %cond_y_lb -> (f64) {
                                %sum_ = scf.if %cond_x_ub -> (f64) {
                                    %sum = scf.if %cond_y_ub -> (f64) {
                                        // load from input
                                        %input_val = memref.load %input[%img_x, %img_y] : memref<4x4xf64>

                                        // load from kernel
                                        %ker_x = arith.addi %kx, %delta_dim_ub : index
                                        %ker_y = arith.addi %ky, %delta_dim_ub : index
                                        %kernel_val = memref.load %kernel[%ker_x, %ker_y] : memref<3x3xf64>

                                        %img_prod = arith.mulf %input_val, %kernel_val : f64
                                        scf.yield %img_prod : f64
                                    } else {
                                        scf.yield %cst0 : f64
                                    }
                                    scf.yield %sum : f64
                                } else {
                                    scf.yield %cst0 : f64
                                }
                                scf.yield %sum_ : f64
                            } else {
                                scf.yield %cst0 : f64
                            }
                            scf.yield %sum__ : f64
                        }else{
                            scf.yield %cst0 : f64
                        } 
                        
                        %IMGSUM = arith.addf %inner_sum, %img_sum_ : f64
                        scf.yield %IMGSUM : f64
                    }

                    scf.yield %ele_sum : f64
                }
                memref.store %mat_sum, %output[%x, %y] : memref<4x4xf64>
            }
        }
        // dsp.print %input : memref<4x4xf64>
        // dsp.print %kernel : memref<3x3xf64>
        // dsp.print %output : memref<4x4xf64>

        memref.dealloc %input : memref<4x4xf64>
        memref.dealloc %kernel : memref<3x3xf64>
        memref.dealloc %output : memref<4x4xf64>
        return
    }
}
