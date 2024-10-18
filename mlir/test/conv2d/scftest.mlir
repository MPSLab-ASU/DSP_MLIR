module {
func.func @main(%buffer: memref<1024xf32>, %lb: index,
        %ub: index, %step: index) -> (f32) {
              // Initial sum set to 0.
                %sum_0 = arith.constant 0.0 : f32
                  // iter_args binds initial values to the loop's region arguments.
                    %sum = scf.for %iv = %lb to %ub step %step
                    iter_args(%sum_iter = %sum_0) -> (f32) {
                            %t = memref.load %buffer[%iv] : memref<1024xf32>
                                %sum_next = arith.addf %sum_iter, %t : f32
                                    // Yield current iteration sum to next iteration %sum_iter or to %sum
                                        // if final iteration.
                                            scf.yield %sum_next : f32
                                              
                    }
                      return %sum : f32
                      
}
}
