module {
    func.func @main() -> i32 {
        %arg0 = arith.constant 10 : i32
        %arg1 = arith.constant 122: i32

        %result = call @foo(%arg0, %arg1) : (i32, i32) -> i32
        return %result : i32
    }

    func.func @foo(%arg0: i32, %arg1: i32) -> i32 {
        %0 = arith.addi %arg0, %arg1: i32
        return %0: i32
    }
}
