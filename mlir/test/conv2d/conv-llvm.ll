llvm.func @free(!llvm.ptr)
llvm.func @malloc(i64) -> !llvm.ptr
llvm.func @main() {
    %0 = llvm.mlir.constant(4 : index) : i64
    %1 = llvm.mlir.constant(4 : index) : i64
    %2 = llvm.mlir.constant(1 : index) : i64
    %3 = llvm.mlir.constant(16 : index) : i64
    %4 = llvm.mlir.zero : !llvm.ptr
    %5 = llvm.getelementptr %4[%3] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %6 = llvm.ptrtoint %5 : !llvm.ptr to i64
    %7 = llvm.call @malloc(%6) : (i64) -> !llvm.ptr
    %8 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %9 = llvm.insertvalue %7, %8[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %10 = llvm.insertvalue %7, %9[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %11 = llvm.mlir.constant(0 : index) : i64
    %12 = llvm.insertvalue %11, %10[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %13 = llvm.insertvalue %0, %12[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %14 = llvm.insertvalue %1, %13[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %15 = llvm.insertvalue %1, %14[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %16 = llvm.insertvalue %2, %15[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %17 = llvm.mlir.constant(3 : index) : i64
    %18 = llvm.mlir.constant(3 : index) : i64
    %19 = llvm.mlir.constant(1 : index) : i64
    %20 = llvm.mlir.constant(9 : index) : i64
    %21 = llvm.mlir.zero : !llvm.ptr
    %22 = llvm.getelementptr %21[%20] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %23 = llvm.ptrtoint %22 : !llvm.ptr to i64
    %24 = llvm.call @malloc(%23) : (i64) -> !llvm.ptr
    %25 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %26 = llvm.insertvalue %24, %25[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %27 = llvm.insertvalue %24, %26[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %28 = llvm.mlir.constant(0 : index) : i64
    %29 = llvm.insertvalue %28, %27[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %30 = llvm.insertvalue %17, %29[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %31 = llvm.insertvalue %18, %30[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %32 = llvm.insertvalue %18, %31[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %33 = llvm.insertvalue %19, %32[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %34 = llvm.mlir.constant(4 : index) : i64
    %35 = llvm.mlir.constant(4 : index) : i64
    %36 = llvm.mlir.constant(1 : index) : i64
    %37 = llvm.mlir.constant(16 : index) : i64
    %38 = llvm.mlir.zero : !llvm.ptr
    %39 = llvm.getelementptr %38[%37] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    %40 = llvm.ptrtoint %39 : !llvm.ptr to i64
    %41 = llvm.call @malloc(%40) : (i64) -> !llvm.ptr
    %42 = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
    %43 = llvm.insertvalue %41, %42[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %44 = llvm.insertvalue %41, %43[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %45 = llvm.mlir.constant(0 : index) : i64
    %46 = llvm.insertvalue %45, %44[2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %47 = llvm.insertvalue %34, %46[3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %48 = llvm.insertvalue %35, %47[3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %49 = llvm.insertvalue %35, %48[4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %50 = llvm.insertvalue %36, %49[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %51 = llvm.mlir.constant(0 : i64) : i64
    %52 = llvm.mlir.constant(1 : i64) : i64
    %53 = llvm.mlir.constant(2 : i64) : i64
    %54 = llvm.mlir.constant(3 : i64) : i64
    %55 = llvm.mlir.constant(0.000000e+00 : f32) : f32
    %56 = llvm.mlir.constant(1.000000e+00 : f32) : f32
    %57 = llvm.mlir.constant(2.000000e+00 : f32) : f32
    %58 = llvm.mlir.constant(3.000000e+00 : f32) : f32
    %59 = llvm.mlir.constant(4.000000e+00 : f32) : f32
    %60 = llvm.mlir.constant(5.000000e+00 : f32) : f32
    %61 = llvm.mlir.constant(6.000000e+00 : f32) : f32
    %62 = llvm.mlir.constant(7.000000e+00 : f32) : f32
    %63 = llvm.mlir.constant(8.000000e+00 : f32) : f32
    %64 = llvm.mlir.constant(-1.000000e+00 : f32) : f32
    %65 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %66 = llvm.mlir.constant(4 : index) : i64
    %67 = llvm.mul %51, %66 : i64
    %68 = llvm.add %67, %51 : i64
    %69 = llvm.getelementptr %65[%68] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %56, %69 : f32, !llvm.ptr
    %70 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %71 = llvm.mlir.constant(4 : index) : i64
    %72 = llvm.mul %51, %71 : i64
    %73 = llvm.add %72, %52 : i64
    %74 = llvm.getelementptr %70[%73] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %57, %74 : f32, !llvm.ptr
    %75 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %76 = llvm.mlir.constant(4 : index) : i64
    %77 = llvm.mul %51, %76 : i64
    %78 = llvm.add %77, %53 : i64
    %79 = llvm.getelementptr %75[%78] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %58, %79 : f32, !llvm.ptr
    %80 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %81 = llvm.mlir.constant(4 : index) : i64
    %82 = llvm.mul %51, %81 : i64
    %83 = llvm.add %82, %54 : i64
    %84 = llvm.getelementptr %80[%83] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %59, %84 : f32, !llvm.ptr
    %85 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %86 = llvm.mlir.constant(4 : index) : i64
    %87 = llvm.mul %52, %86 : i64
    %88 = llvm.add %87, %51 : i64
    %89 = llvm.getelementptr %85[%88] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %57, %89 : f32, !llvm.ptr
    %90 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %91 = llvm.mlir.constant(4 : index) : i64
    %92 = llvm.mul %52, %91 : i64
    %93 = llvm.add %92, %52 : i64
    %94 = llvm.getelementptr %90[%93] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %58, %94 : f32, !llvm.ptr
    %95 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %96 = llvm.mlir.constant(4 : index) : i64
    %97 = llvm.mul %52, %96 : i64
    %98 = llvm.add %97, %53 : i64
    %99 = llvm.getelementptr %95[%98] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %59, %99 : f32, !llvm.ptr
    %100 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %101 = llvm.mlir.constant(4 : index) : i64
    %102 = llvm.mul %52, %101 : i64
    %103 = llvm.add %102, %54 : i64
    %104 = llvm.getelementptr %100[%103] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %61, %104 : f32, !llvm.ptr
    %105 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %106 = llvm.mlir.constant(4 : index) : i64
    %107 = llvm.mul %53, %106 : i64
    %108 = llvm.add %107, %51 : i64
    %109 = llvm.getelementptr %105[%108] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %59, %109 : f32, !llvm.ptr
    %110 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %111 = llvm.mlir.constant(4 : index) : i64
    %112 = llvm.mul %53, %111 : i64
    %113 = llvm.add %112, %52 : i64
    %114 = llvm.getelementptr %110[%113] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %58, %114 : f32, !llvm.ptr
    %115 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %116 = llvm.mlir.constant(4 : index) : i64
    %117 = llvm.mul %53, %116 : i64
    %118 = llvm.add %117, %53 : i64
    %119 = llvm.getelementptr %115[%118] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %57, %119 : f32, !llvm.ptr
    %120 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %121 = llvm.mlir.constant(4 : index) : i64
    %122 = llvm.mul %53, %121 : i64
    %123 = llvm.add %122, %54 : i64
    %124 = llvm.getelementptr %120[%123] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %56, %124 : f32, !llvm.ptr
    %125 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %126 = llvm.mlir.constant(4 : index) : i64
    %127 = llvm.mul %54, %126 : i64
    %128 = llvm.add %127, %51 : i64
    %129 = llvm.getelementptr %125[%128] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %61, %129 : f32, !llvm.ptr
    %130 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %131 = llvm.mlir.constant(4 : index) : i64
    %132 = llvm.mul %54, %131 : i64
    %133 = llvm.add %132, %52 : i64
    %134 = llvm.getelementptr %130[%133] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %63, %134 : f32, !llvm.ptr
    %135 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %136 = llvm.mlir.constant(4 : index) : i64
    %137 = llvm.mul %54, %136 : i64
    %138 = llvm.add %137, %53 : i64
    %139 = llvm.getelementptr %135[%138] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %59, %139 : f32, !llvm.ptr
    %140 = llvm.extractvalue %16[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %141 = llvm.mlir.constant(4 : index) : i64
    %142 = llvm.mul %54, %141 : i64
    %143 = llvm.add %142, %54 : i64
    %144 = llvm.getelementptr %140[%143] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %62, %144 : f32, !llvm.ptr
    %145 = llvm.extractvalue %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %146 = llvm.mlir.constant(3 : index) : i64
    %147 = llvm.mul %51, %146 : i64
    %148 = llvm.add %147, %51 : i64
    %149 = llvm.getelementptr %145[%148] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %56, %149 : f32, !llvm.ptr
    %150 = llvm.extractvalue %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %151 = llvm.mlir.constant(3 : index) : i64
    %152 = llvm.mul %51, %151 : i64
    %153 = llvm.add %152, %52 : i64
    %154 = llvm.getelementptr %150[%153] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %55, %154 : f32, !llvm.ptr
    %155 = llvm.extractvalue %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %156 = llvm.mlir.constant(3 : index) : i64
    %157 = llvm.mul %51, %156 : i64
    %158 = llvm.add %157, %53 : i64
    %159 = llvm.getelementptr %155[%158] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %64, %159 : f32, !llvm.ptr
    %160 = llvm.extractvalue %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %161 = llvm.mlir.constant(3 : index) : i64
    %162 = llvm.mul %52, %161 : i64
    %163 = llvm.add %162, %51 : i64
    %164 = llvm.getelementptr %160[%163] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %56, %164 : f32, !llvm.ptr
    %165 = llvm.extractvalue %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %166 = llvm.mlir.constant(3 : index) : i64
    %167 = llvm.mul %52, %166 : i64
    %168 = llvm.add %167, %52 : i64
    %169 = llvm.getelementptr %165[%168] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %55, %169 : f32, !llvm.ptr
    %170 = llvm.extractvalue %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %171 = llvm.mlir.constant(3 : index) : i64
    %172 = llvm.mul %52, %171 : i64
    %173 = llvm.add %172, %53 : i64
    %174 = llvm.getelementptr %170[%173] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %64, %174 : f32, !llvm.ptr
    %175 = llvm.extractvalue %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %176 = llvm.mlir.constant(3 : index) : i64
    %177 = llvm.mul %53, %176 : i64
    %178 = llvm.add %177, %51 : i64
    %179 = llvm.getelementptr %175[%178] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %56, %179 : f32, !llvm.ptr
    %180 = llvm.extractvalue %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %181 = llvm.mlir.constant(3 : index) : i64
    %182 = llvm.mul %53, %181 : i64
    %183 = llvm.add %182, %52 : i64
    %184 = llvm.getelementptr %180[%183] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %55, %184 : f32, !llvm.ptr
    %185 = llvm.extractvalue %33[1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    %186 = llvm.mlir.constant(3 : index) : i64
    %187 = llvm.mul %53, %186 : i64
    %188 = llvm.add %187, %53 : i64
    %189 = llvm.getelementptr %185[%188] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %64, %189 : f32, !llvm.ptr
    %190 = llvm.mlir.constant(1.500000e+00 : f32) : f32
    %191 = llvm.fptoui %190 : f32 to i32
    %192 = llvm.sext %191 : i32 to i64
    %193 = llvm.extractvalue %16[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%193) : (!llvm.ptr) -> ()
    %194 = llvm.extractvalue %33[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%194) : (!llvm.ptr) -> ()
    %195 = llvm.extractvalue %50[0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)> 
    llvm.call @free(%195) : (!llvm.ptr) -> ()
    llvm.return
}

