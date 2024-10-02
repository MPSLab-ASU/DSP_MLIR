# RUN: toyc-ch2 %s -emit=mlir 2>&1 | FileCheck %s

def main() {
  # var a = [10,20,30];
  # var b = [40,50,60];
  # var a = [[10,20],[30,40]];
  # var b = [[40,50],[60,70]];

  # var a = [[[10,20],[30,40]] , [[10,20],[30,40]]];
  # var b = [[[40,50],[60,70]] , [[0,0],[10,20]]];
  var a = [[[10,-20],[30.9,0]] ];
  var b = [[[40,50],[-60,70]] ];
  var c = sub(a, b);
  print(c);
}
# /home/local/ASUAD/apkhedka/ForLLVM/build/bin/dsp1 /home/local/ASUAD/apkhedka/ForLLVM/mlir/test/Examples/DspExample/dsp_sub_op.py -emit=mlir

# module {
#   dsp.func @main() {
#     %0 = dsp.constant dense<[1.000000e+01, 2.000000e+01, 3.000000e+01]> : tensor<3xf64>
#     %1 = dsp.constant dense<[4.000000e+01, 5.000000e+01, 6.000000e+01]> : tensor<3xf64>
#     %2 = "dsp.sub"(%0, %1) : (tensor<3xf64>, tensor<3xf64>) -> tensor<*xf64>
#     dsp.print %2 : tensor<*xf64>
#     dsp.return
#   }
# }
