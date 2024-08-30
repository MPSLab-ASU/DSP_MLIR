# RUN: dsp1 %s -emit=mlir 2>&1 | FileCheck %s

def main() {
  var a = [4,20];
  var b = 4;
  #var c = pow(a, b);
  var c = a^b;
  print(c);
}
# /home/local/ASUAD/apkhedka/ForLLVM/build/bin/dsp1 /home/local/ASUAD/apkhedka/ForLLVM/mlir/test/Examples/DspExample/dsp_pow_op.py -emit=mlir

# CHECK-LABEL: dsp.func @main() {
# CHECK-NEXT:       %[[VAL_0:.*]] = dsp.constant dense<{{\[\[}}[1.000000e+01, 2.000000e+01], [3.000000e+01, 0.000000e+00]]]> : tensor<1x2x2xf64>
# CHECK-NEXT:       %[[VAL_1:.*]] = dsp.constant dense<[1.000000e+01]> : tensor<1xf64>
# CHECK-NEXT:       %[[VAL_2:.*]] = "dsp.sub"(%[[VAL_0]], %[[VAL_1]]) : (tensor<3xf64>, tensor<3xf64>) -> tensor<*xf64>
# CHECK-NEXT:       dsp.print %[[VAL_2]] : tensor<*xf64>
# CHECK-NEXT:       dsp.return
# CHECK-NEXT:       }
