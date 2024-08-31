# RUN: toyc-ch2 %s -emit=mlir 2>&1 | FileCheck %s

def main() {
  var a = [0,1,2,9,1000];
  var b = [2,2,2,15,100];
  var c = bitwiseand(a, b);
  # c = [0,0,2,9,96]
  print(c);
}
# ninja && ./bin/dsp1 ../mlir/test/Examples/DspExample/dsp_bitwiseand_op.py --emit=mlir

# module {
#   dsp.func @main() {
#     %0 = dsp.constant dense<[0.000000e+00, 1.000000e+00, 2.000000e+00, 9.000000e+00, 1.000000e+03]> : tensor<5xf64>
#     %1 = dsp.constant dense<[2.000000e+00, 2.000000e+00, 2.000000e+00, 1.500000e+01, 1.000000e+02]> : tensor<5xf64>
#     %2 = dsp.bitwiseand %0, %1 : (tensor<5xf64>, tensor<5xf64>) -> tensor<*xf64>
#     dsp.print %2 : tensor<*xf64>
#     dsp.return
#   }
# }
