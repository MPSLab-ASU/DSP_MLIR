def main() {
  var a = [0.0, 10.0, 340.0, 30.0, 40.0, 110.0, 60.0, 250.0];
  var b = medianFilter(a);
  print(b);
}

# emit=mlir
# -----------
# module {
#   dsp.func @main() {
#     %0 = dsp.constant dense<[0.000000e+00, 1.000000e+01, 3.400000e+02, 3.000000e+01, 4.000000e+01, 1.100000e+02, 6.000000e+01, 2.500000e+02]> : tensor<8xf64>
#     %1 = "dsp.medianFilter"(%0) : (tensor<8xf64>) -> tensor<*xf64>
#     dsp.print %1 : tensor<*xf64>
#     dsp.return
#   }
# }

# emit=mlir-affine
