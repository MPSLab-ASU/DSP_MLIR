# RUN: toyc-ch2 %s -emit=mlir 2>&1 | FileCheck %s

def main() {
  var a = [50,50,50,50];
  var b = [2,3,4,5];
  var c = shiftRight(a, b);
  print(c);
}
