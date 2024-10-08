# RUN: /bin/dsp1 %s -emit=mlir 2>&1 | FileCheck %s

# User defined generic function that operates on unknown shaped arguments
def main() {
  var a = [10,-20,30,-10,40,50,60,-100,-20,-30,10]; # Count should be 6
  var g = zeroCrossCount(a);
  print(g);
}
