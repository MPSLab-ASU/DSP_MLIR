def main() {
  var a = [ 0,  0.96193977, -0.35355339, -0.69134172,  0.5, 0.30865828, -0.35355339, -0.03806023];
  var b = fftReal(a);
  var squared = square(b);
  # var c = fftImag(a);
  print(squared);
  # print(c);
} 
