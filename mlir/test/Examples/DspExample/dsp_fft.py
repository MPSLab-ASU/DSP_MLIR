def main() {
  var a = [0.0, 10.0, 340.0, 30.0, 40.0, 110.0, 60.0, 250.0];
  var b = fftReal(a);
  var c = fftImag(a);
  print(b);
  print(c);
}
