def main() {
    var a = generateDtmf(7, 0.5, 16384);
  var b = fft1dreal(a);
  var c = fft1dimg(a);
  print(b);
  print(c);
} 
  