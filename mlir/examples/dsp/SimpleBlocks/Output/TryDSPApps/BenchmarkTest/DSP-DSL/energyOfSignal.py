
def main() {

	var input = getRangeOfVector(0, 40000, 1);
  #calculate x[l] 


  # var fft1 = fft1d(input);
  #calculate fft : fft1 = fft(conv1)
  var fft_real = fft1dreal(input);
  var fft_img = fft1dimg(input);
  var sq_abs = square(fft_real) + square(fft_img)  ;
  # var sq_abs = square(fft1);
  # sum = sum(sq_abs)
  var sum1 = sum(sq_abs);
  # res = gain(sum , 1/N)
  var len1 = len(input);
  var res = sum1 / len1;
  print(res);
}

