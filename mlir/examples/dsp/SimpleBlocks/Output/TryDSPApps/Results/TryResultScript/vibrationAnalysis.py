def main() {
  var fs = 1000;
  # var step = 1/fs; 
  # print(step);
	var input = getRangeOfVector(0, 100000, 1);
  var pi = 3.14159265359;
  var getMultiplier = 2 * pi * 50;
  # print(getMultiplier);
  var getSinDuration = gain(input, getMultiplier);
  var sig1 = sin(getSinDuration );
  var getMultiplier2 = 2 * pi * 120;
  var getSinDuration2 = gain(input, getMultiplier2);
  var sinsig2 = sin(getSinDuration2);
  var sig2 = gain(sinsig2, 0.5);
  var signal = sig1 + sig2;
  var noise = delay(signal, 5);
  var noisy_sig = signal + noise;
  var threshold = 0.2;
  
  var fft_real = fft1dreal(noisy_sig);
  var fft_img = fft1dimg(noisy_sig);

  var sq_abs = square(fft_real) + square(fft_img)  ;
  # sum = sum(sq_abs)
  var sum1 = sum(sq_abs);
  # res = gain(sum , 1/N)
  var len1 = len(input);
  var res = sum1 / len1;
  # print(sq_abs);
  var GetThresholdReal = threshold( sq_abs , threshold);
  print(GetThresholdReal);
}