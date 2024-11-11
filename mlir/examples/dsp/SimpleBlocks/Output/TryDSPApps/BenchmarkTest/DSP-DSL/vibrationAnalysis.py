def main() {
  var fs = 1000;
  # var step = 1/fs; 
  # print(step);
	var input = getRangeOfVector(0, 300, 0.000125);
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
  var threshold = 20;
  
  var fft_real = fft1dreal(noisy_sig);
  var fft_img = fft1dimg(noisy_sig);
  var sq_abs = square(fft_real) + square(fft_img);
  var magnitudes = sqrt(sq_abs);
  var GetThresholdReal = thresholdUp( magnitudes , threshold,1);
  var final1 = getElemAtIndx(GetThresholdReal , [2]); 
  print(final1);
}