def main() {
  var fs = 1000;
  # var step = 1/fs; 
  # print(step);
	var input = getRangeOfVector(0, 100000000, 0.000125);
  var pi = 3.14159265359;
  var getMultiplier = 2 * pi * 5;
  # print(getMultiplier);
  var getSinDuration = gain(input, getMultiplier);
  var signal = sin(getSinDuration );

  var noise = delay(signal, 5);
  var noisy_sig = signal + noise;


 #design a low-pass filter : filterOrder = 5(odd) , cut-off freq=10
  # get wc = 2 * pi * cutoff_freq / fs
  # get the filter response using filter(b,a, noisy_sig)
  var fc = 1000;
  # var Fs = 8000;
  var wc = 2 * pi * 1000 / 500; #wc should vary from 0 to pi
  var N = 5;
  # var hid = sinc(wc, N);
  var lpf = lowPassFIRFilter(wc, 1); #ideal low -pass filter
  var lpf_w = lpf * hamming(N);
  var FIRfilterResponse = FIRFilterResponse(noisy_sig, lpf_w);
 
  var threshold = 0.5;
  var GetThresholdReal = thresholdUp(FIRfilterResponse, threshold, 0);
  # print(GetThresholdReal);
  var final1 = getElemAtIndx(GetThresholdReal , [3]); 
  print(final1);

}