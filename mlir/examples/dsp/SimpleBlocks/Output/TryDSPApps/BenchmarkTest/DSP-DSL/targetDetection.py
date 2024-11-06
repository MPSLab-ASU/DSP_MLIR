def main() {
  var fs = 1000;
  # var step = 1/fs; 
  # print(step);
	var input = getRangeOfVector(0, 100000000, 0.000125);
  var pi = 3.14159265359;
  var getMultiplier = 2 * pi * 10;
  # print(getMultiplier);
  var getSinDuration = gain(input, getMultiplier);
  var sig1 = sin(getSinDuration );
  var getMultiplier2 = 2 * pi * 20;
  var getSinDuration2 = gain(input, getMultiplier2);
  var sinsig2 = sin(getSinDuration2);
  var sig2 = gain(sinsig2, 0.5);
  var signal = sig1 + sig2;
  var noise = delay(signal, 5);
  var noisy_sig = signal + noise;
  
  var mu = 0.01;
  var filterSize = 20;
  var y = lmsFilterResponse(noisy_sig, signal, mu, filterSize);
  var peaks = find_peaks(y, 1, 50); 
  var final1 = getElemAtIndx(peaks , [1]); 
  var final2 = getElemAtIndx(peaks , [2]); 
  print(final1);
  print(final2);
}