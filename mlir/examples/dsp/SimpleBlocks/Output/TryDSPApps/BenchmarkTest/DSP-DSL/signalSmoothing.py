def main() {
  var fs = 8000;
	var input = getRangeOfVector(0, 100000000, 0.000125);
  var f_sig = 500;
  var pi = 3.14159265359;
  var getMultiplier = 2 * pi * f_sig;
  var getSinDuration = gain(input, getMultiplier);
  var clean_sig = sin(getSinDuration );
  var f_noise = 3000;
  var getNoiseSinDuration = gain(input, 2 * pi * f_noise);
  var noise = sin(getNoiseSinDuration);
  var noise1 = gain(noise, 0.5);

  var noisy_sig = clean_sig + noise1;
  var median = medianFilter(noisy_sig);
  var average = slidingWindowAvg(median);
#   print(average);
  var final1 = getElemAtIndx(average , [1]); 
  print(final1);
}