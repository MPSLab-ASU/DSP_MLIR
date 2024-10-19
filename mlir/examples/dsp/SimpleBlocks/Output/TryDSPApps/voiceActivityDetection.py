def main() {
  var fs = 1000;
  # var step = 1/fs; 
  # print(step);
  var t = getRangeOfVector(0,2000,0.001);
  var pi = 3.14159265359;
  var getMultiplier = 2 * pi * 5;
  # print(getMultiplier);
  var getSinDuration = gain(t, getMultiplier);
  var signal = sin(getSinDuration );

  var noise = delay(signal, 5);
  var noisy_sig = signal + noise;
  var threshold = 1.8;
  var GetThresholdReal = threshold( noisy_sig , threshold);
  var zcr = zeroCrossCount(GetThresholdReal);
  print(GetThresholdReal);
  print(zcr);
}