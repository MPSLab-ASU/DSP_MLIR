def main() {
var fs = 8000;
  # var step = 1/8000; 
  # print(step);
  var t = getRangeOfVector(0,100, 0.000125);
  var f_sig = 500;
  var pi = 3.14159265359;
  var getMultiplier = 2 * pi * f_sig;
  # print(getMultiplier);
  var getSinDuration = gain(t, getMultiplier);
  # print(getSinDuration);
  var clean_sig = sin(getSinDuration );

  #define a noise signal with freq = 3000
  var f_noise = 3000;
  var getNoiseSinDuration = gain(t, 2 * pi * f_noise);
  var noise = sin(getNoiseSinDuration);
  var noise1 = gain(noise, 0.5);

  var noisy_sig = clean_sig + noise1;
  # print(noisy_sig);
  # print(clean_sig);
  var mu = 0.01;
  var filterSize = 32;
  var y = lmsFilterResponse(noisy_sig, clean_sig, mu, filterSize);
  var G1 = 10;
  var sol = gain(y,G1);
  # print(y);
  print(sol);

}

