def main() {
  # GENERATE SIGNAL FOR '5'
  var fs = 8000;
  var duration = 0.5;
  var f1 = 770;
  var f2 = 1336;
#   # var step = 1/fs; 
#   # print(step);
#   # total instances = fs * duration
#   var total_instances = fs * duration;
#   var t = getRangeOfVector(0,4000,0.000125);
#   var pi = 3.14159265359;
#   var getMultiplier = 2 * pi * f1;
#   var getSinDuration = gain(t, getMultiplier);
#   var sig1 = sin(getSinDuration);
 

#   var getMultiplier2 = 2 * pi * f2;
#   var getSinDuration2 = gain(t, getMultiplier2);
#   var sig2 = sin(getSinDuration2);
#   var signal = sig1 + sig2;
#   var finalsig = gain(signal, 0.5);
  


#   var noise = delay(signal, 5);
#   var noisy_sig = signal + noise;
#   var threshold = 4;
  
#   var fft_real = fft1dreal(noisy_sig);
#   var fft_img = fft1dimg(noisy_sig);

#   var magnitude = square(fft_real) + square(fft_img);
# print(magnitude);  
#   # res = gain(sum , 1/N)
#   var len1 = len(t);
#   # var res = sum1 / len1;
#   # print(sq_abs);
#   var GetThresholdReal = threshold( magnitude , threshold);
var dtmf_sig = generateDtmf(5,duration,fs);
print(dtmf_sig);

}