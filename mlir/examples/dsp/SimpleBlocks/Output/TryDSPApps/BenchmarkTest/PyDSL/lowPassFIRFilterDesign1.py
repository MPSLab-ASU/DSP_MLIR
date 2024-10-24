
#fir filter design
#This is low pass FIR filter design
def main() {
  var N = 51 ;

  # for cut-off freq
  var pi = 3.14159265359;
  var fc1 = 500;
  var fc2 = 600;
  var Fs = 8000;
  var wc1 = 2 * pi * fc1 / Fs; #wc should vary from 0 to pi

  # get lowPassFilter for wc coeff as well as using Symmetrical Optimized 
  var lpf = lowPassFIRFilter(wc1, N); #ideal low-pass filter
  var lpf_w = lpf * hamming(N);

  # var lpf2 = lowPassFIRFilter(wc2, N); #ideal low-pass filter
  # var lpf_w2 = lpf2 * hamming(N);

  var final1 = getElemAtIndx(lpf_w , [6]);
  # var final2 = getElemAtIndx(lpf_w2 , [7]);

  print(final1);
  # print(final2);

}

