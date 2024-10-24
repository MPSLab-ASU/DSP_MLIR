
#fir filter design
#This is low pass FIR filter design
def main() {

  #size 10
  # var a10 = [ 10,20,30,40,50,60,70,80,90,100];
  # var a10 = getRangeOfVector(0, 400, 0.000125);
  # var orig = sin(a10);

    var N = 11 ;

  # for cut-off freq
  var pi = 3.14159265359;
  var fc1 = 500;
  var fc2 = 600;
  var fc3 = 1000;
  var fc4 = 1200;
  var Fs = 8000;
  var wc1 = 2 * pi * fc1 / Fs; #wc should vary from 0 to pi
  var wc2 = 2 * pi * fc2 / Fs;
  var wc3 = 2 * pi * fc3 / Fs;
  var wc4 = 2 * pi * fc4 / Fs;
  # get lowPassFilter for wc coeff as well as using Symmetrical Optimized 
  # calculation 
  # var lpf = lowPassFIRFilter(wc, N); #ideal low -pass filter
  # var lpf_w = lpf * hamming(N);
  # var lpf_w2 = FIRFilterHammingOptimized(wc, N);
  var hpf = highPassFIRFilter(wc1, N); #ideal high-pass filter
  var hpf_w = hpf * hamming(N);

  var hpf2 = highPassFIRFilter(wc2, N); #ideal high-pass filter
  var hpf_w2 = hpf2 * hamming(N);

  var hpf3 = highPassFIRFilter(wc3, N); #ideal high-pass filter
  var hpf_w3 = hpf3 * hamming(N);

  var hpf4 = highPassFIRFilter(wc4, N); #ideal high-pass filter
  var hpf_w4 = hpf4 * hamming(N);
  # var hpf_w2 = highPassFIRHammingOptimized(wc, N); 
  # print(lpf_w2);
  var final1 = getElemAtIndx(hpf_w , [6]);
  var final2 = getElemAtIndx(hpf_w2 , [7]);
  var final3 = getElemAtIndx(hpf_w3 , [8]);
  # var final4 = getElemAtIndx(hpf_w4 , [500]); 
  # var final4 = getElemAtIndx(hpf_w4 , [5]);
  print(final1);
  print(final2);
  print(final3);
  # print(final4);
  # print(hpf_w);


}

