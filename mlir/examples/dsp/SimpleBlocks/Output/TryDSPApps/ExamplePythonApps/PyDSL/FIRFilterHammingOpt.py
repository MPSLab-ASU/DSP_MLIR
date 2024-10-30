
#fir filter design
#define hid = sin(wc *n)/ (pi *n) wc = cut off frequency , wc/pi at n=0
# h_lp[n] = hid[n-(N-1)/2] * w[n] //w[n] = window like hamming window
# h_lp_w[n] = h_lp[n] * w[n]


def main() {

  #size 10
  # var a10 = [ 10,20,30,40,50,60,70,80,90,100];
  # var a10 = getRangeOfVector(0, 400, 0.000125);
  # var orig = sin(a10);

  var N = 7;

  # for cut-off freq
  var pi = 3.14159265359;
  var fc = 1000;
  var Fs = 8000;
  var wc = 2 * pi * fc / Fs; #wc should vary from 0 to pi

  # get lowPassFilter for wc coeff as well as using Symmetrical Optimized 
  # calculation 
  # var lpf = lowPassFIRFilter(wc, N); #ideal low -pass filter
  # var lpf_w = lpf * hamming(N);
  # var lpf_w2 = FIRFilterHammingOptimized(wc, N);
  var hpf = highPassFIRFilter(wc, N); #ideal high-pass filter
  var hpf_w = hpf * hamming(N);
  # var hpf_w2 = highPassFIRHammingOptimized(wc, N); 
  # var 
  # print(hamming(5));
  # print(lpf_w2);
  # print(lpf_w);
  print(hpf_w);
  # print(hpf_w2);
  # print(hpf);
  # print(hid);

}

