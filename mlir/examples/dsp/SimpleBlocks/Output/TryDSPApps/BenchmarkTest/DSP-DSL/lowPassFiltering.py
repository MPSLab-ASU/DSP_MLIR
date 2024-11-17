
#fir filter design
#define hid = sin(wc *n)/ (pi *n) wc = cut off frequency , wc/pi at n=0

def main() {

  #define a sin signal with freq, f_sig = 500 
  #if sampling_freq is fs then duration = 2*pi*f_sig * c 
    # where c=(0 to 50 millisecs ) 
    # => now, 50 ms = N / fs => N = 50 s * fs /1000 = 50 * 8000 / 1000 = 400
  var fs = 8000;
  # var step = 1/8000; 
  # print(step);
  var duration = 0.05 ; # 50 milli-secs
	var input = getRangeOfVector(0, 100000000, 0.000125);
  # print(c);
  # var c = getRangeOfVector(0,10, 0.000125);
  var f_sig = 500;
  var pi = 3.14159265359;
  var getMultiplier = 2 * pi * f_sig;
  # print(getMultiplier);
  var getSinDuration = gain(input, getMultiplier);
  # print(getSinDuration);
  var clean_sig = sin(getSinDuration );
  # print(clean_sig);
  # Here, sampling freq = fs & step-size = 1/fs

  #define a noise signal with freq = 3000
  var f_noise = 3000;
  var getNoiseSinDuration = gain(input, 2 * pi * f_noise);
  var noise = sin(getNoiseSinDuration);
  var noise1 = gain(noise, 0.5);
  # print(noise1);

  # var f_noise2 = 2500;
  # var getNoiseSinDuration2 = gain(input, 2 * pi * f_noise2);
  # var noise11 = sin(getNoiseSinDuration2);
  # var noise2 = gain(noise11, 0.5);

  #noisy_sig = clean + noise
  # var noisy_sig = clean_sig + noise1 + noise2;
  var noisy_sig = clean_sig + noise1;
  # print(noisy_sig);

  #design a low-pass filter : filterOrder = 51(odd) , cut-off freq=1000
  # get wc = 2 * pi * cutoff_freq / fs
  # get the filter response using filter(b,a, noisy_sig)
  var fc = 1000;
  # var Fs = 8000;
  var wc = 2 * pi * fc / fs; #wc should vary from 0 to pi
  var N = 101 ;
  # var hid = sinc(wc, N);
  var lpf = lowPassFIRFilter(wc, N); #ideal low -pass filter
  var lpf_w = lpf * hamming(N);

  # var fc2 = 1200;
  # # var Fs = 8000;
  # var wc2 = 2 * pi * fc2 / fs; #wc should vary from 0 to pi
  # # var hid = sinc(wc, N);
  # var lpf2 = lowPassFIRFilter(wc2, N); #ideal low -pass filter
  # var lpf_w2 = lpf2 * hamming(N);
  # print(lpf_w);

  # filter response
  # var a5 = [1,  0 ,0 ,0 ,0  ];
  # var a51 = [1,  0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0 ,0  ];

  # var filterRes = filter(lpf_w , a51, noisy_sig);
  var FIRfilterResponse = FIRFilterResponse(noisy_sig, lpf_w);
  # var FIRfilterResponse = FIRFilterResSymmOptimized(noisy_sig, lpf_w);
  var final1 = getElemAtIndx(FIRfilterResponse , [6]); 
  print(final1);
  
  # var FIRfilterResponse2 = FIRFilterResponse(noisy_sig, lpf_w2);
  # var final2 = getElemAtIndx(FIRfilterResponse2 , [7]); 
  # print(final2);
  
  # print(FIRfilterResponse);
}

