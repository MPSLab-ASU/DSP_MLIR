#Low-pass filter for noisy signal
def main() {
  var fs = 8000;
  #Define a arithmetic series of getRangeOfVector(start, NoOfElements , Increment)  
	var input = getRangeOfVector(0, 30, 0.000125);
  #define a noise signal with freq = 500
  var f_sig = 500;
  var pi = 3.14159265359;
  var getMultiplier = 2 * pi * f_sig;
  # Multiply each element of input vector with gain -- getMultiplier
  var getSinDuration = gain(input, getMultiplier);
  var clean_sig = sin(getSinDuration );
  #define a noise signal with freq = 3000
  var f_noise = 3000;
  var getNoiseSinDuration = gain(input, 2 * pi * f_noise);
  var noise = sin(getNoiseSinDuration);
  #elementWise addition 
  var noisy_sig = clean_sig + gain(noise, 0.5);
  #design a low-pass filter : filterOrder = 51(odd) , cut-off freq=1000
  var fc = 1000;
  var wc = 2 * pi * fc / fs; #wc should vary from 0 to pi
  var N = 101 ;
  # y_lpf[n] = wc/pi * sinc(wc * (n- (N-1)/2)) , n!= (N-1)/2 : 
            #  = wc/pi , n = (N-1)/2
  var lpf = lowPassFIRFilter(wc, N); #ideal low -pass filter
  #Element-Wise Multiplication 
  var lpf_w = lpf * hamming(N);
  #Conv1d -- function 
  var FIRfilterResponse = FIRFilterResponse(noisy_sig, lpf_w); 
  print(FIRfilterResponse);
}

