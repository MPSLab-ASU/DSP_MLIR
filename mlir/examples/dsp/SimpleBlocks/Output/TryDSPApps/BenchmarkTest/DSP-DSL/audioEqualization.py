
#fir filter design
#define hid = sin(wc *n)/ (pi *n) wc = cut off frequency , wc/pi at n=0

def main() {

  # var input = [1,2,3,4,5];
	var input = getRangeOfVector(0, 30000000, 1);
  var pi = 3.14159265359;
  var fc = 300;
  var Fs = 8000;
  var gainForBass = 2;
  var gainForMid = 1.5;
  var gainForTreble = 0.8;

  var wc = 2 * pi * fc / Fs; #wc should vary from 0 to pi
  var N = 101 ;
  var lpf = lowPassFIRFilter(wc, N); #ideal low -pass filter
  var lpf_w = lpf * hamming(N);  
  var FIRfilterResponseForLpf = FIRFilterResponse(input, lpf_w);
  var gainWithLpf = gain(FIRfilterResponseForLpf , gainForBass);

  #For high-pass filter 
  var fc2 = 1500;
  var wc2 = 2 * pi * fc2 / Fs;
  var hpf = highPassFIRFilter(wc2, N); #ideal high -pass filter
  var hpf_w = hpf * hamming(N);  
  var FIRfilterResponseForHpf = FIRFilterResponse(input, hpf_w);
  var gainWithHpf = gain(FIRfilterResponseForHpf , gainForTreble);

  #Band-pass filter
  var lpf2 = lowPassFIRFilter(wc2, N);
  var lpf2_w = lpf2 * hamming(N);
  # var bpf = lpf2 - lpf;
  var bpf_w = sub(lpf2_w,lpf_w);
  var FIRfilterResponseForBpf = FIRFilterResponse(input, bpf_w);
  var gainWithBpf = gain(FIRfilterResponseForBpf , gainForTreble);
  var final_audio = gainWithLpf + gainWithHpf + gainWithBpf ;
  var final1 = getElemAtIndx(final_audio , [3]); 
  print(final1);
  print(final_audio);
}

  

  

