def main() {
     var digit = 8; 
     var duration = 0.5;  
     var fs = 8192; 
     var d = 1/fs;
     var N = fs * duration;
     var dtmf_tone = generateDtmf(digit, duration, fs);
     var fft_real = fft1dreal(dtmf_tone);
     var fft_imag = fft1dimg(dtmf_tone); 
     var squared_fft_real = square(fft_real);
     var squared_fft_imag = square(fft_imag);
     var sum = squared_fft_real + squared_fft_imag;
     # print(sum);
     var magnitudes = sqrt(sum);
    #  print(magnitudes);
     var frequencies = fftfreq(4096, 0.000122);
    #  # print(frequencies);
     var peaks = findDominantPeaks(frequencies, magnitudes);
     print(peaks);
     var freqPairs = [
     [941, 1336],
     [697, 1209],
     [697, 1336],
     [697, 1477], 
     [770, 1209],
     [770, 1336],
     [770, 1477],
     [852, 1209],
     [852, 1336],
     [852, 1477]];
     var recovered_digit = recoverDtmfDigit(peaks, freqPairs);
     print(recovered_digit);
 }