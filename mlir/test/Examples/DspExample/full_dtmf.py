def main() {
    var digit = 0; # digit whose dtmf tone is to be calculated
    var duration = 0.0625; # duration of the dtmf signal 
    var fs = 8192; # sampling frequency 
    var d = 1/fs;
    var N = fs * duration;
    var dtmf_tone = generateDtmf(digit, duration, fs); # generate the dtmf signal
    # print(dtmf_tone);
    var fft_real = fftReal(dtmf_tone); # take fft real
    var fft_imag = fftImag(dtmf_tone); # take fft imag
    # print(fft_real);
    # print(fft_imag);
    var squared_fft_real = square(fft_real);
    var squared_fft_imag = square(fft_imag);
    # print(squared_fft_real);
    # print(squared_fft_imag);
    var sum = squared_fft_real + squared_fft_imag;
    # print(sum);
    var magnitudes = sqrt(sum);
    # print(magnitudes);
    var frequencies = fftfreq(512, 0.000122);
    # print(frequencies);
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
