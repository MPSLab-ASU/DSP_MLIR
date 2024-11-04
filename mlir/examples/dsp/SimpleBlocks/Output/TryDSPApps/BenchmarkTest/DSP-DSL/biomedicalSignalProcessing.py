def main() {
    var fc1 = 1000;
    var fc2 = 7500;
    var Fs = 8000;
    var N = 101;
    var distance = 950;
	var input = getRangeOfVector(0, 20000, 0.000125);
    # print(c);
    var pi = 3.14159265359;
    var f_sig = 500;
    var getMultiplier = 2 * pi * f_sig;
    # print(getMultiplier);
    var getSinDuration = gain(input, getMultiplier);
    # print(getSinDuration);
    var clean_sig = sin(getSinDuration );

    var f_noise = 3000;
    var getNoiseSinDuration = gain(input, 2 * pi * f_noise);
    var noise = sin(getNoiseSinDuration);
    var noise1 = gain(noise, 0.5);

    var noisy_sig = clean_sig + noise1;
    # Step 1: FIR Bandpass Filter
    var wc1 = 2 * pi * fc1 / Fs; #wc should vary from 0 to pi
    var lpf1 = lowPassFIRFilter(wc1, N); #ideal low -pass filter
    var lpf1_w = lpf1 * hamming(N);

    var wc2 = 2 * pi * fc2 / Fs;
    var lpf2 = lowPassFIRFilter(wc2, N);
    var lpf2_w = lpf2 * hamming(N);

    # var bpf = lpf2 - lpf;
    var bpf_w = sub(lpf2_w,lpf1_w);
    var FIRfilterResponseForBpf = FIRFilterResponse(noisy_sig, bpf_w);

    # Step 2: Artifact Removal (R-peak detection)
    var max_signal = max(FIRfilterResponseForBpf);

    var height = 0.3 * max_signal;

    var r_peaks = find_peaks(FIRfilterResponseForBpf, height, distance);

    var len_r_peaks = len(r_peaks);
    var last_peaks_index = sub(len_r_peaks, [1]);
    var peaks_count = getSingleElemAtIndx(r_peaks, last_peaks_index);

    var diff_val = diff(r_peaks, peaks_count);
    var peaks_count_minus_one = sub(peaks_count, 1);
    var diff_mean = mean(diff_val, peaks_count_minus_one);

    var avg_hr = (60 * Fs) / diff_mean;

    print(avg_hr);

}

