def main() {

        var antennas = 4;
        var input_fc = 5;
        var N = 101;
	var input = getRangeOfVector(0, 10000, 0.000125);
        var weights = getRangeOfVector(-90, 180, 1);

        var signal = beam_form(antennas, input_fc, input, weights);
        var b1 = abs(signal);
        var power_profile = b1 * b1;
        var power_angle_max_idx = argmax(power_profile, 0);
        var power_angle_max_ele = argmax(power_profile,0);
        var pi = 3.1415926;
        var fc1 = 1000;
        var fc2 = 7500;
        var Fs = 8000;

        var wc1 = 2*pi*fc1 / Fs;
        var filter1 = lowPassFIRFilter(wc1, N);
        var filter_hamming_1 = filter1 * hamming(N);
        var wc2 = 2*pi*fc2 / Fs;
        var filter2 = highPassFIRFilter(wc2, N);
        var filter_hamming_2 = filter2 * hamming(N);

        var bpf = sub(filter_hamming_2, filter_hamming_1);
        var firFilterResponse = FIRFilterResponse(power_profile, bpf);
        var length = len(firFilterResponse);
        var final = getElemAtIndx(firFilterResponse , 10000);
        print(final);
}

