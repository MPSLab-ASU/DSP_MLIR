def main() {
        # var input = "HELLO FROM SPACE";
	var input = getRangeOfVector(0, 100000000, 1);
        # print(input);
        var binary_sig = thresholdUp(input,50,0);
        var a = space_modulate(binary_sig);
        var noise = sin(a);
        var noisy_signal = a+noise;
        var b = space_demodulate(noisy_signal);
        var e = space_err_correction(b);
        var final = getElemAtIndx(e, [8]);
        print(final);
}