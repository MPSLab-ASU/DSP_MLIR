def main() {
        # var input = "HELLO FROM SPACE";
	var input = getRangeOfVector(0, 40000, 0.000125);
        # print(c);
        var pi = 3.14159265359;
        var f_sig = 500;
        var getMultiplier = 2 * pi * f_sig;
        # print(getMultiplier);
        var getSinDuration = gain(input, getMultiplier);
        # print(getSinDuration);
        var clean_sig = sin(getSinDuration );
        var binary_sig = thresholdUp(clean_sig, 0.4,0);
        var a = space_modulate(binary_sig);
        var noise = sin(a);
        var noisy_signal = a+noise;
        var b = space_demodulate(noisy_signal);
        var e = space_err_correction(b);
        var final = getElemAtIndx(e, [8]);
        print(final);
}