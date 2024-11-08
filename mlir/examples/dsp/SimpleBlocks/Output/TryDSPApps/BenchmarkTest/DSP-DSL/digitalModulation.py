def main() {
        # var input = [1,0,1,1,0,1,0,0];
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
        var modulate_symbol_real = qam_modulate_real(binary_sig);
        # print(modulate_symbol_real);
        var modulate_symbol_imagine = qam_modulate_imagine(binary_sig);
        # print(modulate_symbol_imagine);
        var decode_data = qam_demodulate(modulate_symbol_real, modulate_symbol_imagine);
        print(decode_data);
}

