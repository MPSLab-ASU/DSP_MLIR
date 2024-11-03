def main() {
        var input = [1,0,1,1,0,1,0,0];
        var modulate_symbol_real = qam_modulate_real(input);
        # print(modulate_symbol_real);
        var modulate_symbol_imagine = qam_modulate_imagine(input);
        # print(modulate_symbol_imagine);
        var decode_data = qam_demodulate(modulate_symbol_real, modulate_symbol_imagine);
        print(decode_data);
}

