def main() {
        # var time = [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.];
        var time = [0.0, 0.25, 0.5, 0.75, 1.0];
        var antennas = 4;
        var freq = 5;
        var weights = [1,2,3,4];

        var signal = beam_form(antennas, freq, time, weights);
        print(signal);
}
