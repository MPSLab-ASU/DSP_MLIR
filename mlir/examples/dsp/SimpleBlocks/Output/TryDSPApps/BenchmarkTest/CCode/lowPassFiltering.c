#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define PI 3.14159265359
#define FS 8000
#define N 101
#define INPUT_LENGTH 100
#define FILTER_LENGTH 200

void generate_signal(double *signal, double freq, int length) {
    for (int i = 0; i < length; i++) {
        signal[i] = sin(2 * PI * freq * i / FS);
    }
}

void generate_lowpass_filter(double *filter, double cutoff_freq) {
    double wc = 2 * PI * cutoff_freq / FS;
    for (int i = 0; i < N; i++) {
        int n = i - (N / 2);
        if (n == 0) {
            filter[i] = wc / PI;
        } else {
            filter[i] = sin(wc * n) / (PI * n);
        }
        // Apply Hamming window
        filter[i] *= (0.54 - 0.46 * cos(2 * PI * i / (N - 1)));
    }
}

void apply_fir_filter(double *input, double *output, double *filter) {
    for (int i = 0; i < FILTER_LENGTH; i++) {
        double sum = 0.0;
        for (int j = 0; j < N; j++) {
            if (i - j >= 0 && i - j < INPUT_LENGTH) {
                sum += input[i - j] * filter[j];
            }
        }
        output[i] = sum;
    }
}

int main() {
    double clean_signal[INPUT_LENGTH];
    double noise_signal[INPUT_LENGTH];
    double noisy_signal[INPUT_LENGTH];
    double fir_filter[N];
    double filtered_signal[FILTER_LENGTH] = {0};

    // Generate clean signal with frequency 500Hz
    generate_signal(clean_signal, 500, INPUT_LENGTH);
    
    // Generate noise signal with frequency 3000Hz and scale it
    generate_signal(noise_signal, 3000, INPUT_LENGTH);
    for (int i = 0; i < INPUT_LENGTH; i++) {
        noise_signal[i] *= 0.5;
        noisy_signal[i] = clean_signal[i] + noise_signal[i];
    }

    // Design low-pass filter with cutoff frequency 1000Hz
    generate_lowpass_filter(fir_filter, 1000);

    // Apply FIR filter
    apply_fir_filter(noisy_signal, filtered_signal, fir_filter);

 
    printf("%f\n", filtered_signal[6]);


    return 0;
}
