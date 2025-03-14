#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h>

#define PI 3.14159265359
#define INPUT_LENGTH 100

void getRangeOfVector(double* vector, double start, int length, double increment) {
    for (int i = 0; i < length; i++) {
        vector[i] = start + i * increment;
    }
}

void gain(double* output, const double* input, double gainFactor, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input[i] * gainFactor;
    }
}

void thresholdUp(double* output, const double* input, double threshold, double low_value) {
    for (int i = 0; i < INPUT_LENGTH; i++) {
        output[i] = (input[i] >= threshold) ? 1.0 : low_value;
    }
}

void qam_modulate_real(double* symbols_real, double* binary_sig) {
    for (int i = 0; i < INPUT_LENGTH; i += 2) {
        double bit1 = binary_sig[i];
        double bit2 = binary_sig[i + 1];

        if (bit1 == 0.0 && bit2 == 0.0)
            symbols_real[i / 2] = -1.0;
        else if (bit1 == 0.0 && bit2 == 1.0)
            symbols_real[i / 2] = -1.0;
        else if (bit1 == 1.0 && bit2 == 0.0)
            symbols_real[i / 2] = 1.0;
        else if (bit1 == 1.0 && bit2 == 1.0)
            symbols_real[i / 2] = 1.0;
    }
}

void qam_modulate_imag(double* symbols_imag, double* binary_sig) {
    for (int i = 0; i < INPUT_LENGTH; i += 2) {
        double bit1 = binary_sig[i];
        double bit2 = binary_sig[i + 1];

        if (bit1 == 0.0 && bit2 == 0.0)
            symbols_imag[i / 2] = -1.0;
        else if (bit1 == 0.0 && bit2 == 1.0)
            symbols_imag[i / 2] = 1.0;
        else if (bit1 == 1.0 && bit2 == 0.0)
            symbols_imag[i / 2] = -1.0;
        else if (bit1 == 1.0 && bit2 == 1.0)
            symbols_imag[i / 2] = 1.0;
    }
}

void qam_demodulate(double* decoded_data, double* symbols_real, double* symbols_imag) {
    for (int i = 0; i < INPUT_LENGTH / 2; i++) {
        double real = symbols_real[i];
        double imag = symbols_imag[i];

        if (real == -1.0 && imag == -1.0) {
            decoded_data[2 * i] = 0.0;
            decoded_data[2 * i + 1] = 0.0;
        } else if (real == -1.0 && imag == 1.0) {
            decoded_data[2 * i] = 0.0;
            decoded_data[2 * i + 1] = 1.0;
        } else if (real == 1.0 && imag == -1.0) {
            decoded_data[2 * i] = 1.0;
            decoded_data[2 * i + 1] = 0.0;
        } else if (real == 1.0 && imag == 1.0) {
            decoded_data[2 * i] = 1.0;
            decoded_data[2 * i + 1] = 1.0;
        }
    }
}

int main() {
    srand(time(NULL)); // Seed random number generator

    // Step 1: Generate Input Signal
    double* input = (double*)malloc(sizeof(double) * INPUT_LENGTH);
    getRangeOfVector(input, 0, INPUT_LENGTH, 0.000125);

    // Step 2: Generate clean signal
    double f_sig = 500;
    double getMultiplier = 2 * PI * f_sig;
    double* getSinDuration = (double*)malloc(sizeof(double) * INPUT_LENGTH);
    gain(getSinDuration, input, getMultiplier, INPUT_LENGTH);

    double* clean_sig = (double*)malloc(sizeof(double) * INPUT_LENGTH);
    for (int i = 0; i < INPUT_LENGTH; i++) {
        clean_sig[i] = sin(getSinDuration[i]);
    }

    // Step 3: Apply thresholdUp to get binary signal (stored in double array)
    double* binary_sig = (double*)malloc(sizeof(double) * INPUT_LENGTH);
    thresholdUp(binary_sig, clean_sig, 0.4, 0.0);

    // Step 4: Perform QAM modulation
    double* modulate_symbol_real = (double*)malloc(sizeof(double) * (INPUT_LENGTH / 2));
    double* modulate_symbol_imag = (double*)malloc(sizeof(double) * (INPUT_LENGTH / 2));

    qam_modulate_real(modulate_symbol_real, binary_sig);
    qam_modulate_imag(modulate_symbol_imag, binary_sig);


    // Step 5: Perform QAM Demodulation
    double* decode_data = (double*)malloc(sizeof(double) * INPUT_LENGTH);
    qam_demodulate(decode_data, modulate_symbol_real, modulate_symbol_imag);

    
    printf("%f ", decode_data[2]); 


    // Free allocated memory
    free(input);
    free(getSinDuration);
    free(clean_sig);
    free(binary_sig);
    free(modulate_symbol_real);
    free(modulate_symbol_imag);
    free(decode_data);

    return 0;
}
