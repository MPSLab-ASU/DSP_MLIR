#include <stdio.h>
#include <math.h>

// Define INPUT_LENGTH globally
#define INPUT_LENGTH 50000

void getRangeOfVector(double* vector, double start, int length, double increment) {
    for (int i = 0; i < length; i++) {
        vector[i] = start + i * increment;
    }
}

void reverseInput(double* output, double* input, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input[length - 1 - i];
    }
}

void FIRFilterResponse(double* output, double* input, double* filter, int length) {
    for (int n = 0; n < length; n++) {
        output[n] = 0;
        for (int k = 0; k < length; k++) {
            if (n - k >= 0) {
                output[n] += input[n - k] * filter[k];
            }
        }
    }
}

void dftReal(double* real, double* input, int length) {
    for (int k = 0; k < length; k++) {
        real[k] = 0;
        for (int n = 0; n < length; n++) {
            double angle = 2 * M_PI * k * n / length;
            real[k] += input[n] * cos(angle);
        }
    }
}

void dftImag(double* imag, double* input, int length) {
    for (int k = 0; k < length; k++) {
        imag[k] = 0;
        for (int n = 0; n < length; n++) {
            double angle = 2 * M_PI * k * n / length;
            imag[k] -= input[n] * sin(angle);
        }
    }
}

void squareMagnitude(double* output, double* real, double* imag, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = real[i] * real[i] + imag[i] * imag[i];
    }
}

int main() {
    // Use INPUT_LENGTH instead of hard-coded value
    double input[INPUT_LENGTH];
    getRangeOfVector(input, 0.0, INPUT_LENGTH, 1.0);

    double reverse_input[INPUT_LENGTH];
    reverseInput(reverse_input, input, INPUT_LENGTH);

    double conv1d[INPUT_LENGTH];
    FIRFilterResponse(conv1d, input, reverse_input, INPUT_LENGTH);

    double fft_real[INPUT_LENGTH];
    double fft_img[INPUT_LENGTH];
    dftReal(fft_real, conv1d, INPUT_LENGTH);
    dftImag(fft_img, conv1d, INPUT_LENGTH);

    double sq[INPUT_LENGTH];
    squareMagnitude(sq, fft_real, fft_img, INPUT_LENGTH);

    for (int i = 0; i < INPUT_LENGTH; i++) {
        printf("%f\n", sq[i]);
    }

    return 0;
}