#include <stdio.h>
#include <math.h>

// Define INPUT_LENGTH globally
#define INPUT_LENGTH 500
#define PI 3.14159265358


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
    int conv_length = 2 * length - 1;
    for (int n = 0; n < conv_length; n++) {
        output[n] = 0;
        for (int k = 0; k < length; k++) {
            if (n - k >= 0 && n - k < length) {
                output[n] += input[n - k] * filter[k];
            }
        }
    }
}

void dftReal(double* real, double* input, int length) {
    for (int k = 0; k < length; k++) {
        real[k] = 0;
        for (int n = 0; n < length; n++) {
            double angle = 2.0 * PI * k * n / length;
            real[k] += input[n] * cos(angle);
        }
    }
}

void dftImag(double* imag, double* input, int length) {
    for (int k = 0; k < length; k++) {
        imag[k] = 0;
        for (int n = 0; n < length; n++) {
            double angle = 2.0 * PI * k * n / length;
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
    double input[INPUT_LENGTH];
    getRangeOfVector(input, 0.0, INPUT_LENGTH, 1.0);

    double reverse_input[INPUT_LENGTH];
    reverseInput(reverse_input, input, INPUT_LENGTH);

    int conv_length = 2 * INPUT_LENGTH - 1;
    double conv1d[conv_length];
    FIRFilterResponse(conv1d, input, reverse_input, INPUT_LENGTH);

    double fft_real[conv_length];
    double fft_img[conv_length];
    dftReal(fft_real, conv1d, conv_length);
    dftImag(fft_img, conv1d, conv_length);

    double sq[conv_length];
    squareMagnitude(sq, fft_real, fft_img, conv_length);


    printf("%f\n", sq[2]);
    


    return 0;
}
