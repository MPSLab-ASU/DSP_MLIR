#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <complex.h>

#define INPUT_LENGTH 40000
#define M_PI 3.14159265358979323846

double* getRange(double start, int noOfSamples, double increment) {
    double* output = malloc(noOfSamples * sizeof(double));
    if (!output) {
        perror("Memory allocation failed in getRange");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < noOfSamples; i++) {
        output[i] = start + i * increment;
    }

    return output;
}

void dft(double complex* output, const double* input, int length) {
    for (int k = 0; k < length; k++) {
        output[k] = 0;
        for (int n = 0; n < length; n++) {
            double angle = 2 * M_PI * k * n / length;
            output[k] += input[n] * cexp(-I * angle);
        }
    }
}

void square(double* output, const double* input, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input[i] * input[i];
    }
}

double sum(const double* input, int length) {
    double total = 0;
    for (int i = 0; i < length; i++) {
        total += input[i];
    }
    return total;
}

int main() {
    double* input = getRange(0, INPUT_LENGTH, 1);

    double complex* fft = malloc(INPUT_LENGTH * sizeof(double complex));
    if (!fft) {
        perror("Memory allocation failed");
        free(input);
        return EXIT_FAILURE;
    }

    dft(fft, input, INPUT_LENGTH);

    double* sq_abs = malloc(INPUT_LENGTH * sizeof(double));
    if (!sq_abs) {
        perror("Memory allocation failed");
        free(input);
        free(fft);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < INPUT_LENGTH; i++) {
        sq_abs[i] = creal(fft[i]) * creal(fft[i]) + cimag(fft[i]) * cimag(fft[i]);
    }

    double sum_result = sum(sq_abs, INPUT_LENGTH);
    double res = sum_result / INPUT_LENGTH;

    printf("%f\n", res);

    free(input);
    free(fft);
    free(sq_abs);

    return 0;
}