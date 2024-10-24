#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <complex.h>

#define INPUT_LENGTH 40000
#define NLEVELS 16
#define MIN 0.0
#define MAX 8.0
#define THRESHOLD_VAL 4.0

double* getRangeOfVector(double start, int noOfSamples, double increment) {
    double* output = malloc(noOfSamples * sizeof(double));
    if (!output) {
        perror("Memory allocation failed in getRangeOfVector");
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

void threshold(double* output, const double* input, double thresh, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = (fabs(input[i]) >= thresh) ? input[i] : 0;
    }
}

void quantization(double* output, const double* input, int nlevels, double max, double min, int length) {
    double step = (max - min) / nlevels;
    for (int i = 0; i < length; i++) {
        output[i] = round((input[i] - min) / step) * step + min;
    }
}

int* runLenEncoding(const double* input, int length, int* rleLength) {
    int* rle = malloc(2 * length * sizeof(int));
    if (!rle) {
        perror("Memory allocation failed");
        exit(EXIT_FAILURE);
    }

    int index = 0;
    int count = 1;
    for (int i = 1; i < length; i++) {
        if (input[i] != input[i - 1]) {
            rle[index++] = input[i - 1];
            rle[index++] = count;
            count = 1;
        } else {
            count++;
        }
    }
    rle[index++] = input[length - 1];
    rle[index++] = count;

    *rleLength = index;
    return rle;
}

double getElemAtIndx(const int* rle, int indx) {
    return rle[indx];
}

int main() {
    double* input = getRangeOfVector(0, INPUT_LENGTH, 1);

    double complex* fft = malloc(INPUT_LENGTH * sizeof(double complex));
    if (!fft) {
        perror("Memory allocation failed");
        free(input);
        return EXIT_FAILURE;
    }

    dft(fft, input, INPUT_LENGTH);

    double* GetThresholdReal = malloc(INPUT_LENGTH * sizeof(double));
    double* GetThresholdImg = malloc(INPUT_LENGTH * sizeof(double));
    if (!GetThresholdReal || !GetThresholdImg) {
        perror("Memory allocation failed");
        free(input);
        free(fft);
        free(GetThresholdReal);
        free(GetThresholdImg);
        return EXIT_FAILURE;
    }

    for (int i = 0; i < INPUT_LENGTH; i++) {
        GetThresholdReal[i] = creal(fft[i]);
        GetThresholdImg[i] = cimag(fft[i]);
    }

    threshold(GetThresholdReal, GetThresholdReal, THRESHOLD_VAL, INPUT_LENGTH);
    threshold(GetThresholdImg, GetThresholdImg, THRESHOLD_VAL, INPUT_LENGTH);

    double* QuantOutReal = malloc(INPUT_LENGTH * sizeof(double));
    double* QuantOutImg = malloc(INPUT_LENGTH * sizeof(double));
    if (!QuantOutReal || !QuantOutImg) {
        perror("Memory allocation failed");
        free(input);
        free(fft);
        free(GetThresholdReal);
        free(GetThresholdImg);
        free(QuantOutReal);
        free(QuantOutImg);
        return EXIT_FAILURE;
    }

    quantization(QuantOutReal, GetThresholdReal, NLEVELS, MAX, MIN, INPUT_LENGTH);
    quantization(QuantOutImg, GetThresholdImg, NLEVELS, MAX, MIN, INPUT_LENGTH);

    int rleLengthReal, rleLengthImg;
    int* rLEOutReal = runLenEncoding(QuantOutReal, INPUT_LENGTH, &rleLengthReal);
    int* rLEOutImg = runLenEncoding(QuantOutImg, INPUT_LENGTH, &rleLengthImg);

    double final1 = getElemAtIndx(rLEOutReal, 6);
    double final2 = getElemAtIndx(rLEOutImg, 7);
    printf("%f\n", final1);
    printf("%f\n", final2);

    free(input);
    free(fft);
    free(GetThresholdReal);
    free(GetThresholdImg);
    free(QuantOutReal);
    free(QuantOutImg);
    free(rLEOutReal);
    free(rLEOutImg);

    return 0;
}