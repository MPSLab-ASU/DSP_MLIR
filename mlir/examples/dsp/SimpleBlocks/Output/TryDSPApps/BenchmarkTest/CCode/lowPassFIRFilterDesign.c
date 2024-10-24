#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define INPUT_LENGTH 100000001
#define PI M_PI 
#define FC1 500
#define FS 8000

double* hamming(int length) {
    double* window = malloc(length * sizeof(double));
    if (!window) {
        perror("Memory allocation failed in hamming");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < length; i++) {
        window[i] = 0.54 - 0.46 * cos(2 * PI * i / (length - 1));
    }
    return window;
}

double* lowPassFIRFilter(double wc, int length) {
    double* filter = malloc(length * sizeof(double));
    if (!filter) {
        perror("Memory allocation failed in lowPassFIRFilter");
        exit(EXIT_FAILURE);
    }

    int mid = (length - 1) / 2;
    for (int n = 0; n < length; n++) {
        if (n == mid) {
            filter[n] = wc / PI;
        } else {
            filter[n] = sin(wc * (n - mid)) / (PI * (n - mid));
        }
    }
    return filter;
}

void elementWiseMultiplication(double* output, const double* array1, const double* array2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = array1[i] * array2[i];
    }
}

double getElemAtIndx(const double* array, int index) {
    return array[index];
}

int main() {
    double wc1 = 2 * PI * FC1 / FS;

    double* lpf = lowPassFIRFilter(wc1, INPUT_LENGTH);
    double* hamming_window = hamming(INPUT_LENGTH);

    double* lpf_w = malloc(INPUT_LENGTH * sizeof(double));
    if (!lpf_w) {
        perror("Memory allocation failed for lpf_w");
        free(lpf);
        free(hamming_window);
        exit(EXIT_FAILURE);
    }

    elementWiseMultiplication(lpf_w, lpf, hamming_window, INPUT_LENGTH);

    double final1 = getElemAtIndx(lpf_w, 6);

    printf("%f\n", final1);

    free(lpf);
    free(hamming_window);
    free(lpf_w);

    return 0;
}