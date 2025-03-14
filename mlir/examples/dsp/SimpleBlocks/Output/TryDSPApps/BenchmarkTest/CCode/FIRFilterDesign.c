#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUT_LENGTH 101
#define PI M_PI
#define FS 8000
#define FC1 500
#define FC2 600
#define FC3 1000
#define FC4 1200

double *hamming(int length) {
    double *window = malloc(length * sizeof(double));
    if (!window) {
        perror("Memory allocation failed in hamming");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < length; i++) {
        window[i] = 0.54 - 0.46 * cos(2 * PI * i / (length - 1));
    }
    return window;
}

double *highPassFIRFilter(double wc, int length) {
    double *filter = malloc(length * sizeof(double));
    if (!filter) {
        perror("Memory allocation failed in highPassFIRFilter");
        exit(EXIT_FAILURE);
    }
    int mid = (length - 1) / 2;
    for (int n = 0; n < length; n++) {
        if (n == mid) {
            filter[n] = 1 - (wc / PI);
        } else {
            filter[n] = -sin(wc * (n - mid)) / (PI * (n - mid));
        }
    }
    return filter;
}

void elementWiseMultiplication(double *output, const double *array1, const double *array2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = array1[i] * array2[i];
    }
}

double getElemAtIndx(const double *array, int index) {
    return array[index];
}

int main() {
    double wc1 = 2 * PI * FC1 / FS;
    double wc2 = 2 * PI * FC2 / FS;
    double wc3 = 2 * PI * FC3 / FS;
    double wc4 = 2 * PI * FC4 / FS;

    double *hamming_window = hamming(INPUT_LENGTH);

    double *hpf1 = highPassFIRFilter(wc1, INPUT_LENGTH);
    double *hpf_w1 = malloc(INPUT_LENGTH * sizeof(double));
    elementWiseMultiplication(hpf_w1, hpf1, hamming_window, INPUT_LENGTH);

    double *hpf2 = highPassFIRFilter(wc2, INPUT_LENGTH);
    double *hpf_w2 = malloc(INPUT_LENGTH * sizeof(double));
    elementWiseMultiplication(hpf_w2, hpf2, hamming_window, INPUT_LENGTH);

    double *hpf3 = highPassFIRFilter(wc3, INPUT_LENGTH);
    double *hpf_w3 = malloc(INPUT_LENGTH * sizeof(double));
    elementWiseMultiplication(hpf_w3, hpf3, hamming_window, INPUT_LENGTH);

    double *hpf4 = highPassFIRFilter(wc4, INPUT_LENGTH);
    double *hpf_w4 = malloc(INPUT_LENGTH * sizeof(double));
    elementWiseMultiplication(hpf_w4, hpf4, hamming_window, INPUT_LENGTH);

    double final1 = getElemAtIndx(hpf_w1, 6);
    double final2 = getElemAtIndx(hpf_w2, 7);
    double final3 = getElemAtIndx(hpf_w3, 8);

    printf("%f\n", final1);
    printf("%f\n", final2);
    printf("%f\n", final3);

    free(hamming_window);
    free(hpf1);
    free(hpf2);
    free(hpf3);
    free(hpf4);
    free(hpf_w1);
    free(hpf_w2);
    free(hpf_w3);
    free(hpf_w4);

    return 0;
}
