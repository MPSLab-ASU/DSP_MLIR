#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <complex.h>

#define PI 3.1415926
#define INPUT_LENGTH 10000
// Function declarations
double *getRangeOfVector(double start, double end, double step);
double complex *beam_form(int antennas, double input_fc, double *input, double *weights, int input_length);
double *abs_array(double complex *array, int length);
int argmax(double *array, int length);
double *lowPassFIRFilter(double wc, int length);
double *highPassFIRFilter(double wc, int length);
double *hamming(int length);
double *elementWiseMultiply(double *array1, double *array2, int length);
double *subtract(double *array1, double *array2, int length);
double *FIRFilterResponse(double *input, double *filter, int input_length, int filter_length);
double getElemAtIndx(double *array, int index);

int main() {
    int antennas = 4;
    double input_fc = 5;
    int N = 101;
    
    int input_length = INPUT_LENGTH;
    double *input = getRangeOfVector(0, input_length, 0.000125);
    
    int weights_length = (180 - (-90)) / 1 + 1;
    double *weights = getRangeOfVector(-90, 180, 1);

    double complex *signal = beam_form(antennas, input_fc, input, weights, input_length);
    double *b1 = abs_array(signal, input_length);
    
    double *power_profile = elementWiseMultiply(b1, b1, input_length);
    int power_angle_max_idx = argmax(power_profile, input_length);
    double power_angle_max_ele = power_profile[power_angle_max_idx];

    double fc1 = 1000;
    double fc2 = 7500;
    double Fs = 8000;

    double wc1 = 2 * PI * fc1 / Fs;
    double *filter1 = lowPassFIRFilter(wc1, N);
    double *hamming_window = hamming(N);
    double *filter_hamming_1 = elementWiseMultiply(filter1, hamming_window, N);

    double wc2 = 2 * PI * fc2 / Fs;
    double *filter2 = highPassFIRFilter(wc2, N);
    double *filter_hamming_2 = elementWiseMultiply(filter2, hamming_window, N);

    double *bpf = subtract(filter_hamming_2, filter_hamming_1, N);
    double *firFilterResponse = FIRFilterResponse(power_profile, bpf, input_length, N);
    double final = getElemAtIndx(firFilterResponse, 2);

    printf("%f\n", final);

    // Free allocated memory
    free(input);
    free(weights);
    free(signal);
    free(b1);
    free(power_profile);
    free(filter1);
    free(hamming_window);
    free(filter_hamming_1);
    free(filter2);
    free(filter_hamming_2);
    free(bpf);
    free(firFilterResponse);

    return 0;
}

// Function implementations
double *getRangeOfVector(double start, double end, double step) {
    int size = (int)((end - start) / step) + 1;
    double *vector = malloc(size * sizeof(double));
    for (int i = 0; i < size; i++) {
        vector[i] = start + i * step;
    }
    return vector;
}

double complex *beam_form(int antennas, double input_fc, double *input, double *weights, int input_length) {
    double complex *signal = malloc(input_length * sizeof(double complex));
    for (int i = 0; i < input_length; i++) {
        signal[i] = 0;
        for (int j = 0; j < antennas; j++) {
            signal[i] += weights[j] * cexp(I * (2 * PI * input_fc * input[i] + j * PI / 2));
        }
    }
    return signal;
}

double *abs_array(double complex *array, int length) {
    double *result = malloc(length * sizeof(double));
    for (int i = 0; i < length; i++) {
        result[i] = cabs(array[i]);
    }
    return result;
}

int argmax(double *array, int length) {
    int max_idx = 0;
    for (int i = 1; i < length; i++) {
        if (array[i] > array[max_idx]) {
            max_idx = i;
        }
    }
    return max_idx;
}

double *lowPassFIRFilter(double wc, int length) {
    double *filter = malloc(length * sizeof(double));
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

double *highPassFIRFilter(double wc, int length) {
    double *filter = malloc(length * sizeof(double));
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

double *hamming(int length) {
    double *window = malloc(length * sizeof(double));
    for (int i = 0; i < length; i++) {
        window[i] = 0.54 - 0.46 * cos(2 * PI * i / (length - 1));
    }
    return window;
}

double *elementWiseMultiply(double *array1, double *array2, int length) {
    double *result = malloc(length * sizeof(double));
    for (int i = 0; i < length; i++) {
        result[i] = array1[i] * array2[i];
    }
    return result;
}

double *subtract(double *array1, double *array2, int length) {
    double *result = malloc(length * sizeof(double));
    for (int i = 0; i < length; i++) {
        result[i] = array1[i] - array2[i];
    }
    return result;
}

double *FIRFilterResponse(double *input, double *filter, int input_length, int filter_length) {
    double *response = malloc(input_length * sizeof(double));
    for (int i = 0; i < input_length; i++) {
        response[i] = 0;
        for (int j = 0; j < filter_length && i - j >= 0; j++) {
            response[i] += input[i - j] * filter[j];
        }
    }
    return response;
}

double getElemAtIndx(double *array, int index) {
    return array[index];
}