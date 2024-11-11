#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Constants
#define PI 3.1415926

// Function prototypes
void getRangeOfVector(double start, double end, double step, double* result, long long int size);
void beam_form(int antennas, double fc, double* input, long long int input_size, double* weights, int weight_size, double* signal);
void hamming(int N, double* window);
void lowPassFIRFilter(double wc, int N, double* filter);
void highPassFIRFilter(double wc, int N, double* filter);
void FIRFilterResponse(double* input_signal, double* filter, long long int size, double* output_signal);
long long int argmax(double* array, long long int size);
double abs_val(double x);

int main() {
    int antennas = 4;
    double input_fc = 5;
    int N = 101;

    // Use long long int for larger size
    long long int input_size = (100000000 - 0) / 0.000125;
    double* input = (double*)malloc(input_size * sizeof(double));
    getRangeOfVector(0, 100000000, 0.000125, input, input_size);

    int weight_size = (180 - (-90)) / 1;
    double* weights = (double*)malloc(weight_size * sizeof(double));
    getRangeOfVector(-90, 180, 1, weights, weight_size);

    // Beamforming and power profile computation
    double* signal = (double*)malloc(input_size * sizeof(double));
    beam_form(antennas, input_fc, input, input_size, weights, weight_size, signal);

    double* power_profile = (double*)malloc(input_size * sizeof(double));
    for (long long int i = 0; i < input_size; i++) {
        double b1 = abs_val(signal[i]);
        power_profile[i] = b1 * b1;
    }

    long long int power_angle_max_idx = argmax(power_profile, input_size);
    long long int power_angle_max_ele = argmax(power_profile, input_size);

    // FIR Filter Design
    double fc1 = 1000, fc2 = 7500, Fs = 8000;
    double wc1 = 2 * PI * fc1 / Fs;
    double wc2 = 2 * PI * fc2 / Fs;

    double* filter1 = (double*)malloc(N * sizeof(double));
    double* filter2 = (double*)malloc(N * sizeof(double));

    lowPassFIRFilter(wc1, N, filter1);
    highPassFIRFilter(wc2, N, filter2);

    double* hamming_window = (double*)malloc(N * sizeof(double));
    hamming(N, hamming_window);

    // Apply Hamming window to filters
    for (int i = 0; i < N; i++) {
        filter1[i] *= hamming_window[i];
        filter2[i] *= hamming_window[i];
    }

    // Band-pass filter
    double* bpf = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        bpf[i] = filter2[i] - filter1[i];
    }

    // Filter the power profile
    double* fir_filter_response = (double*)malloc(input_size * sizeof(double));
    FIRFilterResponse(power_profile, bpf, input_size, fir_filter_response);

    // Get the final value
    double final = fir_filter_response[2];
    printf("Final Value: %f\n", final);

    // Clean up memory
    free(input);
    free(weights);
    free(signal);
    free(power_profile);
    free(filter1);
    free(filter2);
    free(hamming_window);
    free(bpf);
    free(fir_filter_response);

    return 0;
}

// Helper functions

void getRangeOfVector(double start, double end, double step, double* result, long long int size) {
    for (long long int i = 0; i < size; i++) {
        result[i] = start + i * step;
    }
}

void beam_form(int antennas, double fc, double* input, long long int input_size, double* weights, int weight_size, double* signal) {
    // Placeholder for beamforming logic
    for (long long int i = 0; i < input_size; i++) {
        signal[i] = input[i] * sin(2 * PI * fc * i / input_size);
    }
}

void hamming(int N, double* window) {
    for (int i = 0; i < N; i++) {
        window[i] = 0.54 - 0.46 * cos(2 * PI * i / (N - 1));
    }
}

void lowPassFIRFilter(double wc, int N, double* filter) {
    for (int i = 0; i < N; i++) {
        if (i == N / 2) {
            filter[i] = wc / PI;
        } else {
            filter[i] = sin(wc * (i - N / 2)) / (PI * (i - N / 2));
        }
    }
}

void highPassFIRFilter(double wc, int N, double* filter) {
    for (int i = 0; i < N; i++) {
        if (i == N / 2) {
            filter[i] = 1 - wc / PI;
        } else {
            filter[i] = -sin(wc * (i - N / 2)) / (PI * (i - N / 2));
        }
    }
}

void FIRFilterResponse(double* input_signal, double* filter, long long int size, double* output_signal) {
    // Convolution of input signal with FIR filter
    for (long long int i = 0; i < size; i++) {
        output_signal[i] = 0;
        for (long long int j = 0; j <= i; j++) {
            output_signal[i] += input_signal[j] * filter[i - j];
        }
    }
}

long long int argmax(double* array, long long int size) {
    long long int max_idx = 0;
    for (long long int i = 1; i < size; i++) {
        if (array[i] > array[max_idx]) {
            max_idx = i;
        }
    }
    return max_idx;
}

double abs_val(double x) {
    return x < 0 ? -x : x;
}
