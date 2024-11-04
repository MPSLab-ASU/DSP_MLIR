#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359
#define INPUT_LENGTH 100000000
#define MAX_PEAKS 1000

// Function declarations
void getRangeOfVector(double* vector, double start, int length, double increment);
void gain(double* output, double* input, double multiplier, int length);
void sine(double* output, double* input, int length);
void add_signals(double* output, double* input1, double* input2, int length);
void sub_signals(double* output, double* input1, double* input2, int length);
void multiply_signals(double* output, double* input1, double* input2, int length);
void lowPassFIRFilter(double* lpf, double wc, int N);
void hamming(double* hamming, int N);
void FIRFilterResponse(double* output, double* input, double* filter, int input_length, int filter_length);
double max_signal(double* signal, int length);
int find_peaks(double* signal, int length, double threshold, int minDistance, int* peaks);
void diff(double* output, int* input, int length);
double mean(double* input, int length);

int main() {
    double fc1 = 1000;
    double fc2 = 7500;
    double Fs = 8000;
    int N = 101;
    int distance = 950;
    
    double* input = (double*)malloc(INPUT_LENGTH * sizeof(double));
    getRangeOfVector(input, 0, INPUT_LENGTH, 0.000125);

    double f_sig = 500;
    double getMultiplier = 2 * PI * f_sig;
    double* getSinDuration = (double*)malloc(INPUT_LENGTH * sizeof(double));
    gain(getSinDuration, input, getMultiplier, INPUT_LENGTH);

    double* clean_sig = (double*)malloc(INPUT_LENGTH * sizeof(double));
    sine(clean_sig, getSinDuration, INPUT_LENGTH);

    double f_noise = 3000;
    double* getNoiseSinDuration = (double*)malloc(INPUT_LENGTH * sizeof(double));
    gain(getNoiseSinDuration, input, 2 * PI * f_noise, INPUT_LENGTH);

    double* noise = (double*)malloc(INPUT_LENGTH * sizeof(double));
    sine(noise, getNoiseSinDuration, INPUT_LENGTH);

    double* noise1 = (double*)malloc(INPUT_LENGTH * sizeof(double));
    gain(noise1, noise, 0.5, INPUT_LENGTH);

    double* noisy_sig = (double*)malloc(INPUT_LENGTH * sizeof(double));
    add_signals(noisy_sig, clean_sig, noise1, INPUT_LENGTH);

    // Step 1: FIR Bandpass Filter
    double wc1 = 2 * PI * fc1 / Fs;
    double* lpf1 = (double*)malloc(N * sizeof(double));
    lowPassFIRFilter(lpf1, wc1, N);
    
    double* hamming_window = (double*)malloc(N * sizeof(double));
    hamming(hamming_window, N);
    
    double* lpf1_w = (double*)malloc(N * sizeof(double));
    multiply_signals(lpf1_w, lpf1, hamming_window, N);

    double wc2 = 2 * PI * fc2 / Fs;
    double* lpf2 = (double*)malloc(N * sizeof(double));
    lowPassFIRFilter(lpf2, wc2, N);
    
    double* lpf2_w = (double*)malloc(N * sizeof(double));
    multiply_signals(lpf2_w, lpf2, hamming_window, N);

    double* bpf_w = (double*)malloc(N * sizeof(double));
    sub_signals(bpf_w, lpf2_w, lpf1_w, N);

    double* FIRfilterResponseForBpf = (double*)malloc(INPUT_LENGTH * sizeof(double));
    FIRFilterResponse(FIRfilterResponseForBpf, noisy_sig, bpf_w, INPUT_LENGTH, N);

    // Step 2: Artifact Removal (R-peak detection)
    double max_val = max_signal(FIRfilterResponseForBpf, INPUT_LENGTH);
    double height = 0.3 * max_val;

    int* r_peaks = (int*)malloc(MAX_PEAKS * sizeof(int));
    int peaks_count = find_peaks(FIRfilterResponseForBpf, INPUT_LENGTH, height, distance, r_peaks);

    double* diff_val = (double*)malloc((peaks_count - 1) * sizeof(double));
    diff(diff_val, r_peaks, peaks_count);

    double diff_mean = mean(diff_val, peaks_count - 1);

    double avg_hr = (60 * Fs) / diff_mean;

    printf("%f\n", avg_hr);

    // Free allocated memory
    free(input);
    free(getSinDuration);
    free(clean_sig);
    free(getNoiseSinDuration);
    free(noise);
    free(noise1);
    free(noisy_sig);
    free(lpf1);
    free(hamming_window);
    free(lpf1_w);
    free(lpf2);
    free(lpf2_w);
    free(bpf_w);
    free(FIRfilterResponseForBpf);
    free(r_peaks);
    free(diff_val);

    return 0;
}

void getRangeOfVector(double* vector, double start, int length, double increment) {
    for (int i = 0; i < length; i++) {
        vector[i] = start + i * increment;
    }
}

void gain(double* output, double* input, double multiplier, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input[i] * multiplier;
    }
}

void sine(double* output, double* input, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = sin(input[i]);
    }
}

void add_signals(double* output, double* input1, double* input2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input1[i] + input2[i];
    }
}

void sub_signals(double* output, double* input1, double* input2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input1[i] - input2[i];
    }
}

void multiply_signals(double* output, double* input1, double* input2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input1[i] * input2[i];
    }
}

void hamming(double* hamming, int N) {
    for (int n = 0; n < N; n++) {
        hamming[n] = 0.54 - 0.46 * cos(2 * PI * n / (N - 1));
    }
}

void FIRFilterResponse(double* output, double* input, double* filter, int input_length, int filter_length) {
    int i, j;
    for (i = 0; i < input_length; i++) {
        output[i] = 0;
        for (j = 0; j < filter_length; j++) {
            if (i - j >= 0) {
                output[i] += input[i - j] * filter[j];
            }
        }
    }
}

void lowPassFIRFilter(double* lpf, double wc, int N) {
    int mid = (N - 1) / 2;
    for (int n = 0; n < N; n++) {
        if (n == mid) {
            lpf[n] = wc / PI;
        } else {
            lpf[n] = (wc / PI) * sin(wc * (n - mid)) / (wc * (n - mid));
        }
    }
}

int find_peaks(double* signal, int length, double threshold, int minDistance, int* peaks) {
    int num_peaks = 0;  
    for (int i = 1; i < length - 1 && num_peaks < MAX_PEAKS; i++) {
        if (signal[i] > threshold && signal[i] > signal[i-1] && signal[i] > signal[i+1]) {
            peaks[num_peaks++] = i;
            i += minDistance;  
        }
    }
    return num_peaks;
}

double max_signal(double* signal, int length) {
    double max = signal[0];
    for (int i = 1; i < length; i++) {
        if (signal[i] > max) {
            max = signal[i];
        }
    }
    return max;
}

void diff(double* output, int* input, int length) {
    for (int i = 0; i < length - 1; i++) {
        output[i] = (double)(input[i+1] - input[i]);
    }
}

double mean(double* input, int length) {
    double sum = 0;
    for (int i = 0; i < length; i++) {
        sum += input[i];
    }
    return sum / length;
}