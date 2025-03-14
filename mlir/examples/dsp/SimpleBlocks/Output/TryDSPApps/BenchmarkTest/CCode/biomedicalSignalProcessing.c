#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359
#define FS 8000
#define INPUT_LENGTH 2000
#define FILTER_SIZE 101
#define MAX_PEAKS 950

// Function prototypes
void getRangeOfVector(double* vector, double start, int length, double increment);
void gain(double* output, double* input, double multiplier, int length);
void sine(double* output, double* input, int length);
void add(double* output, double* input1, double* input2, int length);
void sub(double* output, double* input1, double* input2, int length);
void lowPassFIRFilter(double* lpf, double wc, int N);
void hamming(double* window, int length);
void FIRFilterResponse(double* output, double* input, double* filter, int input_length, int filter_length);
double max_signal(double* signal, int length);
void find_peaks(double* peaks, double* input, int length, double height, int distance);
void diff(double* output, double* input, int length);
double mean(double* input, int length);

int main() {
    double fc1 = 1000, fc2 = 7500;
    int N = FILTER_SIZE, distance = 950;

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
    add(noisy_sig, clean_sig, noise1, INPUT_LENGTH);

    // FIR Bandpass Filter
    double wc1 = 2 * PI * fc1 / FS;
    double wc2 = 2 * PI * fc2 / FS;

    double* lpf1 = (double*)malloc(N * sizeof(double));
    double* lpf2 = (double*)malloc(N * sizeof(double));
    lowPassFIRFilter(lpf1, wc1, N);
    lowPassFIRFilter(lpf2, wc2, N);

    double hamming_window[FILTER_SIZE];
    hamming(hamming_window, FILTER_SIZE);

    double* lpf1_w = (double*)malloc(N * sizeof(double));
    double* lpf2_w = (double*)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        lpf1_w[i] = lpf1[i] * hamming_window[i];
        lpf2_w[i] = lpf2[i] * hamming_window[i];
    }

    double* bpf_w = (double*)malloc(N * sizeof(double));
    sub(bpf_w, lpf2_w, lpf1_w, N);

    int conv_length = INPUT_LENGTH + N - 1;
    double* FIRfilterResponseForBpf = (double*)malloc(conv_length * sizeof(double));
    FIRFilterResponse(FIRfilterResponseForBpf, noisy_sig, bpf_w, INPUT_LENGTH, N);

    double max_val = max_signal(FIRfilterResponseForBpf, conv_length);
    double height = 0.3 * max_val;

    double* r_peaks = (double*)malloc(MAX_PEAKS * sizeof(double));
    find_peaks(r_peaks, FIRfilterResponseForBpf, conv_length, height, distance);
    // Get peak count from last index of r_peaks array (converted to double)
    double len_r_peaks = (double)r_peaks[MAX_PEAKS - 1]; // Number of detected peaks as double
    double last_peaks_index = len_r_peaks - 1.0;        // Last peak index as double
    
    double* diff_val = (double*)malloc((int)len_r_peaks * sizeof(double)); 
    diff(diff_val, r_peaks, (int)len_r_peaks);


// Compute peaks_count - 1
double peaks_count_minus_one = len_r_peaks - 1.0;

// Compute mean of peak differences
double diff_mean = mean(diff_val, (int)peaks_count_minus_one);

// Compute heart rate
double avg_hr = (60.0 * FS) / diff_mean;

printf("%f", avg_hr);


    // Free memory
    free(diff_val);
    free(input);
    free(getSinDuration);
    free(clean_sig);
    free(getNoiseSinDuration);
    free(noise);
    free(noise1);
    free(noisy_sig);
    free(lpf1);
    free(lpf2);
    free(lpf1_w);
    free(lpf2_w);
    free(bpf_w);
    free(FIRfilterResponseForBpf);
    free(r_peaks);

    return 0;
}

// Function implementations
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

void add(double* output, double* input1, double* input2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input1[i] + input2[i];
    }
}

void sub(double* output, double* input1, double* input2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input1[i] - input2[i];
    }
}

void hamming(double* window, int length) {
    for (int i = 0; i < length; i++) {
        window[i] = 0.54 - 0.46 * cos(2 * PI * i / (length - 1));
    }
}

void lowPassFIRFilter(double* lpf, double wc, int N) {
    int mid = (N - 1) / 2;
    for (int n = 0; n < N; n++) {
        if (n == mid) {
            lpf[n] = wc / PI;
        } else {
            double x = wc * (n - mid);
            lpf[n] = (wc / PI) * (sin(x) / x);
        }
    }
}

// Perform full convolution for FIR filtering
void FIRFilterResponse(double* output, double* input, double* filter, int input_length, int filter_length) {
    int conv_length = input_length + filter_length - 1;
    
    // Initialize output to zero
    for (int n = 0; n < conv_length; n++) {
        output[n] = 0;
    }

    // Perform full convolution
    for (int n = 0; n < conv_length; n++) {
        for (int k = 0; k < filter_length; k++) {
            if (n - k >= 0 && n - k < input_length) {
                output[n] += input[n - k] * filter[k];
            }
        }
    }
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

// Find peaks in a signal based on a threshold and minimum distance
void find_peaks(double* peaks, double* input, int length, double height, int distance) {
    int peakCount = 0;

    // Initialize peaks array with -1 (default no peaks)
    for (int i = 0; i < MAX_PEAKS; i++) {
        peaks[i] = -1;
    }

    for (int i = 1; i < length - 1; i++) {
        if (input[i] > input[i - 1] && input[i] > input[i + 1] && input[i] >= height) {
            // If it's the first peak, store it
            if (peakCount == 0) {
                peaks[peakCount++] = i;
            } else {
                // Ensure minimum distance between peaks
                if (i - (int)peaks[peakCount - 1] >= distance) {
                    peaks[peakCount++] = i;
                }
            }

            // Stop if max peaks reached
            if (peakCount >= MAX_PEAKS - 1) {
                break;
            }
        }
    }

    // Store peak count at the last index
    peaks[MAX_PEAKS - 1] = peakCount;
}



void diff(double* output, double* input, int length) {
    for (int i = 0; i < length - 1; i++) {
        output[i] = (double)(input[i + 1] - input[i]);
    }
}

double mean(double* input, int length) {
    double sum = 0;
    for (int i = 0; i < length; i++) {
        sum += input[i];
    }
    return sum / length;
}