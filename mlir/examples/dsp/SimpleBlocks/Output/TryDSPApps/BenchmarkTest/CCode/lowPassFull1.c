#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359
#define INPUT_LENGTH 10000

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

void gain(double* output, double* input, double multiplier, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input[i] * multiplier;
    }
}

void elementWiseAdd(double* output, double* input1, double* input2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input1[i] + input2[i];
    }
}

void elementWiseMultiply(double* output, double* input1, double* input2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input1[i] * input2[i];
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

void hammingWindow(double* hamming, int N) {
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

int main() {
    int fs = 8000;

    // Allocate memory dynamically
    double* input = getRangeOfVector(0, INPUT_LENGTH, 0.000125);
    
    // Allocate other large arrays dynamically
    double* getSinDuration = malloc(INPUT_LENGTH * sizeof(double));
    double* clean_sig = malloc(INPUT_LENGTH * sizeof(double));
    double* getNoiseSinDuration = malloc(INPUT_LENGTH * sizeof(double));
    double* noise = malloc(INPUT_LENGTH * sizeof(double));
    double* noisy_sig = malloc(INPUT_LENGTH * sizeof(double));
    double* scaled_noise = malloc(INPUT_LENGTH * sizeof(double));
    double* FIRfilterResponse = malloc(INPUT_LENGTH * sizeof(double));

    // Check if memory allocation was successful
    if (!getSinDuration || !clean_sig || !getNoiseSinDuration || !noise || !noisy_sig || !scaled_noise || !FIRfilterResponse) {
        perror("Memory allocation failed");
        free(input);
        free(getSinDuration);
        free(clean_sig);
        free(getNoiseSinDuration);
        free(noise);
        free(noisy_sig);
        free(scaled_noise);
        free(FIRfilterResponse);
        exit(EXIT_FAILURE);
    }

    // Signal processing steps
    double f_sig = 500;
    gain(getSinDuration, input, 2 * PI * f_sig, INPUT_LENGTH);

    for (int i = 0; i < INPUT_LENGTH; i++) {
        clean_sig[i] = sin(getSinDuration[i]);
    }

    double f_noise = 3000;
    gain(getNoiseSinDuration, input, 2 * PI * f_noise, INPUT_LENGTH);

    for (int i = 0; i < INPUT_LENGTH; i++) {
        noise[i] = sin(getNoiseSinDuration[i]);
    }

    gain(scaled_noise, noise, 0.5, INPUT_LENGTH);
    elementWiseAdd(noisy_sig, clean_sig, scaled_noise, INPUT_LENGTH);

    // Filter design
    double fc = 1000;
    double wc = 2 * PI * fc / fs;
    int N = 101;

    double lpf[N];
    lowPassFIRFilter(lpf, wc, N);

    double hamming[N];
    hammingWindow(hamming, N);

    double lpf_w[N];
    elementWiseMultiply(lpf_w, lpf, hamming, N);

    FIRFilterResponse(FIRfilterResponse, noisy_sig, lpf_w, INPUT_LENGTH, N);
    
    for (int i = 0; i < INPUT_LENGTH; i++) {
        printf("%f\n", FIRfilterResponse[i]);
    }

   // Free allocated memory at the end
   free(input);
   free(getSinDuration);
   free(clean_sig);
   free(getNoiseSinDuration);
   free(noise);
   free(noisy_sig);
   free(scaled_noise);
   free(FIRfilterResponse);

   return 0;
}