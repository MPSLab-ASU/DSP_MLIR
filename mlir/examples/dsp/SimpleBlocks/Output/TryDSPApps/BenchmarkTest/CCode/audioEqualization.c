#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define PI 3.14159265359
#define INPUT_LENGTH 10
#define FILTER_LENGTH 101
#define OUTPUT_LENGTH (INPUT_LENGTH + FILTER_LENGTH - 1)

// Function prototypes
double* getRangeOfVector(double start, int length, double increment);
double* lowPassFIRFilter(double wc, int length);
double* highPassFIRFilter(double wc, int length);
double* hamming(int length);
void elementWiseMultiplication(double* output, const double* array1, const double* array2, int length);
void FIRFilterResponse(double* output, const double* input, const double* filter, int inputLength);
void gain(double* output, const double* input, double gainFactor, int length);
void add(double* output, const double* input1, const double* input2, int length);
void sub(double* output, const double* input1, const double* input2, int length);
void printArray(const double* array, int length);
double getElementAtIndex(const double* array, int index);

// Generate a range of values
double* getRangeOfVector(double start, int length, double increment) {
    double* vector = malloc(length * sizeof(double));
    if (!vector) {
        perror("Memory allocation failed in getRangeOfVector");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < length; i++) {
        vector[i] = start + i * increment;
    }
    return vector;
}

// Generate a Hamming window
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

// Generate an ideal low-pass FIR filter
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

// Generate an ideal high-pass FIR filter
double* highPassFIRFilter(double wc, int length) {
    double* lpf = lowPassFIRFilter(wc, length);
    double* hpf = malloc(length * sizeof(double));
    if (!hpf) {
        perror("Memory allocation failed in highPassFIRFilter");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < length; i++) {
        hpf[i] = -lpf[i];
    }
    int mid = (length - 1) / 2;
    hpf[mid] += 1.0;
    free(lpf);
    return hpf;
}

// Perform element-wise multiplication
void elementWiseMultiplication(double* output, const double* array1, const double* array2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = array1[i] * array2[i];
    }
}

void FIRFilterResponse(double* output, const double* input, const double* filter, int inputLength) {
    int conv_length = inputLength + FILTER_LENGTH - 1; 
    
    // Initialize output to zero
    for (int n = 0; n < conv_length; n++) {
        output[n] = 0;
    }

    // Perform convolution
    for (int n = 0; n < conv_length; n++) {
        for (int k = 0; k < FILTER_LENGTH; k++) {
            if (n - k >= 0 && n - k < inputLength) {
                output[n] += input[n - k] * filter[k];
            }
        }
    }
}

// Apply gain to a signal
void gain(double* output, const double* input, double gainFactor, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input[i] * gainFactor;
    }
}

// Perform element-wise addition
void add(double* output, const double* input1, const double* input2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input1[i] + input2[i];
    }
}

// Perform element-wise subtraction
void sub(double* output, const double* input1, const double* input2, int length) {
    for (int i = 0; i < length; i++) {
        output[i] = input1[i] - input2[i];
    }
}

// Print an array
void printArray(const double* array, int length) {
    for (int i = 0; i < length; i++) {
        printf("%f ", array[i]);
    }
    printf("\n");
}

// Get element at index
double getElementAtIndex(const double* array, int index) {
    return array[index];
}

int main() {
    // Step 1: Generate Input Signal
    double* input = getRangeOfVector(0, INPUT_LENGTH, 1);
    
    // Step 2: Define constants
    double pi = PI;
    double Fs = 8000;
    double gainForBass = 2;
    double gainForMid = 1.5;
    double gainForTreble = 0.8;

    // Step 3: Low-pass filter
    double fc = 300;
    double wc = 2 * pi * fc / Fs;
    double* lpf = lowPassFIRFilter(wc, FILTER_LENGTH);
    double* hamming_window = hamming(FILTER_LENGTH);
    double* lpf_w = malloc(FILTER_LENGTH * sizeof(double));
    elementWiseMultiplication(lpf_w, lpf, hamming_window, FILTER_LENGTH);

    double* FIRfilterResponseForLpf = malloc(OUTPUT_LENGTH * sizeof(double));
    FIRFilterResponse(FIRfilterResponseForLpf, input, lpf_w, INPUT_LENGTH);

    double* gainWithLpf = malloc(OUTPUT_LENGTH * sizeof(double));
    gain(gainWithLpf, FIRfilterResponseForLpf, gainForBass, OUTPUT_LENGTH);

    // Step 4: High-pass filter
    double fc2 = 1500;
    double wc2 = 2 * pi * fc2 / Fs;
    double* hpf = highPassFIRFilter(wc2, FILTER_LENGTH);
    double* hpf_w = malloc(FILTER_LENGTH * sizeof(double));
    elementWiseMultiplication(hpf_w, hpf, hamming_window, FILTER_LENGTH);

    double* FIRfilterResponseForHpf = malloc(OUTPUT_LENGTH * sizeof(double));
    FIRFilterResponse(FIRfilterResponseForHpf, input, hpf_w, INPUT_LENGTH);

    double* gainWithHpf = malloc(OUTPUT_LENGTH * sizeof(double));
    gain(gainWithHpf, FIRfilterResponseForHpf, gainForTreble, OUTPUT_LENGTH);

    // Step 5: Band-pass filter
    double* lpf2 = lowPassFIRFilter(wc2, FILTER_LENGTH);
    double* lpf2_w = malloc(FILTER_LENGTH * sizeof(double));
    elementWiseMultiplication(lpf2_w, lpf2, hamming_window, FILTER_LENGTH);

    double* bpf_w = malloc(FILTER_LENGTH * sizeof(double));
    sub(bpf_w, lpf2_w, lpf_w, FILTER_LENGTH);

    double* FIRfilterResponseForBpf = malloc(OUTPUT_LENGTH * sizeof(double));
    FIRFilterResponse(FIRfilterResponseForBpf, input, bpf_w, INPUT_LENGTH);

  

    // Apply gain to Band-pass filter response
    double* gainWithBpf = malloc(OUTPUT_LENGTH * sizeof(double));
    gain(gainWithBpf, FIRfilterResponseForBpf, gainForTreble, OUTPUT_LENGTH);

    // Compute final audio by summing all filter responses
    double* final_audio = malloc(OUTPUT_LENGTH * sizeof(double));
    add(final_audio, gainWithLpf, gainWithHpf, OUTPUT_LENGTH);
    add(final_audio, final_audio, gainWithBpf, OUTPUT_LENGTH);


    // Extract and print the element at index 3
    double final1 = getElementAtIndex(final_audio, 3);
    printf("%f\n", final1);

    // Free allocated memory
    free(input);
    free(lpf);
    free(hamming_window);
    free(lpf_w);
    free(FIRfilterResponseForLpf);
    free(gainWithLpf);
    free(hpf);
    free(hpf_w);
    free(FIRfilterResponseForHpf);
    free(gainWithHpf);
    free(lpf2);
    free(lpf2_w);
    free(bpf_w);
    free(FIRfilterResponseForBpf);
    free(gainWithBpf);
    free(final_audio);

    return 0;
}

