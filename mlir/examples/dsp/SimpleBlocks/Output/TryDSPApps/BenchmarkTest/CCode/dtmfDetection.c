#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SAMPLING_FREQUENCY 8192  // Sampling frequency
#define DURATION 0.5             // Duration of the DTMF signal
#define N_SAMPLES (int)(SAMPLING_FREQUENCY * DURATION) // Number of samples for the DTMF signal

// DTMF frequencies
const double freqPairs[10][2] = {
    {941, 1336},   // 0
    {697, 1209},   // 1
    {697, 1336},   // 2
    {697, 1477},   // 3
    {770, 1209},   // 4
    {770, 1336},   // 5
    {770, 1477},   // 6
    {852, 1209},   // 7
    {852, 1336},   // 8
    {852, 1477}    // 9
};

// Function to generate the DTMF tone for a given digit
void generateDtmf(int digit, double* dtmf_tone, int fs) {
    double f1 = freqPairs[digit][0];
    double f2 = freqPairs[digit][1];
    
    for (int i = 0; i < N_SAMPLES; i++) {
        double t = (double)i / fs;
        dtmf_tone[i] = 10 * (sin(2 * M_PI * f1 * t) + sin(2 * M_PI * f2 * t));
    }
}

// Function to perform the Discrete Fourier Transform (DFT)
void dft(double* signal, double* real_out, double* imag_out, int N) {
    for (int k = 0; k < N; k++) {
        real_out[k] = 0;
        imag_out[k] = 0;
        for (int n = 0; n < N; n++) {
            double angle = 2 * M_PI * k * n / N;
            real_out[k] += signal[n] * cos(angle);
            imag_out[k] -= signal[n] * sin(angle);
        }
    }
}

// Function to calculate the magnitudes from the real and imaginary parts of DFT
void calculateMagnitudes(double* real, double* imag, double* magnitudes, int N) {
    for (int i = 0; i < N; i++) {
        magnitudes[i] = sqrt(real[i] * real[i] + imag[i] * imag[i]);
    }
}

// Function to find dominant peaks in the magnitude spectrum and ensure they are in ascending order
// Function to find the two highest peaks in the magnitude spectrum and return their frequencies
void findDominantPeaks(double* frequencies, double* magnitudes, int fft_size, double* peaks) {
    double max1 = 0.0, max2 = 0.0;  // Variables to hold the two largest magnitudes
    int idx1 = -1, idx2 = -1;       // Indices for the two largest magnitudes

    // Iterate over the magnitude array to find the two highest magnitudes
    for (int i = 0; i < fft_size; i++) {
        if (magnitudes[i] > max1) {
            // Shift max1 to max2 and update max1
            max2 = max1;
            idx2 = idx1;
            max1 = magnitudes[i];
            idx1 = i;
        } else if (magnitudes[i] > max2) {
            max2 = magnitudes[i];
            idx2 = i;
        }
    }

    // Assign the corresponding frequencies to the peaks array in ascending order
    if (frequencies[idx1] < frequencies[idx2]) {
        peaks[0] = frequencies[idx1];
        peaks[1] = frequencies[idx2];
    } else {
        peaks[0] = frequencies[idx2];
        peaks[1] = frequencies[idx1];
    }
}


// Function to recover the DTMF digit from frequency peaks
int recoverDtmfDigit(double* peaks, const double freqPairs[10][2], int peak_count) {
    for (int i = 0; i < 10; i++) {
        double f1 = freqPairs[i][0];
        double f2 = freqPairs[i][1];

        if ((fabs(peaks[0] - f1) < 10 && fabs(peaks[1] - f2) < 10) ||
            (fabs(peaks[0] - f2) < 10 && fabs(peaks[1] - f1) < 10)) {
            return i; // Digit found
        }
    }
    return -1; // No match found
}

int main() {
    int digit = 9; // DTMF digit to be generated
    double duration = DURATION;
    int fs = SAMPLING_FREQUENCY;

    // Allocate memory for the DTMF signal and DFT output
    double* dtmf_tone = (double*)malloc(N_SAMPLES * sizeof(double));
    double* real_out = (double*)malloc(N_SAMPLES * sizeof(double));
    double* imag_out = (double*)malloc(N_SAMPLES * sizeof(double));
    double* magnitudes = (double*)malloc(N_SAMPLES * sizeof(double));
    double* frequencies = (double*)malloc(N_SAMPLES * sizeof(double));

    // Generate the DTMF tone
    generateDtmf(digit, dtmf_tone, fs);

    // Perform DFT
    dft(dtmf_tone, real_out, imag_out, N_SAMPLES);

    // Calculate magnitudes and frequencies
    for (int i = 0; i < N_SAMPLES / 2; i++) {
        magnitudes[i] = sqrt(real_out[i] * real_out[i] + imag_out[i] * imag_out[i]);
        frequencies[i] = (double)i * fs / N_SAMPLES;
    }

    // Find dominant frequency peaks (in ascending order)
    double peaks[2] = {0, 0}; // We expect 2 dominant peaks
    findDominantPeaks(frequencies, magnitudes, N_SAMPLES, peaks);
    // Recover the DTMF digit
    int recovered_digit = recoverDtmfDigit(peaks, freqPairs, 2);
    if (recovered_digit >= 0) {
        printf("Recovered DTMF digit: %d\n", recovered_digit);
    } else {
        printf("No DTMF digit detected.\n");
    }

    // Cleanup
    free(dtmf_tone);
    free(real_out);
    free(imag_out);
    free(magnitudes);
    free(frequencies);

    return 0;
}
