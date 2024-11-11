#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define M_PI 3.14159265358979323846
#define INPUT_LENGTH 50000

void dftReal(double* real, double* input, int length) {
    for (int k = 0; k < length; k++) {
        real[k] = 0;
        for (int n = 0; n < length; n++) {
            double angle = 2 * M_PI * k * n / length;
            real[k] += input[n] * cos(angle);
        }
    }
}

void dftImag(double* imag, double* input, int length) {
    for (int k = 0; k < length; k++) {
        imag[k] = 0;
        for (int n = 0; n < length; n++) {
            double angle = 2 * M_PI * k * n / length;
            imag[k] -= input[n] * sin(angle);
        }
    }
}

void generateDtmf(double* dtmf_tone, int digit, double duration, int fs) {
    double freqPairs[10][2] = {
        {941, 1336}, {697, 1209}, {697, 1336}, {697, 1477},
        {770, 1209}, {770, 1336}, {770, 1477}, {852, 1209},
        {852, 1336}, {852, 1477}
    };
    
    double f1 = freqPairs[digit][0];
    double f2 = freqPairs[digit][1];
    int N = fs * duration;
    
    for (int i = 0; i < N; i++) {
        double t = (double)i / fs;
        dtmf_tone[i] = 10* sin(2 * M_PI * f1 * t) + sin(2 * M_PI * f2 * t);
    }
}

void findDominantPeaks(double* frequencies, double* magnitudes, int fft_size, double* peaks) {
    double max1 = 0.0, max2 = 0.0;
    double freq1 = 0.0, freq2 = 0.0;

    for (int i = 0; i < fft_size; i++) {
        double currentFreq = frequencies[i];
        double currentMag = magnitudes[i];

        // Check if frequency is positive
        if (currentFreq >= 0.0) {
            // Compare current magnitude with max1
            if (currentMag > max1) {
                // Update max2 and freq2 with previous max1 and freq1
                max2 = max1;
                freq2 = freq1;
                // Update max1 and freq1 with current values
                max1 = currentMag;
                freq1 = currentFreq;
            } else if (currentMag > max2) {
                // Update max2 and freq2 with current values
                max2 = currentMag;
                freq2 = currentFreq;
            }
        }
        // No update for negative frequencies
    }

    // Compare freq1 and freq2 to determine the order
    if (freq1 < freq2) {
        peaks[0] = freq1;
        peaks[1] = freq2;
    } else {
        peaks[0] = freq2;
        peaks[1] = freq1;
    }
}

// Function to recover the DTMF digit from frequency peaks
int recoverDTMFDigit(double* peaks, const double freqPairs[10][2], int peak_count) {
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
    int digit = 6;
    int fs = 8192;
    double duration = (double)INPUT_LENGTH / fs;
    int N = fs * duration;

    double* dtmf_tone = (double*)malloc(N * sizeof(double));
    generateDtmf(dtmf_tone, digit, duration, fs);

    double* fft_real = (double*)malloc(N * sizeof(double));
    double* fft_imag = (double*)malloc(N * sizeof(double));
    
    dftReal(fft_real, dtmf_tone, N);
    dftImag(fft_imag, dtmf_tone, N);

    double* magnitudes = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        magnitudes[i] = sqrt(fft_real[i] * fft_real[i] + fft_imag[i] * fft_imag[i]);
    }

    double* frequencies = (double*)malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) {
        magnitudes[i] = sqrt(fft_real[i] * fft_real[i] + fft_imag[i] * fft_imag[i]);
        if (i <= N / 2) {
            frequencies[i] = (double)i * fs / N;
        } else {
            frequencies[i] = ((double)i - N) * fs / N;
        }
    }

    double peaks[2];
    findDominantPeaks(frequencies, magnitudes, N, peaks);
    printf("Peaks: %.2f, %.2f\n", peaks[0], peaks[1]);

    double freqPairs[10][2] = {
        {941, 1336}, {697, 1209}, {697, 1336}, {697, 1477},
        {770, 1209}, {770, 1336}, {770, 1477}, {852, 1209},
        {852, 1336}, {852, 1477}
    };

    double recovered_digit = recoverDTMFDigit(peaks, freqPairs, 10);
    printf("Recovered digit: %.0f\n", recovered_digit);

    free(dtmf_tone);
    free(fft_real);
    free(fft_imag);
    free(magnitudes);
    free(frequencies);

    return 0;
}