#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define SAMPLE_RATE 1000
#define INPUT_LENGTH 61
#define DURATION ((double)INPUT_LENGTH / SAMPLE_RATE)
#define CORRELATION_LENGTH (2 * INPUT_LENGTH - 1)
#define M_PI 3.14159265358979323846


void generateVoiceSignature(double *signal, double freq1, double freq2) {
    for (int i = 0; i < INPUT_LENGTH; i++) {
        double t = i / (double)SAMPLE_RATE;
        signal[i] = sin(2 * M_PI * freq1 * t) + sin(2 * M_PI * freq2 * t);
    }
}

void correlate(const double *signal1, const double *signal2, double *result) {
    for (int lag = 0; lag < CORRELATION_LENGTH; lag++) {
        result[lag] = 0;
        for (int i = 0; i < INPUT_LENGTH; i++) {
            int j = lag - INPUT_LENGTH + 1 + i;
            if (j >= 0 && j < INPUT_LENGTH) {
                result[lag] += signal1[i] * signal2[j];
            }
        }
    }
}

double max(const double *arr, int length) {
    double max_value = arr[0];
    for (int i = 1; i < length; i++) {
        if (arr[i] > max_value) {
            max_value = arr[i];
        }
    }
    return max_value;
}

int argmax(const double *arr, int length) {
    int max_index = 0;
    for (int i = 1; i < length; i++) {
        if (arr[i] > arr[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}

int main() {
    double *person1 = (double *)malloc(INPUT_LENGTH * sizeof(double));
    double *person2 = (double *)malloc(INPUT_LENGTH * sizeof(double));
    double *person3 = (double *)malloc(INPUT_LENGTH * sizeof(double));
    double *unknown_signal = (double *)malloc(INPUT_LENGTH * sizeof(double));
    double *correlation1 = (double *)malloc(CORRELATION_LENGTH * sizeof(double));
    double *correlation2 = (double *)malloc(CORRELATION_LENGTH * sizeof(double));
    double *correlation3 = (double *)malloc(CORRELATION_LENGTH * sizeof(double));
    
    generateVoiceSignature(person1, 100, 200);
    generateVoiceSignature(person2, 150, 250);
    generateVoiceSignature(person3, 120, 180);
    generateVoiceSignature(unknown_signal, 150, 250);
    
    correlate(person1, unknown_signal, correlation1);
    correlate(person2, unknown_signal, correlation2);
    correlate(person3, unknown_signal, correlation3);
    
    double total_maxes[3];
    total_maxes[0] = max(correlation1, CORRELATION_LENGTH);
    total_maxes[1] = max(correlation2, CORRELATION_LENGTH);
    total_maxes[2] = max(correlation3, CORRELATION_LENGTH);
    
    double temp2 = total_maxes[0];
    double temp3 = total_maxes[1];
    double temp4 = total_maxes[2];
    
    int max_index = argmax(total_maxes, 3);
    double max_value = total_maxes[max_index];
    
    printf("%d\t", max_index);
    printf("%f\t", temp2);
    printf("%f\t", max_value);
    printf("%f\t", temp3);
    printf("%f %f %f\t", total_maxes[0], total_maxes[1], total_maxes[2]);
    printf("%f\t", temp4);
    
    free(person1);
    free(person2);
    free(person3);
    free(unknown_signal);
    free(correlation1);
    free(correlation2);
    free(correlation3);
    
    return 0;
}
