#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define INPUT_LENGTH 100
#define WEIGHT_LENGTH 4
#define NUM_ANTENNAS WEIGHT_LENGTH
#define FREQUENCY 5  // Hz
#define PI 3.14159265359
#define STARTTIME 0
#define ENDTIME 99

void generate_time(double *time) {
    double start = 0.0;
    double end = 99.0;
    double step = (end - start) / (INPUT_LENGTH - 1);
    for (int i = 0; i < INPUT_LENGTH; i++) {
        time[i] = start + i * step;
    }
}


// Generate input signals for each antenna
void generate_signals(double** signals, double* time) {
    for (int i = 0; i < NUM_ANTENNAS; i++) {
        double phase_shift = i * PI / 4.0;  // Different phase for each antenna
        for (int j = 0; j < INPUT_LENGTH; j++) {
            double time = j;
            signals[i][j] = sin(2.0 * PI * FREQUENCY * time + phase_shift);
        }
    }
}

void beamform_signal(double** input_signals, double *weights, double *beamformed_signal) {
    for (int j = 0; j < INPUT_LENGTH; j++) {
        beamformed_signal[j] = 0.0;
        for (int i = 0; i < NUM_ANTENNAS; i++) {
            beamformed_signal[j] += input_signals[i][j] * weights[i];
        }
    }
}

int main() {
    double* time; // [INPUT_LENGTH];
    double** input_signals; // [NUM_ANTENNAS][INPUT_LENGTH];
    double* weights; // [NUM_ANTENNAS] = {1, 2, 3, 4};  // Example weights for alignment
    double* beamformed_signal; // [INPUT_LENGTH];

    time = (double*)malloc(INPUT_LENGTH * sizeof(double));
    generate_time(time);

    input_signals = (double**)malloc(NUM_ANTENNAS * sizeof(double*));
    for (int i = 0; i < NUM_ANTENNAS; i++) {
        input_signals[i] = (double*)malloc(INPUT_LENGTH * sizeof(double));
    }
    generate_signals(input_signals, time);

    weights = (double*)malloc(NUM_ANTENNAS * sizeof(double));
    for(int i=1; i<=NUM_ANTENNAS; i++) {
        weights[i-1] = i;
    }

    beamformed_signal = (double*)malloc(INPUT_LENGTH * sizeof(double));
    beamform_signal(input_signals, weights, beamformed_signal);

    // Print the beamformed signal
    for (int j = 0; j < INPUT_LENGTH; j++) {
        printf("Beamformed Signal[%d]: %f\n", j, beamformed_signal[j]);
    }

    for(int i=0; i<NUM_ANTENNAS; i++) {
        free(input_signals[i]);
    }
    free(input_signals);
    free(weights);
    free(beamformed_signal);
    return 0;
}
