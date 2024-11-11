#include <math.h>
#include <stdio.h>

#define SAMPLE_RATE 1000
#define INPUT_LENGTH 20000000

// Declare arrays globally
double person1[INPUT_LENGTH];
double person2[INPUT_LENGTH];
double person3[INPUT_LENGTH];
double unknown_signal[INPUT_LENGTH];

void generateVoiceSignature(double freq1, double freq2, double signal[],
                            int length, int sample_rate) {
    for (int i = 0; i < length; i++) {
        double t = (double)i / sample_rate;
        signal[i] = sin(2 * M_PI * freq1 * t) + cos(2 * M_PI * freq2 * t);
    }
}

double correlate(double signal1[], double signal2[], int length) {
    double result = 0.0;
    for (int i = 0; i < length; i++) {
        result += signal1[i] * signal2[i];
    }
    return result;
}

int argmax(double arr[], int length) {
    int max_index = 0;
    for (int i = 1; i < length; i++) {
        if (arr[i] > arr[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}

int main() {
    // Sample rate
    int sample_rate = SAMPLE_RATE;

    // Generate voice signatures
    generateVoiceSignature(100, 200, person1, INPUT_LENGTH, sample_rate); // Alice
    generateVoiceSignature(150, 250, person2, INPUT_LENGTH, sample_rate); // Bob
    generateVoiceSignature(120, 180, person3, INPUT_LENGTH, sample_rate); // Charlie

    // Generate an unknown signal (Bob's signature in this case)
    generateVoiceSignature(150, 250, unknown_signal, INPUT_LENGTH, sample_rate);

    // Correlate unknown signal with each person's signature
    double max1 = correlate(person1, unknown_signal, INPUT_LENGTH);
    double max2 = correlate(person2, unknown_signal, INPUT_LENGTH);
    double max3 = correlate(person3, unknown_signal, INPUT_LENGTH);

    // Store correlation results
    double total_maxes[3] = {max1, max2, max3};

    // Find the index of the maximum correlation result
    int max_index = argmax(total_maxes, 3);

    // Output results
    printf("Correlation with Alice: %f\n", max1);
    printf("Correlation with Bob: %f\n", max2);
    printf("Correlation with Charlie: %f\n", max3);
    printf("Best match index: %d\n", max_index);

    return 0;
}