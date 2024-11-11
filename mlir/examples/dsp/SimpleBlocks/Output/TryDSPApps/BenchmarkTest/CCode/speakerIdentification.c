#include <stdio.h>
#include <math.h>

#define SAMPLE_RATE 1000
#define DURATION 1
#define SIGNAL_LENGTH (SAMPLE_RATE * DURATION)

// Function to generate voice signature (sinusoidal wave with two frequencies)
void generateVoiceSignature(double freq1, double freq2, double signal[], int length, int sample_rate) {
    for (int i = 0; i < length; i++) {
        double t = (double)i / sample_rate;
        signal[i] = sin(2 * M_PI * freq1 * t) + cos(2 * M_PI * freq2 * t);
    }
}

// Function to compute the dot product (correlation) between two signals
double correlate(double signal1[], double signal2[], int length) {
    double result = 0.0;
    for (int i = 0; i < length; i++) {
        result += signal1[i] * signal2[i];
    }
    return result;
}

// Function to find the index of the maximum value in an array
int argmax(double arr[], int length) {
    int max_index = 0;
    for (int i = 1; i < length; i++) {
        if (arr[i] > arr[max_index]) {
            max_index = i;
        }
    }
    return max_index;
}

// Main function
int main() {
    // Sample rate
    int sample_rate = SAMPLE_RATE;
    
    // Signal arrays
    double person1[SIGNAL_LENGTH], person2[SIGNAL_LENGTH], person3[SIGNAL_LENGTH], unknown_signal[SIGNAL_LENGTH];
    
    // Generate voice signatures for Alice, Bob, Charlie
    generateVoiceSignature(100, 200, person1, SIGNAL_LENGTH, sample_rate); // Alice
    generateVoiceSignature(150, 250, person2, SIGNAL_LENGTH, sample_rate); // Bob
    generateVoiceSignature(120, 180, person3, SIGNAL_LENGTH, sample_rate); // Charlie

    // Generate an unknown signal (Bob's signature in this case)
    generateVoiceSignature(150, 250, unknown_signal, SIGNAL_LENGTH, sample_rate);
    
    // Correlate unknown signal with each person's signature
    double max1 = correlate(person1, unknown_signal, SIGNAL_LENGTH);
    double max2 = correlate(person2, unknown_signal, SIGNAL_LENGTH);
    double max3 = correlate(person3, unknown_signal, SIGNAL_LENGTH);
    
    // Store correlation results
    double total_maxes[3] = {0, 0, 0};
    
    // Set correlation results
    total_maxes[0] = max1;
    total_maxes[1] = max2;
    total_maxes[2] = max3;
    
    // Find the index of the maximum correlation result
    int max_index = argmax(total_maxes, 3);
    
    // Output results
    printf("Max Index: %d\n", max_index);
    printf("Max Value: %f\n", total_maxes[max_index]);
    printf("Correlation with Alice: %f\n", max1);
    printf("Correlation with Bob: %f\n", max2);
    printf("Correlation with Charlie: %f\n", max3);
    
    return 0;
}
