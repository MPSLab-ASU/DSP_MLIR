#include <stdio.h>
#include <stdlib.h>
#include <math.h> // Include math.h for sin()

#ifndef M_PI
    #define M_PI 3.14159265358979323846 // Define Pi if it's not defined
#endif

// Function to generate an arbitrary noisy signal
void generate_noisy_signal(float *signal, int length, float noise_level) {
    for (int i = 0; i < length; i++) {
        // Generate a random signal value between -1.0 and 1.0
        float random_value = ((float)rand() / RAND_MAX) * 2.0f - 1.0f; // Random value in [-1, 1]
        
        // Add random noise
        float noise = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * noise_level; // Random noise in [-noise_level, noise_level]
        
        // Combine random value with noise
        signal[i] = random_value + noise;
    }
}

// Function to print the signal
void print_signal(const char *label, float *signal, int length) {
    printf("%s:\n", label);
    for (int i = 0; i < length; i++) {
        printf("%.4f ", signal[i]); // Display 4 digits after the decimal point
    }
    printf("\n");
}

// Function to return the maximum of three numbers
float max_of_three(float a, float b, float c) {
    float max = a;
    if (b > max) max = b;
    if (c > max) max = c;
    return max;
}

// Function to return the minimum of three numbers
float min_of_three(float a, float b, float c) {
    float min = a;
    if (b < min) min = b;
    if (c < min) min = c;
    return min;
}

// Function to apply sliding window average filter with kernel size of 3
void sliding_avg_filter(float *input, float *output, int length) {
    int new_length = length - 3 + 1;
    for (int i = 0; i < new_length; i++) {
        output[i] = (input[i] + input[i + 1] + input[i + 2]) / 3.0f;
    }
}

// Function to apply sliding window median filter with kernel size of 3
void sliding_median_filter(float *input, float *output, int length) {
    int new_length = length - 3 + 1;
    for (int i = 0; i < new_length; i++) {
        float a = input[i];
        float b = input[i + 1];
        float c = input[i + 2];
        // Median formula: median = a + b + c - max(a, b, c) - min(a, b, c)
        float max_val = max_of_three(a, b, c);
        float min_val = min_of_three(a, b, c);
        output[i] = a + b + c - max_val - min_val;
    }
}

int main() {
    int length = 20;  // Length of the original signal
    int avg_length = length - 3 + 1; // New length after filtering
    int median_length = avg_length - 3 + 1;

    float signal[length];            // Original signal
    float avg_filtered[avg_length];  // Signal after average filter
    float median_filtered[median_length]; // Signal after median filter

    float noise_level = 0.3f; // Example noise level

    // Generate an arbitrary noisy signal
    generate_noisy_signal(signal, length, noise_level);

    // Print the original signal
    print_signal("Original Signal", signal, length);

    // Apply sliding window average filter
    sliding_avg_filter(signal, avg_filtered, length);
    print_signal("After Average Filter", avg_filtered, avg_length);

    // Apply sliding window median filter
    sliding_median_filter(avg_filtered, median_filtered, avg_length);
    print_signal("After Median Filter", median_filtered, median_length);

    return 0;
}
