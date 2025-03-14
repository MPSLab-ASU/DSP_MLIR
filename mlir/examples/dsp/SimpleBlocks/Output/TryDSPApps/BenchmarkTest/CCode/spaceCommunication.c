#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define INPUT_LENGTH 40000

double *getRangeOfVector(double start, int length, double increment) {
    double *vector = malloc(length * sizeof(double));
    if (!vector) {
      perror("Memory allocation failed in getRangeOfVector");
      exit(EXIT_FAILURE);
    }
    for (int i = 0; i < length; i++) {
      vector[i] = start + i * increment;
    //   printf("%.6f ", vector[i]); 
    }
    return vector;
  }

double *gain(const double *input, int length, double increment) {
    double *vector = malloc(length * sizeof(double));
    if (!vector) {
      perror("Memory allocation failed in getRangeOfVector");
      exit(EXIT_FAILURE);
    }
    for (int i = 0; i < length; i++) {
      vector[i] = input[i] * increment;
    //   printf("%.6f ", vector[i]); 
    }
    return vector;
  }

double *Sin(const double *input, int length) {
    double *vector = malloc(length * sizeof(double));
    if (!vector) {
      perror("Memory allocation failed in getRangeOfVector");
      exit(EXIT_FAILURE);
    }
    for (int i = 0; i < length; i++) {
      vector[i] = sin(input[i]);
    //   printf("%.6f ", vector[i]); 
    }
    return vector;
  }

double *thresholdUp(const double *input, int length, double threshold, int returnOrignal) {
    double *vector = malloc(length * sizeof(double));
    if (!vector) {
      perror("Memory allocation failed in getRangeOfVector");
      exit(EXIT_FAILURE);
    }

    if (returnOrignal == 0) {
        for (int i = 0; i < length; i++) {
            if (input[i] >= threshold) {
                vector[i] = 1;
            }
            else {
                vector[i] = 0;
            }
            // printf("%.6f ", vector[i]);
        }
    }
    else {
        for (int i = 0; i < length; i++) {
            if (input[i] >= threshold) {
                vector[i] = input[i];
            }
            else {
                vector[i] = 0;
            }
            // printf("%.6f ", vector[i]);
        }
    }
    return vector;
  }

double *space_modulate(const double *input, int length) {
    double *vector = malloc(length * sizeof(double));
    if (!vector) {
      perror("Memory allocation failed in getRangeOfVector");
      exit(EXIT_FAILURE);
    }
    for (int i = 0; i < length; i++) {
        vector[i] = (input[i] == 1) ? 1 : -1;
        // printf("%.6f ", vector[i]);
      }
    return vector;
}

double *add_noise(const double *input, int length) {
    double *vector = malloc(length * sizeof(double));
    if (!vector) {
      perror("Memory allocation failed in getRangeOfVector");
      exit(EXIT_FAILURE);
    }
    for (int i = 0; i < length; i++) {
        double noise = sin(input[i]);
        vector[i] = input[i] + noise;
        // printf("%.6f ", vector[i]);
    }
    return vector;
}

double *space_demodulate(const double *input, int length) {
    double *vector = malloc(length * sizeof(double));
    if (!vector) {
      perror("Memory allocation failed in getRangeOfVector");
      exit(EXIT_FAILURE);
    }
    for (int i = 0; i < length; i++) {
        vector[i] = (input[i] > 0) ? 1 : 0;
        // printf("%.6f ", vector[i]);
    }
    return vector;
}

double *error_correction(const double *data, int length) {
    double *corrected = malloc(length * sizeof(double));
    if (!corrected) {
        perror("Memory allocation failed for corrected");
        exit(EXIT_FAILURE);
    }
  
    int corrected_index = 0;
    for (int i = 0; i < length; i += 8) {
        int count = 0;
        for (int j = 0; j < 8 && (i + j) < length; j++) {  // Ensure within bounds
            if (data[i + j] == 1)  
                count++;
        }
  
        if (count % 2 == 0) {
            // Copy the original 8-bit chunk if parity is even
            for (int j = 0; j < 8 && (i + j) < length; j++) {
                corrected[corrected_index + j] = data[i + j];
                // printf("%.6f ", corrected[i]);
            }
        } else {
            // If parity is odd, correct the first bit by setting it to 0
            corrected[corrected_index] = 0;
            for (int j = 1; j < 8 && (i + j) < length; j++) {
                corrected[corrected_index + j] = data[i + j];
                // printf("%.6f ", corrected[i]);
            }
        }
        corrected_index += 8;
    }
    return corrected;
}

void print_array(const double *arr, int length) {
    for (int i = 0; i < length; i++) {
        printf("%.6f", arr[i]); // Print each element with 6 decimal places
        if (i < length - 1) {
            printf(", "); // Add comma between elements except the last one
        }
    }
}

int main() {
    double *input = getRangeOfVector(0, INPUT_LENGTH, 0.000125);
    if (!input) {
      perror("Memory allocation failed for input");
      return EXIT_FAILURE;
    }
  
    double pi = 3.14159265359;
    double f_sig = 500;
    double getMultiplier = 2 * pi * f_sig;
    double threshold = 0.4;
    int returnOrignal = 0;
  
    double *getSinDuration = gain(input, INPUT_LENGTH, getMultiplier);
    double *clean_sig      = Sin(getSinDuration, INPUT_LENGTH);
    double *binary_sig     = thresholdUp(clean_sig, INPUT_LENGTH, threshold, returnOrignal);
    double *a              = space_modulate(binary_sig, INPUT_LENGTH);
    double *noisy_signal   = add_noise(a, INPUT_LENGTH);
    double *b              = space_demodulate(noisy_signal, INPUT_LENGTH);
    double *e              = error_correction(b, INPUT_LENGTH);
    printf("%f", e[8]);

    // Free allocated memory
    free(input);
    free(getSinDuration);
    free(clean_sig);
    free(binary_sig);
    free(a);
    free(noisy_signal);
    free(b);
    free(e);

    return 0;
}