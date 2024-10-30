#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUT_LENGTH 100000000

void getRangeOfVector(double *vector, double start, int length,
                      double increment) {
  for (int i = 0; i < length; i++) {
    vector[i] = start + i * increment;
  }
}

void gain(double *output, double *input, double multiplier, int length) {
  for (int i = 0; i < length; i++) {
    output[i] = input[i] * multiplier;
  }
}

void sine(double *output, double *input, int length) {
  for (int i = 0; i < length; i++) {
    output[i] = sin(input[i]);
  }
}

void add(double *output, double *input1, double *input2, int length) {
  for (int i = 0; i < length; i++) {
    output[i] = input1[i] + input2[i];
  }
}

void lmsFilterResponse(double *output, double *noisy_sig, double *clean_sig,
                       double mu, int filterSize, int length) {
  double w[32] = {0};
  for (int n = 0; n < length; n++) {
    double y = 0;
    for (int i = 0; i < filterSize; i++) {
      if (n - i >= 0) {
        y += w[i] * noisy_sig[n - i];
      }
    }
    double e = clean_sig[n] - y;
    for (int i = 0; i < filterSize; i++) {
      if (n - i >= 0) {
        w[i] += mu * e * noisy_sig[n - i];
      }
    }
    output[n] = y;
  }
}

int main() {
  // Allocate memory dynamically
  double *t = (double *)malloc(INPUT_LENGTH * sizeof(double));
  double *getSinDuration = (double *)malloc(INPUT_LENGTH * sizeof(double));
  double *clean_sig = (double *)malloc(INPUT_LENGTH * sizeof(double));
  double *getNoiseSinDuration = (double *)malloc(INPUT_LENGTH * sizeof(double));
  double *noise = (double *)malloc(INPUT_LENGTH * sizeof(double));
  double *noise1 = (double *)malloc(INPUT_LENGTH * sizeof(double));
  double *noisy_sig = (double *)malloc(INPUT_LENGTH * sizeof(double));
  double *y = (double *)malloc(INPUT_LENGTH * sizeof(double));
  double *sol = (double *)malloc(INPUT_LENGTH * sizeof(double));

  // Check if memory allocation was successful
  if (!t || !getSinDuration || !clean_sig || !getNoiseSinDuration || !noise ||
      !noise1 || !noisy_sig || !y || !sol) {
    perror("Memory allocation failed");
    free(t);
    free(getSinDuration);
    free(clean_sig);
    free(getNoiseSinDuration);
    free(noise);
    free(noise1);
    free(noisy_sig);
    free(y);
    free(sol);
    exit(EXIT_FAILURE);
  }

  // Signal processing steps
  getRangeOfVector(t, 0, INPUT_LENGTH, 0.000125);

  double f_sig = 500;
  double pi = 3.14159265359;
  gain(getSinDuration, t, 2 * pi * f_sig, INPUT_LENGTH);

  sine(clean_sig, getSinDuration, INPUT_LENGTH);

  double f_noise = 3000;
  gain(getNoiseSinDuration, t, 2 * pi * f_noise, INPUT_LENGTH);

  sine(noise, getNoiseSinDuration, INPUT_LENGTH);

  gain(noise1, noise, 0.5, INPUT_LENGTH);

  add(noisy_sig, clean_sig, noise1, INPUT_LENGTH);

  // LMS filter response
  lmsFilterResponse(y, noisy_sig, clean_sig, 0.01, 32, INPUT_LENGTH);

  gain(sol, y, 10, INPUT_LENGTH);

  // Print the filtered signal
  for (int i = 0; i < INPUT_LENGTH; i++) {
    printf("%f\n", sol[i]);
  }

  // Free allocated memory at the end
  free(t);
  free(getSinDuration);
  free(clean_sig);
  free(getNoiseSinDuration);
  free(noise);
  free(noise1);
  free(noisy_sig);
  free(y);
  free(sol);

  return 0;
}