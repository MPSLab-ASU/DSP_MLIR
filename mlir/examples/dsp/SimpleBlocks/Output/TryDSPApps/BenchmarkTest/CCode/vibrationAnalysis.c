#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define PI 3.14159265359
#define INPUT_LENGTH 1000

// Function prototypes
double *getRangeOfVector(double start, int length, double increment);
void gain(double *output, const double *input, double multiplier, int length);
void sine(double *output, const double *input, int length);
void add(double *output, const double *input1, const double *input2,
         int length);
void delay(double *output, const double *input, int delaySamples, int length);
void dft(double complex *output, const double *input, int length);
void square(double *output, const double *input, int length);
double sum(const double *input, int length);
void threshold(double *output, const double *input, double thresholdValue,
               int length);
void sqrt_array(double *output, const double *input, int length);

// Function implementations
double *getRangeOfVector(double start, int length, double increment) {
  double *vector = malloc(length * sizeof(double));
  if (!vector) {
    perror("Memory allocation failed in getRangeOfVector");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < length; i++) {
    vector[i] = start + i * increment;
  }
  return vector;
}

void gain(double *output, const double *input, double multiplier, int length) {
  for (int i = 0; i < length; i++) {
    output[i] = input[i] * multiplier;
  }
}

void sine(double *output, const double *input, int length) {
  for (int i = 0; i < length; i++) {
    output[i] = sin(input[i]);
  }
}

void add(double *output, const double *input1, const double *input2,
         int length) {
  for (int i = 0; i < length; i++) {
    output[i] = input1[i] + input2[i];
  }
}

void delay(double *output, const double *input, int delaySamples, int length) {
  for (int i = 0; i < length; i++) {
    if (i < delaySamples) {
      output[i] = 0;
    } else {
      output[i] = input[i - delaySamples];
    }
  }
}

void dft(double complex *output, const double *input, int length) {
  for (int k = 0; k < length; k++) {
    output[k] = 0;
    for (int n = 0; n < length; n++) {
      double angle = 2 * PI * k * n / length;
      output[k] += input[n] * cexp(-I * angle);
    }
  }
}

void square(double *output, const double *input, int length) {
  for (int i = 0; i < length; i++) {
    output[i] = input[i] * input[i];
  }
}

double sum(const double *input, int length) {
  double total = 0;
  for (int i = 0; i < length; i++) {
    total += input[i];
  }
  return total;
}

void threshold(double *output, const double *input, double thresholdValue,
               int length) {
  for (int i = 0; i < length; i++) {
    if (input[i] <= -thresholdValue || input[i] >= thresholdValue) {
      output[i] = input[i];
    } else {
      output[i] = 0;
    }
  }
}
void sqrt_array(double *output, const double *input, int length) {
  for (int i = 0; i < length; i++) {
    output[i] = sqrt(input[i]);
  }
}

int main() {
  int fs = 1000;
  double *input = getRangeOfVector(0, INPUT_LENGTH, 0.000125);

  double getMultiplier = 2 * PI * 50;
  double *getSinDuration = malloc(INPUT_LENGTH * sizeof(double));
  gain(getSinDuration, input, getMultiplier, INPUT_LENGTH);

  double *sig1 = malloc(INPUT_LENGTH * sizeof(double));
  sine(sig1, getSinDuration, INPUT_LENGTH);

  double getMultiplier2 = 2 * PI * 120;
  double *getSinDuration2 = malloc(INPUT_LENGTH * sizeof(double));
  gain(getSinDuration2, input, getMultiplier2, INPUT_LENGTH);

  double *sinsig2 = malloc(INPUT_LENGTH * sizeof(double));
  sine(sinsig2, getSinDuration2, INPUT_LENGTH);

  double *sig2 = malloc(INPUT_LENGTH * sizeof(double));
  gain(sig2, sinsig2, 0.5, INPUT_LENGTH);

  double *signal = malloc(INPUT_LENGTH * sizeof(double));
  add(signal, sig1, sig2, INPUT_LENGTH);

  double *noise = malloc(INPUT_LENGTH * sizeof(double));
  delay(noise, signal, 5, INPUT_LENGTH);

  double *noisy_sig = malloc(INPUT_LENGTH * sizeof(double));
  add(noisy_sig, signal, noise, INPUT_LENGTH);

  double threshold_value = 0.2;

  double complex *dft_output = malloc(INPUT_LENGTH * sizeof(double complex));
  dft(dft_output, noisy_sig, INPUT_LENGTH);

  double *fft_real = malloc(INPUT_LENGTH * sizeof(double));
  double *fft_img = malloc(INPUT_LENGTH * sizeof(double));
  for (int i = 0; i < INPUT_LENGTH; i++) {
    fft_real[i] = creal(dft_output[i]);
    fft_img[i] = cimag(dft_output[i]);
  }

  double *sq_abs = malloc(INPUT_LENGTH * sizeof(double));
  double *temp_real = malloc(INPUT_LENGTH * sizeof(double));
  double *temp_img = malloc(INPUT_LENGTH * sizeof(double));
  square(temp_real, fft_real, INPUT_LENGTH);
  square(temp_img, fft_img, INPUT_LENGTH);
  add(sq_abs, temp_real, temp_img, INPUT_LENGTH);
  double *magnitude = malloc(INPUT_LENGTH * sizeof(double));
  sqrt_array(magnitude, sq_abs, INPUT_LENGTH);
  double *GetThresholdReal = malloc(INPUT_LENGTH * sizeof(double));
  threshold(GetThresholdReal, magnitude, threshold_value, INPUT_LENGTH);

  printf("%f ", GetThresholdReal[3]);

  // Free allocated memory
  free(input);
  free(getSinDuration);
  free(sig1);
  free(getSinDuration2);
  free(sinsig2);
  free(sig2);
  free(signal);
  free(noise);
  free(noisy_sig);
  free(dft_output);
  free(fft_real);
  free(fft_img);
  free(sq_abs);
  free(temp_real);
  free(temp_img);
  free(GetThresholdReal);

  return 0;
}