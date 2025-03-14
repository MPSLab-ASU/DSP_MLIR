#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define INPUT_LENGTH 10
#define NLEVELS 16
#define MIN 0.0
#define MAX 8.0
#define THRESHOLD_VAL 4.0

double *getRangeOfVector(double start, int noOfSamples, double increment) {
  double *output = malloc(noOfSamples * sizeof(double));
  if (!output) {
    perror("Memory allocation failed in getRangeOfVector");
    exit(EXIT_FAILURE);
  }

  for (int i = 0; i < noOfSamples; i++) {
    output[i] = start + i * increment;
  }

  return output;
}

void dft(double complex *output, const double *input, int length) {
  for (int k = 0; k < length; k++) {
    output[k] = 0;
    for (int n = 0; n < length; n++) {
      double angle = 2 * M_PI * k * n / length;
      output[k] += input[n] * cexp(-I * angle);
    }
  }
}

void threshold(double *output, const double *input, double thresh, int length) {
  for (int i = 0; i < length; i++) {
    output[i] = (fabs(input[i]) >= thresh) ? input[i] : 0;
  }
}

void quantization(double *output, const double *input, int nlevels, double max,
                  double min, int length) {
  double stepSize = (max - min) / nlevels;

  for (int i = 0; i < length; ++i) {
    double level = (input[i] - min) / stepSize;
    int roundedLevel = (unsigned int)(level);
    output[i] = roundedLevel * stepSize + min;
  }
}

void runLenEncoding(double *output, const double *input, int length) {
  int k = 0;     // Index for storing encoded values
  int count = 1; // Initialize count

  output[k] = input[0];      // First value
  int half_len = length / 2; // Output array's second half starts here

  for (int i = 1; i < half_len; i++) {
    if (input[i] == input[i - 1]) {
      count++; // Increase count if value is the same as previous
    } else {
      output[k + half_len] = count; // Store count at second half
      k++;                          // Move to next unique value position
      output[k] = input[i];         // Store new value
      count = 1;                    // Reset count
    }
  }
  output[k + half_len] = count; // Store count of last element
}

double getElemAtIndx(const double *rle, int indx) { return rle[indx]; }

int main() {
  double *input = getRangeOfVector(0, INPUT_LENGTH, 1);

  double complex *fft = malloc(INPUT_LENGTH * sizeof(double complex));
  if (!fft) {
    perror("Memory allocation failed");
    free(input);
    return EXIT_FAILURE;
  }

  dft(fft, input, INPUT_LENGTH);

  double *GetThresholdReal = malloc(INPUT_LENGTH * sizeof(double));
  double *GetThresholdImg = malloc(INPUT_LENGTH * sizeof(double));
  if (!GetThresholdReal || !GetThresholdImg) {
    perror("Memory allocation failed");
    free(input);
    free(fft);
    free(GetThresholdReal);
    free(GetThresholdImg);
    return EXIT_FAILURE;
  }

  for (int i = 0; i < INPUT_LENGTH; i++) {
    GetThresholdReal[i] = creal(fft[i]);
    GetThresholdImg[i] = cimag(fft[i]);
  }

  threshold(GetThresholdReal, GetThresholdReal, THRESHOLD_VAL, INPUT_LENGTH);
  threshold(GetThresholdImg, GetThresholdImg, THRESHOLD_VAL, INPUT_LENGTH);

  double *QuantOutReal = malloc(INPUT_LENGTH * sizeof(double));
  double *QuantOutImg = malloc(INPUT_LENGTH * sizeof(double));
  if (!QuantOutReal || !QuantOutImg) {
    perror("Memory allocation failed");
    free(input);
    free(fft);
    free(GetThresholdReal);
    free(GetThresholdImg);
    free(QuantOutReal);
    free(QuantOutImg);
    return EXIT_FAILURE;
  }

  quantization(QuantOutReal, GetThresholdReal, NLEVELS, MAX, MIN, INPUT_LENGTH);
  quantization(QuantOutImg, GetThresholdImg, NLEVELS, MAX, MIN, INPUT_LENGTH);

  double *rLEOutReal = (double *)malloc(2 * INPUT_LENGTH * sizeof(double));
  double *rLEOutImg = (double *)malloc(2 * INPUT_LENGTH * sizeof(double));

  runLenEncoding(rLEOutReal, QuantOutReal, INPUT_LENGTH);
  runLenEncoding(rLEOutImg, QuantOutImg, INPUT_LENGTH);

  double final1 = getElemAtIndx(rLEOutReal, 0);
  double final2 = getElemAtIndx(rLEOutImg, 1);
  printf("%f\t", final1);
  printf("%f", final2);

  free(input);
  free(fft);
  free(GetThresholdReal);
  free(GetThresholdImg);
  free(QuantOutReal);
  free(QuantOutImg);
  free(rLEOutReal);
  free(rLEOutImg);

  return 0;
}