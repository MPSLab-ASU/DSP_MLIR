#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PI 3.14159265358979323846
#define INPUT_LENGTH 100000000
// DTMF frequencies
const int dtmf_freqs[12][2] = {
    {697, 1209}, {697, 1336}, {697, 1477}, {770, 1209}, {770, 1336},
    {770, 1477}, {852, 1209}, {852, 1336}, {852, 1477}, {941, 1336}};

const char dtmf_digits[12] = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '0'};

void generate_dtmf_tone(double *signal, char digit, int N, int fs) {
  int f1 = 0, f2 = 0;
  for (int i = 0; i < 12; i++) {
    if (dtmf_digits[i] == digit) {
      f1 = dtmf_freqs[i][0];
      f2 = dtmf_freqs[i][1];
      break;
    }
  }

  for (int i = 0; i < N; i++) {
    double t = (double)i / fs;
    signal[i] = sin(2 * PI * f1 * t) + sin(2 * PI * f2 * t);
  }
}

double goertzel(double *data, int N, double frequency, int fs) {
  double omega = 2.0 * PI * frequency / fs;
  double cosine = cos(omega);
  double coeff = 2.0 * cosine;
  double q0 = 0, q1 = 0, q2 = 0;

  for (int i = 0; i < N; i++) {
    q0 = coeff * q1 - q2 + data[i];
    q2 = q1;
    q1 = q0;
  }

  return sqrt(q1 * q1 + q2 * q2 - coeff * q1 * q2);
}

char detect_dtmf(double *signal, int N, int fs) {
  double magnitudes[8];
  int frequencies[] = {697, 770, 852, 941, 1209, 1336, 1477, 1633};

  // Calculate magnitudes for all 8 DTMF frequencies
  for (int i = 0; i < 8; i++) {
    magnitudes[i] = goertzel(signal, N, frequencies[i], fs);
  }

  // Find the maximum magnitude in low and high frequency groups
  int max_low_index = 0, max_high_index = 4;
  for (int i = 1; i < 4; i++) {
    if (magnitudes[i] > magnitudes[max_low_index])
      max_low_index = i;
  }
  for (int i = 5; i < 8; i++) {
    if (magnitudes[i] > magnitudes[max_high_index])
      max_high_index = i;
  }

  // Calculate the average magnitude
  double avg_magnitude = 0;
  for (int i = 0; i < 8; i++) {
    avg_magnitude += magnitudes[i];
  }
  avg_magnitude /= 8;

  // Set thresholds
  double threshold = avg_magnitude * 2;

  // Check if the detected magnitudes are significantly above the threshold
  if (magnitudes[max_low_index] > threshold &&
      magnitudes[max_high_index] > threshold) {
    return dtmf_digits[max_low_index * 3 + (max_high_index - 4)];
  }

  return '\0';
}

int main() {
  int fs = 8000;
  char test_digit = '5'; // Changed to char type
  double duration = (double)INPUT_LENGTH / fs;
  int N = INPUT_LENGTH;

  double *input = (double *)malloc(N * sizeof(double));
  if (!input) {
    perror("Memory allocation failed for input");
    exit(EXIT_FAILURE);
  }

  generate_dtmf_tone(input, test_digit, N, fs);

  int delay_samples = fs / 100;
  for (int j = N - 1; j >= delay_samples; j--) {
    input[j] = input[j - delay_samples];
  }
  for (int j = 0; j < delay_samples; j++) {
    input[j] = 0;
  }

  char detected_digit = detect_dtmf(input, N, fs);

  if (detected_digit != '\0') {
    printf("Generated: %c, Detected: %c\n", test_digit, detected_digit);
  } else {
    printf("Generated: %c, No DTMF digit detected\n", test_digit);
  }

  free(input);

  return 0;
}