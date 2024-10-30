% Constants
INPUT_LENGTH = 100000000;
NLEVELS = 16;
MIN = 0.0;
MAX = 8.0;
THRESHOLD_VAL = 4.0;

% Function to get range of vector
function output = getRangeOfVector(start, noOfSamples, increment)
    output = start + (0:noOfSamples-1) * increment;
end

% DFT function
function output = dft(input)
    N = length(input);
    n = 0:N-1;
    k = n';
    M = exp(-1j * 2 * pi * k * n / N);
    output = M * input(:);
end

% Threshold function
function output = threshold(input, thresh)
    output = input .* (abs(input) >= thresh);
end

% Quantization function
function output = quantization(input, nlevels, max, min)
    step = (max - min) / nlevels;
    output = round((input - min) / step) * step + min;
end

% Run Length Encoding function
function [rle, rleLength] = runLenEncoding(input)
    diffs = diff([input(:); NaN]);
    runs = find(diffs ~= 0);
    lengths = diff([0; runs]);
    values = input(runs);
    rle = [values, lengths];
    rle = rle';
    rle = rle(:);
    rleLength = length(rle);
end

% Get element at index function
function elem = getElemAtIndx(rle, indx)
    elem = rle(indx);
end

% Main script
input = getRangeOfVector(0, INPUT_LENGTH, 1);

fft_result = dft(input);

GetThresholdReal = real(fft_result);
GetThresholdImg = imag(fft_result);

GetThresholdReal = threshold(GetThresholdReal, THRESHOLD_VAL);
GetThresholdImg = threshold(GetThresholdImg, THRESHOLD_VAL);

QuantOutReal = quantization(GetThresholdReal, NLEVELS, MAX, MIN);
QuantOutImg = quantization(GetThresholdImg, NLEVELS, MAX, MIN);

[rLEOutReal, rleLengthReal] = runLenEncoding(QuantOutReal);
[rLEOutImg, rleLengthImg] = runLenEncoding(QuantOutImg);

final1 = getElemAtIndx(rLEOutReal, 2);
final2 = getElemAtIndx(rLEOutImg, 1);

fprintf('%f\n', final1);
fprintf('%f\n', final2);