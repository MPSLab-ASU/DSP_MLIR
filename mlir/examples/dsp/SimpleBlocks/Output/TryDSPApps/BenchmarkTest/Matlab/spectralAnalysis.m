% Constants
INPUT_LENGTH = 100000000;

% getRange function
input = getRange(0, INPUT_LENGTH, 1);

% DFT function (using built-in FFT)
fft_result = fft(input);

% Square of absolute values
sq_abs = abs(fft_result).^2;

% Sum and average
res = mean(sq_abs);

% Display result
fprintf('%f\n', res);

%  getRange function
function output = getRange(start, noOfSamples, increment)
    output = start + (0:noOfSamples-1) * increment;
end