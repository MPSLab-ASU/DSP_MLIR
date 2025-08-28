% Constants
INPUT_LENGTH = 1000;
NLEVELS = 16;
MIN = 0.0;
MAX = 8.0;
THRESHOLD_VAL = 4.0;


function output = getRangeOfVector(start, noOfSamples, increment)
    output = zeros(1, noOfSamples);
    for i = 1:noOfSamples
        output(i) = start + (i - 1) * increment;
    end
end


function output = dft(input, length)
    output = zeros(1, length);
    for k = 0:length-1
        sum = 0;
        for n = 0:length-1
            angle = 2 * pi * k * n / length;
            sum = sum + input(n+1) * exp(-1j * angle);
        end
        output(k+1) = sum;
    end
end

function output = threshold(input, thresh, length)
    output = zeros(1, length);
    for i = 1:length
        if abs(input(i)) >= thresh
            output(i) = input(i);
        else
            output(i) = 0;
        end
    end
end

function output = quantization(input, nlevels, max, min, length)
    stepSize = (max - min) / nlevels;
    output = zeros(1, length);
    for i = 1:length
        level = (input(i) - min) / stepSize;
        roundedLevel = floor(level);
        output(i) = roundedLevel * stepSize + min;
    end
end


function output = runLenEncoding(input, length)
    output = zeros(1, 2 * length);
    k = 1;          % MATLAB index starts at 1
    count = 1;
    output(k) = input(1);
    half_len = length;
    for i = 2:length
        if input(i) == input(i-1)
            count = count + 1;
        else
            output(k + half_len) = count;
            k = k + 1;
            output(k) = input(i);
            count = 1;
        end
    end
    output(k + half_len) = count;
end


function elem = getElemAtIndx(rle, indx)
    elem = rle(indx);
end


input = getRangeOfVector(0, INPUT_LENGTH, 1);

fft = dft(input, INPUT_LENGTH);

GetThresholdReal = real(fft);
GetThresholdImg = imag(fft);

GetThresholdReal = threshold(GetThresholdReal, THRESHOLD_VAL, INPUT_LENGTH);
GetThresholdImg = threshold(GetThresholdImg, THRESHOLD_VAL, INPUT_LENGTH);

QuantOutReal = quantization(GetThresholdReal, NLEVELS, MAX, MIN, INPUT_LENGTH);
QuantOutImg = quantization(GetThresholdImg, NLEVELS, MAX, MIN, INPUT_LENGTH);

rLEOutReal = runLenEncoding(QuantOutReal, INPUT_LENGTH);
rLEOutImg = runLenEncoding(QuantOutImg, INPUT_LENGTH);

final1 = getElemAtIndx(rLEOutReal, 1);
final2 = getElemAtIndx(rLEOutImg, 2);

fprintf('%f\t', final1);
fprintf('%f\n', final2);

