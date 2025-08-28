INPUT_LENGTH = 101;
fs = 8000;
fc1 = 500;
fc2 = 600;
fc3 = 1000;
fc4 = 1200;

wc1 = 2 * pi * fc1 / fs;
wc2 = 2 * pi * fc2 / fs;
wc3 = 2 * pi * fc3 / fs;
wc4 = 2 * pi * fc4 / fs;

hamming_window = hammingWindow(INPUT_LENGTH);

hpf1 = highPassFIRFilter(wc1, INPUT_LENGTH);
hpf2 = highPassFIRFilter(wc2, INPUT_LENGTH);
hpf3 = highPassFIRFilter(wc3, INPUT_LENGTH);
hpf4 = highPassFIRFilter(wc4, INPUT_LENGTH);

hpf_w1 = elementWiseMultiply(hpf1, hamming_window);
hpf_w2 = elementWiseMultiply(hpf2, hamming_window);
hpf_w3 = elementWiseMultiply(hpf3, hamming_window);
hpf_w4 = elementWiseMultiply(hpf4, hamming_window);

final1 = getElemAtIndex(hpf_w1, 6);
final2 = getElemAtIndex(hpf_w2, 7);
final3 = getElemAtIndex(hpf_w3, 8);

fprintf('%.6f\n', final1);
fprintf('%.6f\n', final2);
fprintf('%.6f\n', final3);

function window = hammingWindow(length)
    n = 0:length-1;
    window = 0.54 - 0.46 * cos(2 * pi * n / (length - 1));
end

function filter = highPassFIRFilter(wc, length)
    mid = (length - 1) / 2;
    filter = zeros(1, length);
    for n = 0:length-1
        if n == mid
            filter(n+1) = 1 - (wc / pi);
        else
            filter(n+1) = -sin(wc * (n - mid)) / (pi * (n - mid));
        end
    end
end

function output = elementWiseMultiply(array1, array2)
    output = array1 .* array2;
end

function value = getElemAtIndex(array, index)
    value = array(index + 1);
end