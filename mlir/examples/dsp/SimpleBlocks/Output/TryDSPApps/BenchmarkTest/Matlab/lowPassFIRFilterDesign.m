% Constants
INPUT_LENGTH = 100000000;
FS = 8000;
FC1 = 500;
FC2 = 600;
FC3 = 1000;

% Calculate normalized frequencies
wc1 = 2 * pi * FC1 / FS;
wc2 = 2 * pi * FC2 / FS;
wc3 = 2 * pi * FC3 / FS;

% Create Hamming window
hamming_window = hamming(INPUT_LENGTH);

% Create high-pass filters
hpf1 = highPassFIRFilter(wc1, INPUT_LENGTH);
hpf2 = highPassFIRFilter(wc2, INPUT_LENGTH);
hpf3 = highPassFIRFilter(wc3, INPUT_LENGTH);

% Element-wise multiplication with Hamming window
hpf_w1 = hpf1 .* hamming_window';
hpf_w2 = hpf2 .* hamming_window';
hpf_w3 = hpf3 .* hamming_window';

% Get specific elements
final1 = hpf_w1(7);  
final2 = hpf_w2(8);
final3 = hpf_w3(9);

% Display results
fprintf('%f\n', final1);
fprintf('%f\n', final2);
fprintf('%f\n', final3);

% High-pass FIR filter function
function h = highPassFIRFilter(wc, filterLength)
    n = 0:(filterLength-1);
    mid = (filterLength-1) / 2;
    h = zeros(1, filterLength);
    
    % Use logical indexing to avoid issues with non-integer indices
    midIndex = (n ~= mid);
    h(midIndex) = -sin(wc * (n(midIndex) - mid)) ./ (pi * (n(midIndex) - mid));
    
    % Handle the middle point separately
    h(floor(mid)+1) = 1 - (wc / pi);
end