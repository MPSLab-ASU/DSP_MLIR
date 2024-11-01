% Define a function to generate an arbitrary noisy signal
function signal = generate_noisy_signal(length, noise_level)
    % Initialize the signal
    signal = zeros(1, length);
    for i = 1:length
        % Generate a random signal value between -1.0 and 1.0
        random_value = rand() * 2 - 1; % Random value in [-1, 1]
        
        % Add random noise
        noise = (rand() - 0.5) * 2 * noise_level; % Random noise in [-noise_level, noise_level]
        
        % Combine random value with noise
        signal(i) = random_value + noise;
    end
end

% Main script
length = 20; % Length of the original signal
noise_level = 0.3; % Example noise level

% Generate an arbitrary noisy signal
signal = generate_noisy_signal(length, noise_level);

% Print the original signal
fprintf('Original Signal:\n');
fprintf('%.4f ', signal);

% Apply moving average filter with kernel size of 3
avg_filtered = movmean(signal, 3, 'Endpoints', 'discard');

% Print the output after applying the moving average filter
fprintf('After Moving Average Filter:\n');
fprintf('%.4f ', avg_filtered);

% Apply moving median filter to the output of the moving average filter
median_filtered = movmedian(avg_filtered, 3, 'Endpoints', 'discard');

% Print the output after applying the moving median filter
fprintf('After Moving Median Filter:\n');
fprintf('%.4f ', median_filtered);
