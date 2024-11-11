% Function to generate voice signature (sinusoidal wave with two frequencies)
function signal = generateVoiceSignature(freq1, freq2, duration, sample_rate)
    t = linspace(0, duration, sample_rate * duration);
    signal = sin(2 * pi * freq1 * t) + cos(2 * pi * freq2 * t);
end

% Function to compute the dot product (correlation) between two signals
function result = correlate(signal1, signal2)
    result = sum(signal1 .* signal2);
end

% Main function
function main()
    % Sample rate and duration
    sample_rate = 1000;
    duration = 1;
    
    % Generate voice signatures for Alice, Bob, Charlie
    person1 = generateVoiceSignature(100, 200, duration, sample_rate); % Alice
    person2 = generateVoiceSignature(150, 250, duration, sample_rate); % Bob
    person3 = generateVoiceSignature(120, 180, duration, sample_rate); % Charlie
    
    % Generate an unknown signal (Bob's signature in this case)
    unknown_signal = generateVoiceSignature(150, 250, duration, sample_rate); % Change this to test
    
    % Correlate unknown signal with each person's signature
    max1 = correlate(person1, unknown_signal);
    max2 = correlate(person2, unknown_signal);
    max3 = correlate(person3, unknown_signal);
    
    % Store correlation results
    total_maxes = [max1, max2, max3];
    
    % Find the index of the maximum correlation result
    [max_value, max_index] = max(total_maxes);
    
    % Output results
    fprintf('Max Index: %d\n', max_index);
    fprintf('Max Value: %f\n', max_value);
    fprintf('Correlation with Alice: %f\n', max1);
    fprintf('Correlation with Bob: %f\n', max2);
    fprintf('Correlation with Charlie: %f\n', max3);
end

% Call the main function
main();
