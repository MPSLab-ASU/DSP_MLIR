import subprocess
import fileinput
import re

file_name = "/home/local/ASURITE/apkhedka/MLIR/build/bin/matlab_result.py"


# Define the replacement pattern
replacement_pattern = r"^\s*var c = dct\(a\d+[K]?\);\s*$"

# Values to replace the entire line
input_sizes = ["a10", "a100", "a10K", "a100K"]
# function_list = ["slidingWindowAvg", "sub", "add"]
function_list = ["dct"]
count = 0
# Iterate through the file and replace matching lines with each value
for new_value in input_sizes:
    with fileinput.FileInput(file_name, inplace=True, backup='.bak') as file:
        for line in file:
            # Check if the line matches the pattern
            if re.match(replacement_pattern, line):
                # Replace the line with the new value
                print("var c = {}({});".format(function_list[0], new_value))
            else:
                # Print the original line
                print(line, end='')

    commands = [
        "./dsp1 matlab_result.py -emit=llvm 2> matlab_test.ll",
        "clang-17 matlab_test.ll -o clang_matlab -lm",
        "time ./clang_matlab "
    ]

    # Iterate through the commands and execute them sequentially
    for command in commands:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = process.communicate()
        if command.startswith("time"):
            if output:
                print(output.decode())
            if error:
                print(error.decode())

        # Check if any error occurred
        if process.returncode != 0:
            print("Error occurred while executing command:", command)
            print("Error:", error.decode())
            break
        else:
            continue
