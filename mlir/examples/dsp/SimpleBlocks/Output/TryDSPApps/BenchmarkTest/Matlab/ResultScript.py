import os
import subprocess
import time
import re
# The script does the following
# Input : filename.py
# Output : TimeOfExecution for different IP sizes :
# Steps to run:
# Open a terminal at the path of the script --
# Run: python ScriptForCases.py #3.11 validated

# Pseudo-code:
# Iterate for all the input-size & update the input value in file
# Update logic -- change the 2nd parameter of line: var c = getRangeOfVector(init , Count, StepSize)
# Run the respective commands on the file

# Path to the input file
# Apps = "noiseCancelling.m" , "echoCancelling.m", "periodogram.m", "lowPassFull.m", "hearingAid.m", "lowPassFIRFilterDesign", "energyOfSignal", "audioEqualizer", "audioCompression","vibrationAnalysis", "underWaterCommunication", "voiceActivityDetection", "signalSmoothing", "targetDetection", "biomedicalSignalProcessing", "digitalModulation", "spaceCommunication", "radarSignalProcessing"
input_file = "speakerIdentification"
input_file_path = input_file + ".m"
BasePathForLLVM = "/home/local/ASURITE/apkhedka/ForLLVM/"
OutputScriptPath = "mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/BenchmarkTest/Matlab/"
mcc_path ="/home/local/ASURITE/apkhedka/Matlab_Installation/bin/mcc"
mrt_path ="/home/local/ASURITE/apkhedka/Matlab_Runtime/R2024b/"
# OutputPath = BasePathForLLVM + "mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/Results/TryResultScript/Output/"
print(f"Running Application {input_file_path}")
# Construct full output path
OutputPath = os.path.join(BasePathForLLVM, OutputScriptPath, "Output")

# Check if the Output folder exists, create it if it doesn't
if not os.path.exists(OutputPath):
    os.makedirs(OutputPath)

# Now OutputPath is ready for use
print("InputPath:{}".format(BasePathForLLVM))
print(f"OutputPath: {OutputPath}")
# exit()

# ************ Don't change unless u required
# Define the values dictionary
inputValues = {
    # "10": 10,
    "100": 100,
    "1K": 1000,
    "10K": 10000,
    "20K": 20000,
    "30K": 30000,
    "40K": 40000,
    "50K": 50000,
    "100K": 100000,
     "1M": 1000000,
    "10M": 10000000,
    "20M": 20000000,
    "30M": 30000000,
    "40M": 40000000,
    "50M": 50000000,
    "100M": 100000000,
    # "1B": 1000000000
}
NoOfIterations = 3

def delete_folder_contents(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')


with open(input_file_path, "r") as file:
    lines = file.readlines()

print("", end="\t")


for key, value in inputValues.items():
    # Update the specific line in the file
    # print("Updating for {}".format(value))
    print("\n{}".format(key), end="\t")
    with open(input_file_path, "w") as file:
        for line in lines:
            if line.strip().startswith("INPUT_LENGTH = "):
                updated_line = f"INPUT_LENGTH = {value};\n"
                file.write(updated_line)
            else:
                file.write(line)

    command = f"{mcc_path} -m {input_file_path} -d 'Output/' -o {input_file}{key}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Modify the generated shell script
    script_path = f"./Output/run_{input_file}{key}.sh"
    # Modify the generated shell script
    script_path = f"./Output/run_{input_file}{key}.sh"
    with open(script_path, 'r') as file:
        script_content = file.readlines()

    # Find the line with the eval command and modify it
    for i, line in enumerate(script_content):
        if line.strip().startswith('eval'):
            script_content[i] = f"""  start_time=$(date +%s.%N)
  {line.strip()}
  end_time=$(date +%s.%N)
  execution_time=$(echo "$end_time - $start_time" | bc)
  echo "Execution time: $execution_time"
"""
            break

    # Write the modified content back to the script
    with open(script_path, 'w') as file:
        file.writelines(script_content)


    sum_exe_time = 0
    for i in range(0, NoOfIterations):
        try:
            subprocess.run("sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'", shell=True, check=True)
        except subprocess.CalledProcessError as exc:
            print(exc)

        command2 = f"taskset -c 0 ./Output/run_{input_file}{key}.sh {mrt_path}"

        try:
            result = subprocess.run(command2, shell=True, capture_output=True, text=True, check=True)
            output = result.stdout
            
            # Extract execution time from the output
            match = re.search(r"Execution time: (\d+\.\d+)", output)
            if match:
                execution_time = float(match.group(1))
                sum_exe_time += execution_time
            else:
                print(f"Execution time not found in output: {output}")
        except subprocess.CalledProcessError as exc:
            print(f"Process failed. Returned {exc.returncode}\n{exc}")

    avg_exe_time = sum_exe_time / NoOfIterations
    print(f"{avg_exe_time}", end="\t")
    # delete_folder_contents("./Output")


