import os
import subprocess
import time
import re
import sys
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
mcc_path ="/home/local/ASURITE/apkhedka/Matlab_Installation/bin/mcc"
mrt_path ="/home/local/ASURITE/apkhedka/Matlab_Runtime/R2024b/"
# OutputPath = BasePathForLLVM + "mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/Results/TryResultScript/Output/"
input_file_name = sys.argv[1]
full_path = os.path.abspath(__file__)

# Find the path up to 'DSP_MLIR'
if 'DSP_MLIR' in full_path:
    BasePathForLLVM = full_path.split('DSP_MLIR')[0] + 'DSP_MLIR/'
OutputScriptPath = "mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/BenchmarkTest/Matlab/"
# OutputPath = BasePathForLLVM + "mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/Results/TryResultScript/Output/"
input_file_path = BasePathForLLVM + OutputScriptPath + input_file_name

print(f"Running Application {input_file_path}")
# Construct full output path
if sys.argv[2]:
    OutputPath = os.path.join(BasePathForLLVM, OutputScriptPath, "Output", sys.argv[2])

else:
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
    "10": 10,
    "100": 100,
    "500": 500,
    "1K": 1000,
    "2K": 2000,
    "5K": 5000,
    "10K": 10000,
    "20K": 20000,
    "30K": 30000,
    "40K": 40000,
    "50K": 50000,
    "100K": 100000,
    "1M": 1000000,
    # "10M": 10000000,
    # "20M": 20000000,
    # "30M": 30000000,
    # "40M": 40000000,
    # "50M": 50000000,
    # "100M": 100000000,
    # "1B": 1000000000
}

if sys.argv[1] == "noiseCancellation.m":
    inputValues = {
        "10": 10,
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
    }

elif sys.argv[1] == "echoCancellation.m":
    inputValues = {
        "10": 10,
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
    }

elif sys.argv[1] == "periodogram.m":
    inputValues = {
        "10": 10,
        "100": 100,
        "1K": 1000,
        "10K": 10000,
        "20K": 20000,
    }

elif sys.argv[1] == "lowPassFiltering.m":
    inputValues = {
        "10": 10,
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
    }

elif sys.argv[1] == "hearingAid.m":
    inputValues = {
        "10": 10,
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
    }

elif sys.argv[1] == "FIRFilterDesign.m":
    inputValues = {
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
    }

elif sys.argv[1] == "spectralAnalysis.m":
    inputValues = {
        "10": 10,
        "100": 100,
        "1K": 1000,
        "10K": 10000,
        "20K": 20000,
        "30K": 30000,
        "40K": 40000
    }

elif sys.argv[1] == "audioEqualization.m":
    inputValues = {
        "10": 10,
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
    }

elif sys.argv[1] == "audioCompression.m":
    inputValues = {
        "10": 10,
        "100": 100,
        "1K": 1000,
        "10K": 10000,
        "20K": 20000,
        "30K": 30000,
        "40K": 40000,
    }

elif sys.argv[1] == "vibrationAnalysis.m":
    inputValues = {
        "10": 10,
        "100": 100,
        "1K": 1000,
        "10K": 10000,
        "20K": 20000,
        "30K": 30000,
        "40K": 40000,
        "50K": 50000,
    }

elif sys.argv[1] == "underWaterCommunication.m":
    inputValues = {
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
    }

elif sys.argv[1] == "voiceActivityDetection.m":
    inputValues = {
        "10": 10,
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
    }

elif sys.argv[1] == "signalSmoothing.m":
    inputValues = {
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
    }

elif sys.argv[1] == "targetDetection.m":
    inputValues = {
        "10": 10,
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
    }

elif sys.argv[1] == "biomedicalSignalProcessing.m":
    inputValues = {
        "10": 10,
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
    }

elif sys.argv[1] == "digitalModulation.m":
    inputValues = {
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
    }

elif sys.argv[1] == "spaceCommunication.m":
    inputValues = {
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
    }

elif sys.argv[1] == "radarSignalProcessing.m":
    inputValues = {
        "10": 10,
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
    }

elif sys.argv[1] == "dtmfDetection.m":
    inputValues = {
        "100": 100,
        "1K": 1000,
        "10K": 10000,
        "20K": 20000,
        "30K": 30000,
        "40K": 40000,
        "50K": 50000,
    }

elif sys.argv[1] == "speakerIdentification.m":
    inputValues = {
        "100": 100,
        "1K": 1000,
        "10K": 10000,
        "20K": 20000,
        "30K": 30000,
        "40K": 40000,
        "50K": 50000,
        "100K": 100000,
        "1M": 1000000,
    }

NoOfIterations = 30

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

    command = f"{mcc_path} -m {input_file_path} -d 'Output/' -o {sys.argv[2]}{key}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Modify the generated shell script
    script_path = f"./Output/run_{sys.argv[2]}{key}.sh"
    # Modify the generated shell script
    script_path = f"./Output/run_{sys.argv[2]}{key}.sh"
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

        command2 = f"taskset -c 21 ./Output/run_{sys.argv[2]}{key}.sh {mrt_path}"

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


