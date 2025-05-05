import os
import subprocess
import time
import sys
import math
# The script does the following
# Input : filename.c
# Output : TimeOfExecution for different IP sizes :
# Steps to run:
# Open a terminal at the path of the script --
# Run: python ScriptForCases.c #3.11 validated

# Pseudo-code:
# Iterate for all the input-size & update the input value in file
# Update logic -- change the 2nd parameter of line: var c = getRangeOfVector(init , Count, StepSize)
# Run the respective commands on the file

# Path to the input file
# Apps = "lowPassFIRFilterDesign.c", "noisecancelling.c" , "echocancelling.c",  "hearingAid.c", "audioEqualizer.c", "vibrationAnalysis.c", "underWaterCommunication.c", "voiceActivityDetection.c", "signalSmoothing",  "targetDetection", "biomedicalSignalProcessing", "periodogram2Conv", "spaceCommunication", "dtmfDetection", "speakerIdentification"
input_file_name = sys.argv[1]
full_path = os.path.abspath(__file__)

# Find the path up to 'DSP_MLIR'
if 'DSP_MLIR' in full_path:
    BasePathForLLVM = full_path.split('DSP_MLIR')[0] + 'DSP_MLIR/'
OutputScriptPath = "mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/BenchmarkTest/CCode/"
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

    
    #"10M": 10000000,
    #"20M": 20000000,
    #"30M": 30000000,
    #"40M": 40000000,
    #"50M": 50000000,
    #"100M": 100000000,
    # "1B": 1000000000
}

if sys.argv[1] == "noiseCancellation.c":
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
        }
    
elif sys.argv[1] == "echoCancellation.c":
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
        }

elif sys.argv[1] == "periodogram.c":
    inputValues = {
        "10": 10,
        "100": 100,
        "500": 500,
        "1K": 1000,
        }

elif sys.argv[1] == "lowPassFiltering.c":
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
        }


elif sys.argv[1] == "hearingAid.c":
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
        }

elif sys.argv[1] == "FIRFilterDesign.c":
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
        }


elif sys.argv[1] == "spectralAnalysis.c":
    inputValues = {
        "10": 10,
        "100": 100,
        "500": 500,
        "1K": 1000,
        "2K": 2000,
        }


elif sys.argv[1] == "audioEqualization.c":
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
        }


elif sys.argv[1] == "audioCompression.c":
    inputValues = {
        "10": 10,
        "100": 100,
        "500": 500,
        "1K": 1000,
        "2K": 2000,
        }


elif sys.argv[1] == "vibrationAnalysis.c":
    inputValues = {
        "10": 10,
        "100": 100,
        "500": 500,
        "1K": 1000,
        "2K": 2000,
        }


elif sys.argv[1] == "underWaterCommunication.c":
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
        }

elif sys.argv[1] == "voiceActivityDetection.c":
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
        }


elif sys.argv[1] == "signalSmoothing.c":
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
        }



elif sys.argv[1] == "targetDetection.c":
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
        }


elif sys.argv[1] == "biomedicalSignalProcessing.c":
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
        }
        

elif sys.argv[1] == "digitalModulation.c":
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
        }



elif sys.argv[1] == "spaceCommunication.c":
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
        }


elif sys.argv[1] == "radarSignalProcessing.c":
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
        }
        
        
elif sys.argv[1] == "speakerIdentification.c":
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
        }


elif sys.argv[1] == "dtmfDetection.c":
    inputValues = {
        "10": 10,
        "100": 100,
        "500": 500,
        "1K": 1000,
        "2K": 2000,
        }
NoOfIterations = 1

# Define the cases
cases = [
    {
        "hexagon-clang": True,
        "llvm-clang": False,
        "exe": "Clang-hexagon",
    },
    {
        "llvm-clang": True,
        "hexagon-clang": False,
        "exe": "Clang",
    },
]


with open(input_file_path, "r") as file:
    lines = file.readlines()

print("", end="\t")

for case in cases:
    print(f"{case['exe']}", end="\t")

for key, value in inputValues.items():
    # Update the specific line in the file
    # print("Updating for {}".format(value))
    print("\n{}".format(key), end="\t")
    with open(input_file_path, "w") as file:
        for line in lines:
            if line.strip().startswith("#define INPUT_LENGTH"):
                if sys.argv[1] == "speakerIdentification.c":
                    updated_line = f"#define INPUT_LENGTH {math.floor(value/8.192)}\n"
                if sys.argv[1] == "FIRFilterDesign.c":
                    updated_line = f"#define INPUT_LENGTH {value +1}\n"
                else:     
                    updated_line = f"#define INPUT_LENGTH {value}\n"
                file.write(updated_line)
            else:
                file.write(line)

    for case in cases:
        
        if case["hexagon-clang"]:
            command = f"/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.2.0.1/tools/HEXAGON_Tools/8.8.06/Tools/bin/hexagon-clang -O3 -o {OutputPath}/{case['exe']} {input_file_path} -lm",
        if case["llvm-clang"]:
            command = f"{BasePathForLLVM}/build/bin/clang-19 -O3 {input_file_path} -o {OutputPath}/{case['exe']} -lm --target=hexagon -mcpu=hexagonv68 -fuse-ld=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.2.0.1/tools/HEXAGON_Tools/8.8.06/Tools/bin/hexagon-link",
        
        result = subprocess.run(command, shell=True, capture_output=True, text=True)

        sum_exe_time = 0
        for i in range(0, NoOfIterations):
            command2 = f"{BasePathForLLVM}/Hexagon_Tools/bin/hexagon-sim --mv68 ./Output/{sys.argv[2]}/{case['exe']}"
            
            # Record the start time
            #start_time = time.time()

            # Execute the command
            try:
                result2 = subprocess.run(
                    command2,
                    shell=True,
                    #stdout=subprocess.DEVNULL,
                    #stderr=subprocess.DEVNULL,
                    #check=True,
                    capture_output = True,
                    text = True,
                    check=True,
                )
                # subprocess.run(command2, shell=True)
                pcycle = int(result2.stderr.split("Pcycles=")[1].replace("\n","").replace(" ","").replace("\t",""))
            except subprocess.CalledProcessError as exc:
                print(
                    f"Process failed because did not return a successful return code. "
                    f"Returned {exc.returncode}\n{exc}"
                )

            #end_time = time.time()
            
            #execution_time = end_time - start_time
            #sum_exe_time = sum_exe_time + execution_time
            sum_exe_time = sum_exe_time + pcycle
        avg_exe_time = sum_exe_time
        print("{}".format(avg_exe_time), end="\t")
