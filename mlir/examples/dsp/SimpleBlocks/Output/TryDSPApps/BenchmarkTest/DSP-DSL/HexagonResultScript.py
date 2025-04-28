import os
import subprocess
import time
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
# Apps = "hearingAid.py" , "lowPassFull.py" , " audioCompression.py", "lowPassFIRFilterDesign.py" , "EnergyOfSignal.py", "periodogram2Conv1.py", "audioEqualizer.py", "vibrationAnalysis.py", "signalSmoothing.py", "targetDetection.py", "biomedicalSignalProcessing.py", "spaceCommunication.py", "echocancelling", "noisecancelling.py", "digitalModulation", "underWaterCommunication", "voiceActivityDetection", "radarSignalProcessing", "speakerIdentification"
# input_file_name = "speakerIdentification.py"
input_file_name = sys.argv[1]
full_path = os.path.abspath(__file__)

# Find the path up to 'DSP_MLIR'
if 'DSP_MLIR' in full_path:
    BasePathForLLVM = full_path.split('DSP_MLIR')[0] + 'DSP_MLIR/'
OutputScriptPath = (
    "mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/BenchmarkTest/DSP-DSL/"
)
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

if sys.argv[1] == "noiseCancellation.py":
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
    
elif sys.argv[1] == "echoCancellation.py":
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

elif sys.argv[1] == "periodogram.py":
    inputValues = {
        "10": 10,
        "100": 100,
        "500": 500,
        "1K": 1000,
        }

elif sys.argv[1] == "lowPassFiltering.py":
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


elif sys.argv[1] == "hearingAid.py":
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

elif sys.argv[1] == "FIRFilterDesign.py":
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


elif sys.argv[1] == "spectralAnalysis.py":
    inputValues = {
        "10": 10,
        "100": 100,
        "500": 500,
        "1K": 1000,
        "2K": 2000,
        }


elif sys.argv[1] == "audioEqualization.py":
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


elif sys.argv[1] == "audioCompression.py":
    inputValues = {
        "10": 10,
        "100": 100,
        "500": 500,
        "1K": 1000,
        "2K": 2000,
        }


elif sys.argv[1] == "vibrationAnalysis.py":
    inputValues = {
        "10": 10,
        "100": 100,
        "500": 500,
        "1K": 1000,
        "2K": 2000,
        }


elif sys.argv[1] == "underWaterCommunication.py":
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

elif sys.argv[1] == "voiceActivityDetection.py":
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


elif sys.argv[1] == "signalSmoothing.py":
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



elif sys.argv[1] == "targetDetection.py":
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


elif sys.argv[1] == "biomedicalSignalProcessing.py":
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
        

elif sys.argv[1] == "digitalModulation.py":
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



elif sys.argv[1] == "spaceCommunication.py":
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


elif sys.argv[1] == "radarSignalProcessing.py":
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
        
        
elif sys.argv[1] == "speakerIdentification.py":
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


elif sys.argv[1] == "dtmfDetection.py":
    inputValues = {
        "10": 10,
        "100": 100,
        "500": 500,
        "1K": 1000,
        "2K": 2000,
        }
NoOfIterations = 1

# --------------------------------------------------
commands_base = [
    # "./dsp1 lowPassFull.py -emit=mlir-affine",
    # f"./dsp1 {input_file_path} -emit=llvm",
    f"{BasePathForLLVM}/build/bin/dsp1 {input_file_path} -emit=llvm-hexagonv68",
    # "clang-17 -O0 file.ll -o fileexe -lm",
]

clang = f"{BasePathForLLVM}/build/bin/clang-19 LL_FILE_PATH -O3 -o OUT_FILE_PATH --target=hexagon -mcpu=hexagonv68 -fuse-ld={BasePathForLLVM}/Hexagon_Tools/bin/hexagon-link"

# clang = f"/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.2.0.1/tools/HEXAGON_Tools/8.8.06/Tools/bin/hexagon-clang LL_FILE_PATH -O3 -o OUT_FILE_PATH --target=hexagon -mcpu=hexagonv68 -fuse-ld=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.2.0.1/tools/HEXAGON_Tools/8.8.06/Tools/bin/hexagon-link"

# clang = f"{BasePathForLLVM}/build/bin/clang LL_FILE_PATH -O3 -o OUT_FILE_PATH --target=hexagon -mcpu=hexagonv68 -fuse-ld=/local/mnt/workspace/Qualcomm/Hexagon_SDK/6.2.0.1/tools/HEXAGON_Tools/8.8.06/Tools/bin/hexagon-link"



# Define the cases
cases = [
    # {
    #     "affineOpt": False,
    #     "canonOpt": False,
    #     "suffix": "fileNoOpt.ll",
    #     "exe": "fileNoOptExe",
    # },
    
    #HWISOO: Temporal for debugging
    # {
    #     "affineOpt": False,
    #     "canonOpt": False,
    #     "suffix": "fileNoOpt.ll",
    #     "exe": "fileNoOptExe",
    # },

    {
        "affineOpt": True,
        "canonOpt": False,
        "suffix": "fileAffineOpt.ll",
        "exe": "AffineExe",
    },
    # {
    #     "affineOpt": False,
    #     "canonOpt": True,
    #     "suffix": "fileOnlyCanonOpt.ll",
    #     "exe": "fileOnlyCanonOptExe",
    # },
    {
        "affineOpt": True,
        "canonOpt": True,
        "suffix": "fileAffineCanonOpt.ll",
        "exe": "DSP_MLIRExe",
    },
]



print(input_file_path)

# Read the input file
with open(input_file_path, "r") as file:
    lines = file.readlines()

print("", end="\t")
for case in cases:
    print(f"{case['exe']}", end="\t")
    
for key, value in inputValues.items():
    value2 = 1 / value
    dur = value / 8192
    print(f"\n{key}", end="\t")

    with open(input_file_path, "r") as file:
        lines = file.readlines()

    with open(input_file_path, "w") as file:
        for line in lines:
            if line.strip().startswith("var input = getRangeOfVector("):
                if input_file_name in ["audioCompression.py", "audioEqualization.py",  "periodogram.py", "spectralAnalysis.py"]:
                    updated_line = (
                    f"\tvar input = getRangeOfVector(0, {value}, 1);\n"
                )
                if input_file_name in ["voiceActivityDetection.py"]:
                    updated_line = (
                    f"\tvar input = getRangeOfVector(0, {value}, 0.125);\n"
                )
                else: 
                    updated_line = (
                        f"\tvar input = getRangeOfVector(0, {value}, 0.000125);\n"
                    )
                    
                file.write(updated_line)
            elif line.strip().startswith("var duration ="):
                updated_line = f"\tvar duration = {dur};\n"
                file.write(updated_line)
            elif line.strip().startswith("var N_input ="):
                updated_line = f"\tvar N_input = {value+1};\n"
                file.write(updated_line)
            elif line.strip().startswith("var frequencies = fftfreq"):
                updated_line = f"\tvar frequencies = fftfreq({value}, 0.000122);\n"
                file.write(updated_line)
            else:
                file.write(line)

    

    # Iterate through the cases and run the commands
    for case in cases:
        command_llvm = commands_base[0]
        if case["affineOpt"]:
            command_llvm += " -affineOpt"
        if case["canonOpt"]:
            command_llvm += " -canonOpt"
        # command_llvm += f" 2> {case['suffix']}" #OutputPath
        
        

        
        ll_file_path = f"{OutputPath}/{case['suffix']}"
        command_llvm += f" 2> {OutputPath}/{case['suffix']}"  # OutputPath
        
        out_file_path = ll_file_path.replace(".ll", ".out")
        clang_command = clang.replace("LL_FILE_PATH", ll_file_path)
        clang_command = clang_command.replace("OUT_FILE_PATH", out_file_path)
        

        commands = [
            "rm "+ll_file_path,
            "rm "+out_file_path,
            command_llvm,
            clang_command,
            
        ]

        # Iterate over each value and perform the necessary operations
        for command in commands:
            # Run the commands for the current case
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            
            

        sum_exe_time = 0
        for i in range(0, NoOfIterations):
            #NOTE: for simulation environment, we don't need to take care of cachine impact on host

            # The command to be executed
            # command2 = "./fileexe"
            # Limit execution to a single core
            # command2 = "taskset -c 0 ./fileexe"
            # command2 = f"taskset -c 0 ./{case['exe']}" #{OutputPath}
            # command2 = f"taskset -c 0 ./Output/{case['exe']}"
            
            # command2 = f"/{BasePathForLLVM}/build/bin/hexagon-sim --mv68 OUT_FILE_PATH".replace("OUT_FILE_PATH", out_file_path)

            command2 = f"{BasePathForLLVM}/Hexagon_Tools/bin/hexagon-sim --mv68 OUT_FILE_PATH".replace("OUT_FILE_PATH", out_file_path)
            
            
            

            # Record the start time
            start_time = time.time()


            pcycle = "failed"
            # Execute the command
            try:
                result = subprocess.run(
                    command2,
                    shell=True,
                    #stdout=subprocess.DEVNULL,
                    #stdout="test_stdout",
                    #stderr="test_stderr",
                    #stderr=subprocess.DEVNULL,
                    capture_output = True,
                    text = True,
                    check=True,
                )
                # subprocess.run(command2, shell=True)
                
                pcycle = result.stderr.split("Pcycles=")[1].replace("\n","").replace(" ","").replace("\t","")
            except subprocess.CalledProcessError as exc:
                print(
                    f"Process failed because did not return a successful return code. "
                    f"Returned {exc.returncode}\n{exc}"
                )
            
            

            
            # Record the end time
            end_time = time.time()

            # Calculate the elapsed time
            execution_time = end_time - start_time
            sum_exe_time = sum_exe_time + execution_time
            # print("{}".format(execution_time), end="\t")
        avg_exe_time = sum_exe_time / NoOfIterations
        # print(pcycle + "/" + "{}".format(round(avg_exe_time, 4)), end="\t")
        print(pcycle, end="\t")
        
        
    # print(f"The command took {execution_time} seconds to execute.")
