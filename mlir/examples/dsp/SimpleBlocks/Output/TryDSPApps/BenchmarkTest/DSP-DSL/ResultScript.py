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
# print("InputPath:{}".format(BasePathForLLVM))
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
    # "10M": 10000000,
    # "20M": 20000000,
    # "30M": 30000000,
    # "40M": 40000000,
    # "50M": 50000000,
    # "100M": 100000000,
    # "1B": 1000000000
}

if sys.argv[1] == "noiseCancellation.py":
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

elif sys.argv[1] == "echoCancellation.py":
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

elif sys.argv[1] == "periodogram.py":
    inputValues = {
        "10": 10,
        "100": 100,
        "1K": 1000,
        "10K": 10000,
        "20K": 20000,
    }

elif sys.argv[1] == "lowPassFiltering.py":
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

elif sys.argv[1] == "hearingAid.py":
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

elif sys.argv[1] == "FIRFilterDesign.py":
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

elif sys.argv[1] == "spectralAnalysis.py":
    inputValues = {
        # "10": 10,
        "100": 100,
        "1K": 1000,
        "10K": 10000,
        "20K": 20000,
        "30K": 30000,
    }

elif sys.argv[1] == "audioEqualization.py":
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

elif sys.argv[1] == "audioCompression.py":
    inputValues = {
        "10": 10,
        "100": 100,
        "1K": 1000,
        "10K": 10000,
        "20K": 20000,
        "30K": 30000,
        "40K": 40000,
    }

elif sys.argv[1] == "vibrationAnalysis.py":
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

elif sys.argv[1] == "underWaterCommunication.py":
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

elif sys.argv[1] == "voiceActivityDetection.py":
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

elif sys.argv[1] == "signalSmoothing.py":
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

elif sys.argv[1] == "targetDetection.py":
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

elif sys.argv[1] == "biomedicalSignalProcessing.py":
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

elif sys.argv[1] == "digitalModulation.py":
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

elif sys.argv[1] == "spaceCommunication.py":
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

elif sys.argv[1] == "radarSignalProcessing.py":
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

elif sys.argv[1] == "dtmfDetection.py":
    inputValues = {
        "100": 100,
        "1K": 1000,
        "10K": 10000,
        "20K": 20000,
        "30K": 30000,
        "40K": 40000,
        "50K": 50000,
    }

elif sys.argv[1] == "speakerIdentification.py":
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

NoOfIterations = 3

# --------------------------------------------------
commands_base = [
    # "./dsp1 lowPassFull.py -emit=mlir-affine",
    # f"./dsp1 {input_file_path} -emit=llvm",
    f"{BasePathForLLVM}/build/bin/dsp1 {input_file_path} -emit=llvm",
    # "clang-17 -O0 file.ll -o fileexe -lm",
]

# Define the cases
cases = [
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
    {
        "affineOpt": True,
        "canonOpt": True,
        "suffix": "fileAffineCanonOpt.ll",
        "exe": "DSP_MLIRExe",
    },
]

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
                if input_file_name in [
                    "audioCompression.py",
                    "audioEqualization.py",
                    "periodogram.py",
                    "spectralAnalysis.py",
                ]:
                    updated_line = f"\tvar input = getRangeOfVector(0, {value}, 1);\n"
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
        command_llvm += f" 2> {OutputPath}/{case['suffix']}"  # OutputPath

        commands = [
            command_llvm,
            # f"clang-17 -O0 {case['suffix']} -o fileexe -lm",
            f"{BasePathForLLVM}/build/bin/clang-19 -O3 {OutputPath}/{case['suffix']} -o {OutputPath}/{case['exe']} -lm",
        ]
        # print(case,end="\n")
        # print("\n")

        # Iterate over each value and perform the necessary operations
        for command in commands:
            # Run the commands for the current case
            result = subprocess.run(command, shell=True, capture_output=True, text=True)

        sum_exe_time = 0
        for i in range(0, NoOfIterations):
            # for command in commands:
            #     # print("running command {}".format(command))
            #     # os.system(command)
            #     result = subprocess.run(command, shell=True, capture_output=True, text=True)

            # Clear the cache to minimize caching effects
            # subprocess.run("sync; echo 3 > /proc/sys/vm/drop_caches", shell=True)
            try:
                process = subprocess.run(
                    "sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'",
                    shell=True,
                    check=True,
                )
                # process.wait()
            except subprocess.CalledProcessError as exc:
                print(exc)
                process.terminate()
            # The command to be executed
            # command2 = "./fileexe"
            # Limit execution to a single core
            # command2 = "taskset -c 0 ./fileexe"
            command2 = f"taskset -c 0 ./Output/{sys.argv[2]}/{case['exe']}" #{OutputPath}
            # command2 = f"taskset -c 0 {OutputPath}/{case['exe']}"

            # Record the start time
            start_time = time.time()

            # Execute the command
            try:
                subprocess.run(
                    command2,
                    shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=True,
                )
                # subprocess.run(command2, shell=True)
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
        print("{}".format(avg_exe_time), end="\t")
    # print(f"The command took {execution_time} seconds to execute.")
