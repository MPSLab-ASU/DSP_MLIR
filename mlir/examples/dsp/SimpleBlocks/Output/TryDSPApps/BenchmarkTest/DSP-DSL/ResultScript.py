import os
import subprocess
import time

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
# Apps = "hearingAid.py" , "lowPassFull.py" , " audioCompression.py", "lowPassFIRFilterDesign.py" , "EnergyOfSignal.py", "periodogram2Conv1.py", "audioEqualizer.py", "vibrationAnalysis.py", "signalSmoothing.py", "targetDetection.py", "biomedicalSignalProcessing.py", "spaceCommunication.py", "echocancelling", "noisecancelling.py", "digitalModulation", "underWaterCommunication", "voiceActivityDetection", "radarSignalProcessing"
input_file_path = "vibrationAnalysis.py"
BasePathForLLVM = "/home/local/ASURITE/apkhedka/ForLLVM/"
OutputScriptPath = (
    "mlir/examples/dsp/SimpleBlocks/Output/TryDSPApps/BenchmarkTest/DSP-DSL/"
)
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
    # "1B": 1000000000
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
        "exe": "fileAffineOptExe",
    },
    {
        "affineOpt": True,
        "canonOpt": True,
        "suffix": "fileAffineCanonOpt.ll",
        "exe": "fileAffineCanonOptExe",
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
                updated_line = (
                    f"\tvar input = getRangeOfVector(0, {value}, 0.000125);\n"
                )
                file.write(updated_line)
            elif line.strip().startswith("var duration ="):
                updated_line = f"\tvar duration = {dur};\n"
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
            f"clang-17 -O3 {OutputPath}/{case['suffix']} -o {OutputPath}/{case['exe']} -lm",
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
                    "sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'",
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
            # command2 = f"taskset -c 0 ./{case['exe']}" #{OutputPath}
            command2 = f"taskset -c 0 ./Output/{case['exe']}"

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
