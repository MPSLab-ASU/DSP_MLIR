import os
import subprocess
import pandas as pd

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
input_files = [
    "spectralAnalysis",
    "audioCompression",
    "audioEqualization",
    "biomedicalSignalProcessing",
    "digitalModulation",
    "dtmfDetection",
    "echoCancellation",
    "FIRFilterDesign",
    "hearingAid",
    "lowPassFiltering",
    "noiseCancellation",
    "periodogram",
    "vibrationAnalysis",
    "radarSignalProcessing",
    "signalSmoothing",
    "spaceCommunication",
    "speakerIdentification",
    "targetDetection",
    "underWaterCommunication",
    "voiceActivityDetection"
]
data = []

for input_file_path in input_files:
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
            f"{BasePathForLLVM}/build/bin/dsp1 {input_file_path}.py -emit=llvm",
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
    with open(input_file_path+".py", "r") as file:
        lines = file.readlines()

    print("", end="\t")
    for case in cases:
        print(f"{case['exe']}", end="\t")

    size_test = {"100M": 100000000}
    for key, value in size_test.items():
        value2 = 1 / value
        dur = value / 8192
        print(f"\n{key}", end="\t")

        with open(input_file_path+".py", "r") as file:
            lines = file.readlines()

        with open(input_file_path+".py", "w") as file:
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
                    f"{BasePathForLLVM}/build/bin/clang-19 -O3 {OutputPath}/{case['suffix']} -o {OutputPath}/{case['exe']} -lm",
                    ]
            # print(case,end="\n")
            # print("\n")

            # Iterate over each value and perform the necessary operations
            for command in commands:
                # Run the commands for the current case
                result = subprocess.run(command, shell=True, capture_output=True, text=True)

            command2 = f"size ./Output/{case['exe']}"

            # Execute the command
            try:
                result = subprocess.run(
                        command2,
                        shell=True,
                        capture_output=True, text=True
                        )
                
                output_parts = result.stdout.splitlines()
                if len(output_parts) > 1:
                    size_data = output_parts[1].split()

                    data.append({
                        "filename": input_file_path,
                        # "input size" : key,
                        "opt": case['exe'],
                        # "text": size_data[0],
                        # "data": size_data[1],
                        # "bss": size_data[2],
                        # "dec": size_data[3],
                        # "hex": size_data[4],
                        "total": sum(map(int, size_data[:4]))
                    })
            except subprocess.CalledProcessError as exc:
                print(
                        f"Process failed because did not return a successful return code. "
                        f"Returned {exc.returncode}\n{exc}"
                        )

        df = pd.DataFrame(data)
        
        df.to_csv("codesize.csv", index=False)