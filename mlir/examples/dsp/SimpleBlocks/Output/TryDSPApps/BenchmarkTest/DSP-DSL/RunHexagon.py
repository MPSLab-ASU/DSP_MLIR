import subprocess
import os

# Ensure the log directory exists
log_dir = "LClanglogs"
os.makedirs(log_dir, exist_ok=True)

# Corrected list of application names (without non-application entries)
app_names = [
    "biomedicalSignalProcessing",
    "audioCompression",
    "audioEqualization",
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
    "voiceActivityDetection",
    "spectralAnalysis"
]

# Loop through each application and execute the HexagonResultScript.py script


log_dir2 = "HClanglogs"
os.makedirs(log_dir2, exist_ok=True)



for app_name in app_names:
    app_script = f"{app_name}.py"
    log_file2 = os.path.join(log_dir2, f"{app_name}.log")
    with open(log_file2, "w") as log:
        command = ["python", "HexagonClangResultScript.py", app_script, app_name]
        print(f"Running command: {' '.join(command)}")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        for line in process.stderr:
            print(line, end="")
            log.write(line)
        process.wait()

for app_name in app_names:
    app_script = f"{app_name}.py"
    log_file = os.path.join(log_dir, f"{app_name}.log")
    with open(log_file, "w") as log:
        command = ["python", "HexagonResultScript.py", app_script, app_name]
        print(f"Running command: {' '.join(command)}")
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        for line in process.stderr:
            print(line, end="")
            log.write(line)
        process.wait()
