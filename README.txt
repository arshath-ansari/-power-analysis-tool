readme_text = """Power and Binary Analysis Toolkit
=================================

This toolkit analyzes power profiling data from either CSV waveform files or binary `.prof` files. It automatically detects the input format and runs the appropriate analysis pipeline.

Installation
------------
Before running the scripts, install the required Python packages:

    pip install numpy pandas plotly psutil tqdm
        (or)
    pip install -r requirements.txt (This will install all the required packages automatically)
      (or)
    pip install --trusted-host pypi.org --trusted-host files.pythonhosted.org numpy pandas plotly psutil tqdm (for testers)



Environment Setup
-----------------
1. Open `settings.py` and set the path to your input files:

    
IOP_PATH = r"\\path\\to\\your\\iop.csv"
VOP_PATH = r"\\path\\to\\your\\vop.csv"
POWER_PATH = r"\\path\\to\\your\\power.csv"
BINARY_PROF_PATH = r"\\path\\to\\your\\binary.prof"


Update this path to match the location of your waveform or binary profiling files.

How to Run
----------
Step 1: Prepare Input Files

Ensure your input files are placed in the appropriate directory:
- CSV waveform files: IOP, VOP, POWER
- Binary profiling file: `.prof`

Step 2: Run CSV Analysis

To analyze CSV waveform files, run:

    python poweranalysis_csv.py

This script will:
- Load voltage, current, and power data from CSV files.
- Compute metrics like max, average, P95, and energy.
- Detect threshold violations.
- Generate HTML plots and CSV summaries.
Additional Tip
--------------
If you only want to run with IOP, change this line in `main()`:

    hpsv = HighPowerSpecViolation(VOP_PATH, IOP_PATH, POWER_PATH)
to:
     hpsv = HighPowerSpecViolation(iop_path=IOP_PATH)



Step 3: Run Binary Analysis

To analyze binary `.prof` files, run:

    python poweranalysis_binary.py

This script will:
- Parse binary waveform profiling data.
- Match waveform entries to testsuites.
- Generate voltage, current, power plots and summaries.

Output
------
Depending on the input format, the following outputs will be generated:
- HTML Plots: Voltage, Current, Power, Energy (Max and P95)
- CSV Summaries: Average power, threshold violations
- HTML Reports: Rail summaries per site

Notes
-----
- All configuration paths and thresholds are defined in `settings.py`.
- Ensure Python 3.8+ is installed.
- Run scripts from the directory:

    cd \\\\qcdfs\\QCT\\prj\\vlsi\\pete\\kaanapalit\\eng\\developers\\moan\\finalpresentation\\code
"""

with open("run_instructions.txt", "w") as f:
    f.write(readme_text)

print("run_instructions.txt file has been created successfully.")





