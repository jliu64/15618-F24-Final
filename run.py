import os
import sys
import subprocess

# Define paths for executables
SEQUENTIAL_EXECUTABLE = "./Sequential/flightroute"
OPENMP_EXECUTABLE = "./OpenMP/flightroute"

# Define input and output paths
INPUT_FOLDER = "Inputs"
OUTPUT_FOLDER = "Outputs"
SEQ_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "Sequential")
OMP_OUTPUT_FOLDER = os.path.join(OUTPUT_FOLDER, "OpenMP")

# List of horizons
HORIZONS = [2, 3]

# Thread counts for different machines
THREAD_COUNTS = {
    "ghc": [2, 4, 8, 16],
    "psc": [2, 4, 8, 16, 32, 64, 128]
}

# Create necessary output folders if they don't exist
os.makedirs(SEQ_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(OMP_OUTPUT_FOLDER, exist_ok=True)


def run_sequential(horizon, machine):
    input_file = f"{INPUT_FOLDER}/data_{horizon}.txt"
    output_file = f"{SEQ_OUTPUT_FOLDER}/{machine}_seq_{horizon}.txt"

    try:
        print(f"Running Sequential on {machine.upper()} for H={horizon}...")
        result = subprocess.run(
            [SEQUENTIAL_EXECUTABLE, input_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        with open(output_file, "w") as f:
            f.write(result.stdout.decode('utf-8'))  # Decode bytes to string for compatibility
        print(f"Sequential output saved to {output_file}.")
    except subprocess.CalledProcessError as e:
        print(f"Error running Sequential on {machine.upper()} for H={horizon}: {e.stderr.decode('utf-8')}")


def run_openmp(horizon, threads, machine):
    input_file = f"{INPUT_FOLDER}/data_{horizon}.txt"
    output_file = f"{OMP_OUTPUT_FOLDER}/{machine}_openmp_{horizon}_{threads}.txt"

    try:
        print(f"Running OpenMP on {machine.upper()} for H={horizon} with {threads} threads...")
        result = subprocess.run(
            [OPENMP_EXECUTABLE, input_file, str(threads)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        with open(output_file, "w") as f:
            f.write(result.stdout.decode('utf-8'))  # Decode bytes to string for compatibility
        print(f"OpenMP output saved to {output_file}.")
    except subprocess.CalledProcessError as e:
        print(
            f"Error running OpenMP on {machine.upper()} for H={horizon} with {threads} threads: {e.stderr.decode('utf-8')}")


if __name__ == "__main__":
    # Validate command-line arguments
    if len(sys.argv) < 3:
        print("Usage for Sequential: python run.py seq <ghc/psc> <horizon/all>")
        print("Usage for OpenMP: python run.py openmp <ghc/psc> <horizon/all> <threads/all>")
        sys.exit(1)

    program_type = sys.argv[1].lower()
    machine = sys.argv[2].lower()
    horizon_arg = sys.argv[3]
    threads_arg = sys.argv[4] if len(sys.argv) > 4 else None

    if program_type not in ["seq", "openmp"]:
        print("Error: First argument must be 'seq' or 'openmp'.")
        sys.exit(1)

    if machine not in ["ghc", "psc"]:
        print("Error: Second argument must be 'ghc' or 'psc'.")
        sys.exit(1)

    # Validate horizon argument
    if horizon_arg.isdigit():
        horizons_to_run = [int(horizon_arg)]
    elif horizon_arg == "all":
        horizons_to_run = HORIZONS
    else:
        print("Error: <horizon> must be a number or 'all'.")
        sys.exit(1)

    # Adjust thread counts based on machine
    thread_counts = THREAD_COUNTS[machine]

    if program_type == "seq":
        # Sequential program requires only the horizon argument
        for horizon in horizons_to_run:
            run_sequential(horizon, machine)
    elif program_type == "openmp":
        # OpenMP program requires both horizon and threads arguments
        if not threads_arg:
            print("Error: OpenMP requires <threads> argument.")
            sys.exit(1)

        # Validate threads argument
        if threads_arg.isdigit():
            threads_to_run = [int(threads_arg)]
        elif threads_arg == "all":
            threads_to_run = thread_counts
        else:
            print("Error: <threads> must be a number or 'all'.")
            sys.exit(1)

        for horizon in horizons_to_run:
            for threads in threads_to_run:
                run_openmp(horizon, threads, machine)
