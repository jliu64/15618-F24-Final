import os


def extract_flight_times(directory):
    """
    Reads all files in a given directory, filters for 'psc' or 'ghc' in filenames,
    extracts 'Time taken to compute flight strings', and prints it alongside the
    extracted H and thread count from the filenames, also indicating psc or ghc.

    :param directory: Directory containing the output files
    """
    for file_name in os.listdir(directory):
        # Check if the file is relevant (contains 'psc' or 'ghc')
        if file_name.endswith(".txt") and ("psc" in file_name or "ghc" in file_name):
            # Identify whether the file corresponds to psc or ghc
            system = "psc" if "psc" in file_name else "ghc"

            # Extract H and thread count from the file name
            parts = file_name.split("_")
            if len(parts) >= 4:
                h_value = parts[2]
                threads = parts[3].replace(".txt", "")
            else:
                continue

            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r') as file:
                lines = file.readlines()

            # Search for 'Time taken to compute flight strings' in the file
            for line in lines:
                if "Time taken to compute flight strings:" in line:
                    time_value = line.split(":")[1].strip().replace("s", "").strip()
                    print(
                        f"OpenMP ({system.upper()}) H = {h_value}, Threads = {threads}, Time taken to compute flight strings: {time_value} s")


# Example usage
directory_path = "./Outputs/OpenMP"
extract_flight_times(directory_path)
