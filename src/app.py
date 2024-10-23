import shutil
import os
import traceback
import time
import json

from src.constants import (
    CONFIG_FILE,
    ERROR_FOLDER,
    INPUT_FOLDER,
    OUTPUT_FOLDER,
    PROCESSED_INPUT_FOLDER,
    TAG_DEFAULT_VALUES_FILE,
)
from src.process.format import csv_output
from src.process.main import process_file


def wait_for_file_completion(file_path, check_interval=1, max_attempts=10):
    previous_size = -1
    attempts = 0

    while attempts < max_attempts:
        current_size = os.path.getsize(file_path)
        if current_size == previous_size:
            # print("attempts", attempts, flush=True)
            return True  # File is stable (done writing)
        previous_size = current_size
        attempts += 1
        time.sleep(check_interval)

    print(
        f"File {file_path} did not stabilize after {max_attempts} attempts.", flush=True
    )
    return False


with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = json.load(f)

with open(TAG_DEFAULT_VALUES_FILE, "r", encoding="utf-8") as f:
    tag_default_values = json.load(f)


def main():
    # Start fresh observer instance
    print("Watching for new files...", flush=True)
    while True:
        files = os.listdir(INPUT_FOLDER)
        if files:
            files_with_time = [
                (f, os.path.getatime(os.path.join(INPUT_FOLDER, f))) for f in files
            ]

            # Sort files based on modification time
            sorted_files = sorted(files_with_time, key=lambda x: x[1])

            first_added_file = sorted_files[0][0]
            first_added_file_path = os.path.join(INPUT_FOLDER, first_added_file)

            file_path = first_added_file_path
            file_name = os.path.basename(file_path)
            print(f"Starting processing: {file_name}", flush=True)

            if wait_for_file_completion(first_added_file_path):
                try:
                    response = process_file(file_path, config, tag_default_values)

                    # Save response
                    response_output_path = os.path.join(
                        OUTPUT_FOLDER, f"{file_name}-output.csv"
                    )
                    csv_output(response, response_output_path)
                    # with open(response_output_path, "w", encoding="utf-8") as f:
                    #     json.dump(response, f)

                    # Clean up
                    processed_input_path = os.path.join(
                        PROCESSED_INPUT_FOLDER, file_name
                    )
                    shutil.move(file_path, processed_input_path)

                    print(f"File processed: {file_name}", flush=True)
                except Exception as e:
                    print(f"Error with file: {file_name}", flush=True)
                    print(e)
                    error_file_output_path = os.path.join(ERROR_FOLDER, file_name)
                    shutil.move(file_path, error_file_output_path)
                    error_msg_output_path = os.path.join(
                        ERROR_FOLDER, f"{file_name}-error.txt"
                    )
                    with open(
                        error_msg_output_path, "w", encoding="utf-8"
                    ) as output_file:
                        traceback.print_exc(file=output_file)
        else:
            time.sleep(1)


if __name__ == "__main__":
    main()
