import shutil
import os
import traceback
import time

from src.counter import init_db

# Init Ledget database
init_db()

from src.constants import (
    ERROR_FOLDER,
    INPUT_FOLDER,
)
from src.process_file import process_file


def wait_for_file_completion(file_path, check_interval=1, max_attempts=10):
    previous_size = -1
    attempts = 0

    while attempts < max_attempts:
        current_size = os.path.getsize(file_path)
        if current_size == previous_size:
            return True  # File is stable (done writing)
        previous_size = current_size
        attempts += 1
        time.sleep(check_interval)

    print(
        f"File {file_path} did not stabilize after {max_attempts} attempts.", flush=True
    )
    return False


def main():
    # Start fresh observer instance
    print("Watching for new files...", flush=True)
    while True:
        files = [
            f
            for f in os.listdir(INPUT_FOLDER)
            if os.path.isfile(os.path.join(INPUT_FOLDER, f))
        ]
        # files = os.listdir(INPUT_FOLDER)
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
                    # Process file
                    process_file(file_path, file_name)

                    print(f"File processed: {file_name}", flush=True)
                except Exception as e:
                    print(f"Error with file: {file_name}", flush=True)
                    print(e)
                    print("-", flush=True)
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
