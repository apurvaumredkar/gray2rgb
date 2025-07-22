import os
import argparse
import subprocess

def folder_exists(path):
    return os.path.exists(path) and os.path.isdir(path)

def run_script(script_name):
    print(f"\nRunning {script_name}...")
    result = subprocess.run(["python", script_name])
    if result.returncode != 0:
        raise RuntimeError(f"{script_name} failed")

def main():
    parser = argparse.ArgumentParser(description="Pipeline controller for data generation, training, evaluation")
    parser.add_argument("--generate-data", action="store_true", help="Force dataset subset generation")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")

    args = parser.parse_args()

    data_dirs_exist = folder_exists("./data_subset/rgb") and folder_exists("./data_subset/gray")

    # If no flags specified, run all steps, generate data if missing
    if not (args.generate_data or args.train or args.evaluate):
        if not data_dirs_exist:
            run_script("generate_data_subset.py")
        run_script("train.py")
        run_script("evaluate.py")
        return

    # Run requested steps based on flags
    if args.generate_data:
        run_script("generate_data_subset.py")
    if args.train:
        run_script("train.py")
    if args.evaluate:
        run_script("evaluate.py")

if __name__ == "__main__":
    main()