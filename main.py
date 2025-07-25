import os
import argparse
import generate_data_subset
import train
import evaluate


def folder_exists(path):
    return os.path.exists(path) and os.path.isdir(path)


def main():
    parser = argparse.ArgumentParser(
        description="Main script for data subset generation, training & evaluation")
    parser.add_argument("--generate-data", action="store_true",
                        help="Force dataset subset generation")
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--evaluate", action="store_true",
                        help="Run evaluation")

    args = parser.parse_args()

    data_dirs_exist = folder_exists("./data_subset")

    if not (args.generate_data or args.train or args.evaluate):
        if not data_dirs_exist:
            generate_data_subset.generate_subset()
        else:
            print("Skipping data subset generation.")
        train.train_model_pipeline()
        evaluate.evaluate_model()
        return

    if args.generate_data:
        generate_data_subset.generate_subset()
    if args.train:
        train.train_model_pipeline()
    if args.evaluate:
        evaluate.evaluate_model()


if __name__ == "__main__":
    main()
