import os
import argparse
import generate_data_subset
import train
import evaluate


def main():
    parser = argparse.ArgumentParser(
        description="Main script for data subset generation, training & evaluation"
    )
    parser.add_argument(
        "--generate-data", action="store_true", help="Force dataset subset generation"
    )
    parser.add_argument("--train", action="store_true", help="Run training")
    parser.add_argument("--evaluate", action="store_true", help="Run evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for evaluation (optional)",
    )

    args = parser.parse_args()

    folder_exists = lambda path: os.path.exists(path) and os.path.isdir(path)

    data_dirs_exist = (
        folder_exists("./data_subset/train")
        and folder_exists("./data_subset/val")
        and folder_exists("./data_subset/test")
    )

    if not (args.generate_data or args.train or args.evaluate):
        print("Running full pipeline: data generation -> training -> evaluation")

        if not data_dirs_exist:
            print("Generating data subset...")
            generate_data_subset.generate_subset()
        else:
            print("Data subset already exists, skipping generation.")

        print("Starting training...")
        train.main()

        print("Starting evaluation...")
        evaluate.evaluate_model()

        return

    if args.generate_data:
        print("Generating data subset...")
        generate_data_subset.generate_subset()

    if args.train:
        print("Starting training...")
        train.main()

    if args.evaluate:
        print("Starting evaluation...")
        if args.checkpoint:
            evaluate.evaluate_model(checkpoint_path=args.checkpoint)
        else:
            evaluate.evaluate_model()


if __name__ == "__main__":
    main()
