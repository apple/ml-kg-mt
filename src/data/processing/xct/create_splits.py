#
# For licensing see accompanying LICENSE.txt file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

"""
create_splits.py

This script is used to create the splits for the Cross-Cultural Translation (XC-Translate) dataset.

The script reads the XC-Translate dataset and creates three splits: sample, validation, test. The sample split is used to get a quick look at the data. The validation split shall be used to tune the model hyperparameters. The test split shall be used only to evaluate the model performance.

Usage:
    python src/data/processing/xct/create_splits.py --data_dir <path_to_data> --output_dir <path_to_output> [--sample_size <sample_size> --validation_size <validation_size> --seed <seed>]

Arguments:
    data_path: str: Path to the XC-Translate dataset (all the references for all the languages).
    output_dir: str: Path to the output directory where the splits will be saved.

Options:
    --sample_size: int: Sample split size [default: 25].
    --validation_size: int: Validation split size [default: 250].
    --seed: int: Seed for reproducibility [default: 42].

Example:
    python src/data/processing/xct/create_splits.py --data_dir data/xct/references/all --output_dir data/xct/references --sample_size 50 --validation_size 500 --test_size -1 --seed 42
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple

from loguru import logger


def load_dataset(data_dir: str) -> Dict[str, dict]:
    """
    Read the XC-Translate dataset from the given directory.

    Args:
        data_dir (str): Path to the directory containing the XC-Translate dataset.

    Returns:
        references: Dict[str, dict]: A dictionary containing the references for each language.
    """
    references = {}

    # List all the files in the data directory.
    files = os.listdir(data_dir)

    for file in files:
        if not file.endswith(".jsonl"):
            continue

        language = file.split(".")[0]
        references[language] = load_references(os.path.join(data_dir, file))

    return references


def load_references(input_path: str) -> List[dict]:
    """
    Load data from the input file (JSONL) and return a list of dictionaries, one for each instance in the dataset.

    Args:
        input_path (str): Path to the input file.

    Returns:
        references (List[dict]): List of dictionaries, one for each instance in the dataset.
    """
    references = []

    with open(input_path, "r") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            references.append(json.loads(line))

    return references


def get_entity_ids(references: List[dict]) -> List[str]:
    """
    Args:
        references (List[dict]): A dictionary containing the references for each language.

    Returns:
        entity_ids (List[str]): List of entity IDs shared across all the languages.
    """
    entity_ids = set()

    for instance in references:
        entity_ids.add(instance["wikidata_id"])

    return list(entity_ids)


def create_splits(
    references: Dict[str, dict],
    sample_size: int = 25,
    validation_size: int = 250,
    seed: int = 42,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Create the sample, validation, and test splits from the given references.

    Args:
        references (Dict[str, dict]): A dictionary containing the references for each language.
        sample_size (int): Sample split size.
        validation_size (int): Validation split size.
        seed (int): Seed for reproducibility.

    Returns:
        sample_split: List[dict]: Sample split.
        validation_split: List[dict]: Validation split.
        test_split: List[dict]: Test split.
    """
    # Set the seed for reproducibility.
    random.seed(seed)

    sample_split = {l: [] for l in references}
    validation_split = {l: [] for l in references}
    test_split = {l: [] for l in references}

    for lang, instances in references.items():
        entity_ids = get_entity_ids(instances)

        # Shuffle the entity IDs.
        random.shuffle(entity_ids)

        # Create the splits.
        sample_ids = entity_ids[:sample_size]
        validation_ids = entity_ids[sample_size : sample_size + validation_size]
        test_ids = entity_ids[sample_size + validation_size :]

        for instance in instances:
            if instance["wikidata_id"] in sample_ids:
                sample_split[lang].append(instance)
            elif instance["wikidata_id"] in validation_ids:
                validation_split[lang].append(instance)
            elif instance["wikidata_id"] in test_ids:
                test_split[lang].append(instance)

    return sample_split, validation_split, test_split


def save_split(output_dir: str, split_name: str, split: Dict[str, List[dict]]):
    """
    Save the split to the output directory.

    Args:
        output_dir (str): Path to the output directory.
        split_name (str): Name of the split (sample, validation, test).
        split (Dict[str, List[dict]]): Split to save.
    """
    for lang, instances in split.items():
        # Get all the instance_ids and sort them alphabetically.
        instances = sorted(instances, key=lambda x: x["id"])

        # Save the split to the output directory.
        os.makedirs(os.path.join(output_dir, split_name), exist_ok=True)

        with open(os.path.join(output_dir, split_name, f"{lang}.jsonl"), "w") as file:
            for instance in instances:
                file.write(json.dumps(instance, ensure_ascii=False) + "\n")


def main(
    data_dir: str,
    output_dir: str,
    sample_size: int,
    validation_size: int,
    seed: int,
):
    # Load the dataset.
    logger.info("Loading the dataset...")
    references = load_dataset(data_dir)
    logger.info(f"Dataset loaded successfully with {len(references)} languages.")

    # Create the splits.
    logger.info("Creating the splits...")
    sample_split, validation_split, test_split = create_splits(
        references,
        sample_size,
        validation_size,
        seed,
    )
    logger.info("Splits created successfully.")

    # Save the splits.
    logger.info("Saving the splits...")
    split_names = ["sample", "validation", "test"]
    splits = [sample_split, validation_split, test_split]

    for split_name, split in zip(split_names, splits):
        save_split(output_dir, split_name, split)

    logger.info("Splits saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(usage=__doc__)
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Path to the XC-Translate dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=25,
        help="Sample split size.",
    )
    parser.add_argument(
        "--validation_size",
        type=int,
        default=250,
        help="Validation split size.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for reproducibility.",
    )

    args = parser.parse_args()

    main(
        args.data_dir,
        args.output_dir,
        args.sample_size,
        args.validation_size,
        args.seed,
    )
