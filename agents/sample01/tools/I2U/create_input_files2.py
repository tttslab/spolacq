import pickle
from collections import Counter
from pathlib import Path
from tqdm import tqdm

from .utils2 import create_input_files

from agents.base_agent_with_units import BaseAgentWithUnits


def make_captions(paths: list[list[Path]], agent: BaseAgentWithUnits) -> list[list[list[str]]]:
    """
    Generate captions for images in the given paths.
    Args:
        paths: List of image paths.
    Returns:
        List of lists of captions for each image.
    """
    image_captions = []
    print(f"Generating captions for {len(paths)} itms...")
    for item_paths in tqdm(paths, desc="Items", position=0):
        captions = []
        for path in tqdm(item_paths, desc="Captions", position=1, leave=False):
            captions.append(agent.s2u.get_unit(path))
        image_captions.append(captions)
    return image_captions


def create_i2u_dataset(
    agent: BaseAgentWithUnits,
    task_name,
    image_paths,
    audio_paths,
    output_data_dir,
    captions_per_image=4,
    min_word_freq=1,
    max_len=100,
):
    files = [
        "train_image_paths.pickle",
        "train_image_captions.pickle",
        "val_image_paths.pickle",
        "val_image_captions.pickle",
        "test_image_paths.pickle",
        "test_image_captions.pickle",
        "word_freq.pickle",
    ]
    output_data_dir = Path(output_data_dir)
    if  all((output_data_dir / file).exists() for file in files):
        print("All files already exist. Skipping...")
    else:
        # Make Captions
        train_image_captions = make_captions(audio_paths["train"], agent)
        val_image_captions = make_captions(audio_paths["val"], agent)
        test_image_captions = make_captions(audio_paths["test"], agent)

        word_freq = Counter()
        for captions in train_image_captions:
            for caption in captions:
                word_freq.update(caption)

        train_image_paths = image_paths["train"]
        val_image_paths = image_paths["val"]
        test_image_paths = image_paths["test"]

        output_data_dir.mkdir(parents=True, exist_ok=True)

        with open((output_data_dir / "train_image_paths.pickle"), "wb") as f:
            pickle.dump(train_image_paths, f)
        with open((output_data_dir / "train_image_captions.pickle"), "wb") as f:
            pickle.dump(train_image_captions, f)
        with open((output_data_dir / "val_image_paths.pickle"), "wb") as f:
            pickle.dump(val_image_paths, f)
        with open((output_data_dir / "val_image_captions.pickle"), "wb") as f:
            pickle.dump(val_image_captions, f)
        with open((output_data_dir / "test_image_paths.pickle"), "wb") as f:
            pickle.dump(test_image_paths, f)
        with open((output_data_dir / "test_image_captions.pickle"), "wb") as f:
            pickle.dump(test_image_captions, f)
        with open((output_data_dir / "word_freq.pickle"), "wb") as f:
            pickle.dump(word_freq, f)

    # Create input files (along with word map)
    data_info = create_input_files(
        dataset=task_name,
        captions_per_image=captions_per_image,
        min_word_freq=min_word_freq,
        output_folder=output_data_dir,
        max_len=max_len,
    )

    return {
        "i2u_data_dir": output_data_dir,
        "word_map_path": data_info["word_map_path"],
        "data_filename": data_info["data_filename"],
    }
