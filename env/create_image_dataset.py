from pathlib import Path
import torch
from diffusers import DiffusionPipeline, DPMSolverSinglestepScheduler
import numpy as np
from tqdm import tqdm
from PIL import Image as PILImage

from abc import ABC, abstractmethod


class ImageGenerator(ABC):
    @abstractmethod
    def generate(self, prompt: str, num_images: int, config) -> list[PILImage.Image]:
        pass

    def evaluate(self, image: PILImage.Image) -> bool:
        return True


class MannETurboImageGenerator(ImageGenerator):
    def __init__(self):
        # # Load model
        self.diffusion_pipe = DiffusionPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16,
            cache_dir="./cache/image_generator",
            safety_checker=None,
        )
        self.diffusion_pipe.to(torch.device("cuda"))
        self.diffusion_pipe.enable_model_cpu_offload()

        # Load LoRA weights
        try:
            self.diffusion_pipe.load_lora_weights(
                "mann-e/Mann-E_Turbo",
                weight_name="manne_turbo.safetensors",
                cache_dir="./cache/image_generator",
            )
        except Exception as e:
            print(f"LoRA weights loading failed: {e}")
            print("Continuing without LoRA weights...")

        # Configure scheduler
        self.diffusion_pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
            self.diffusion_pipe.scheduler.config,
            use_karras_sigmas=True,
        )
        self.diffusion_pipe.set_progress_bar_config(disable=True)

    def generate(self, prompt: str, num_images: int, config) -> list[PILImage.Image]:
        images = self.diffusion_pipe(
            prompt=prompt,
            num_images_per_prompt=num_images,
            num_inference_steps=config["num_inference_steps"],
            guidance_scale=config["guidance_scale"],
            height=config["height"],
            width=config["width"],
            output_type="pil",
        ).images

        return images

    def evaluate(self, image: PILImage.Image) -> bool:
        # Check if the image is not completely black
        return np.mean(np.array(image)) >= 1


def get_image_generator(model_name: str = "Mann-E_Turbo") -> ImageGenerator:
    """Get an image generator"""
    if model_name == "Mann-E_Turbo":
        return MannETurboImageGenerator()


def create_dataset_structure(base_dir, items):
    """Create dataset folder structure"""
    for item in items:
        # Main folder
        item_dir = Path(base_dir) / item

        # Training folder
        for i in range(1, 4):
            train_dir = item_dir / f"train_number{i}"
            train_dir.mkdir(parents=True, exist_ok=True)

        # Test folder
        for i in range(1, 4):
            test_dir = item_dir / f"test_number{i}"
            test_dir.mkdir(parents=True, exist_ok=True)


def generate_and_save_images(
    image_generator: ImageGenerator,
    model_config,
    prompt,
    save_dir,
    num_images,
    start_group=1,
    batch_size=4,
    overwrite=False,
):
    """Generate images in batches, validate them, and save individually"""
    current_group = start_group
    group_image_count = 1
    saved_images = 0

    print("overwrite", overwrite)
    if not overwrite:
        # Count existing images
        print(f"Checking existing images in {save_dir}...")
        print(f"all images in {save_dir}:{list(save_dir.rglob('*.jpg'))}")
        for path in save_dir.rglob("*.jpg"):
            saved_images += 1
        print(f"Found {saved_images} existing images in {save_dir}.")
        print(f"num_images:{num_images}, saved_images:{saved_images}, overwrite:{overwrite}")
        print(f"Decision to stop generation: {saved_images >= num_images and not overwrite}")
        if saved_images >= num_images and not overwrite:
            print(f"Images already exist in {save_dir}. Skipping generation.")
            return

    with tqdm(total=num_images, desc=f"Valid images saved", leave=False, position=2) as pbar:
        while saved_images < num_images:
            # Generate batch images
            images = image_generator.generate(prompt=prompt, num_images=batch_size, config=model_config)

            # Validate and save each image
            for image in images:
                # Check if the image is not completely black
                if image_generator.evaluate(image):
                    # Generate file name
                    for _ in range(num_images):
                        filename = f"group{current_group}_{group_image_count}.jpg"
                        save_path = save_dir / filename
                        if (not save_path.exists()) or overwrite:
                            continue
                        group_image_count += 1
                        if group_image_count > 2:
                            group_image_count = 1
                            current_group += 1
                    print("save_path", save_path)

                    # Save image
                    image.save(save_path)

                    # Update group counter
                    group_image_count += 1
                    if group_image_count > 2:
                        group_image_count = 1
                        current_group += 1

                    # Update saved image count
                    saved_images += 1
                    pbar.update(1)

                    # Stop when enough images have been saved
                    if saved_images >= num_images:
                        torch.cuda.empty_cache()
                        break

            # Clear GPU memory
            torch.cuda.empty_cache()


def create_image_dataset(
    image_data_dir,
    model_name,
    model_config,
    items,
    item_prompts,
    train_images_per_folder=30,
    test_images_per_folder=10,
    overwrite=False,
    batch_size=15,
):
    """Generate image dataset for each item."""
    # Load model
    image_generator = get_image_generator(model_name=model_name)

    # Create folder structure
    create_dataset_structure(image_data_dir, items)

    # Generate images for each item
    for item, prompt in tqdm(zip(items, item_prompts), desc="Item loop", position=0):
        print(f"\nGenerating images for: {item}")
        if prompt is None:
            base_prompt = f"A {item} on a white background, uncooked, realistic."
        else:
            base_prompt = prompt

        # Generate training images
        for i in tqdm(range(1, 4), desc="Train loop", position=1, leave=False):
            train_dir = Path(image_data_dir) / item / f"train_number{i}"
            print(f"Generating training images in {train_dir}")
            generate_and_save_images(
                image_generator,
                model_config,
                base_prompt,
                train_dir,
                train_images_per_folder,
                start_group=1,
                batch_size=batch_size,
                overwrite=overwrite,
            )

        # Generate test images
        for i in tqdm(range(1, 4), desc="Test loop", position=1, leave=False):
            test_dir = Path(image_data_dir) / item / f"test_number{i}"
            print(f"Generating test images in {test_dir}")
            generate_and_save_images(
                image_generator,
                model_config,
                base_prompt,
                test_dir,
                test_images_per_folder,
                start_group=1,
                batch_size=batch_size,
                overwrite=overwrite,
            )

    train_image_paths, val_image_paths, test_image_paths = get_image_paths(image_data_dir)
    return {
        "train": train_image_paths,
        "val": val_image_paths,
        "test": test_image_paths,
    }


def get_image_paths(image_dir: Path, image_type="jpg"):
    """
    Get image paths from the specified directory.
    Args:
        image_dir (Path): Directory containing images.
        image_type (str): Type of images (e.g., 'jpg', 'png').
    Returns:
        list: List of Path objects for images.
    """
    train_image_paths = sorted(list(image_dir.glob(f"*/train_number[123]/*.{image_type}")))
    val_image_paths = sorted(list(image_dir.glob(f"*/test_number[12]/*.{image_type}")))
    test_image_paths = sorted(list(image_dir.glob(f"*/test_number3/*.{image_type}")))
    return train_image_paths, val_image_paths, test_image_paths
