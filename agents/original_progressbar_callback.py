import time

from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm


# from food_task.py
class ProgressBarCallback(BaseCallback):
    """
    Custom callback for plotting progress bar and training time.
    """

    def __init__(self, total_timesteps):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.start_time = None

    def _on_training_start(self):
        self.start_time = time.time()
        self.pbar = tqdm(total=self.total_timesteps, desc="Training Progress")

    def _on_step(self):
        self.pbar.update(1)
        elapsed_time = time.time() - self.start_time
        self.pbar.set_postfix({"Elapsed": f"{elapsed_time:.2f}s", "FPS": f"{self.num_timesteps / elapsed_time:.2f}"})
        return True

    def _on_training_end(self):
        self.pbar.close()
        total_time = time.time() - self.start_time
        print(f"\nTotal training time: {total_time:.2f} seconds")
        print(f"Average FPS: {self.total_timesteps / total_time:.2f}")
