import numpy as np
import torch

from vidmap.frontend.depth_loading import WindowDataset


def test_even_window_has_exact_size_and_contains_center():
    class Images:
        names = tuple(f"{index}.jpg" for index in range(20))

        def __len__(self):
            return len(self.names)

        def __getitem__(self, index):
            return {
                "name": self.names[index],
                "image": torch.full((3, 4, 4), index),
                "original_size": np.array([4, 4]),
            }

    dataset = WindowDataset(Images(), Images.names, 10)
    for index in range(len(dataset)):
        item = dataset[index]
        assert item["images"].shape == (10, 3, 4, 4)
        assert item["images"][item["center_index"], 0, 0, 0] == index
