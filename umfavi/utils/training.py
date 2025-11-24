from torch.utils.data import DataLoader
from umfavi.types import FeedbackType
from typing import Iterator
from typing import Any


def get_batch(dloader_iters: dict[FeedbackType, Iterator], dataloaders: dict[FeedbackType, DataLoader], fb_type: FeedbackType) -> dict[str, Any]:
    try:
        batch = next(dloader_iters[fb_type])
    except StopIteration:
        # Restart iterator if we've exhausted this dataloader
        dloader_iters[fb_type] = iter(dataloaders[fb_type])
        batch = next(dloader_iters[fb_type])
    return batch
