import pandas as pd
from lib import BrennanDataset
from torch.utils.data import Subset, ConcatDataset
from pathlib import Path

base_dir = Path("/ocean/projects/cis240129p/shared/data/eeg_alice")
# phoneme_dir = "/ocean/projects/cis240129p/shared/data/eeg_alice/phonemes"
# phoneme_dict_path = "/ocean/projects/cis240129p/shared/data/eeg_alice/phoneme_dict.txt"
subjects_used = ["S04", "S13", "S19"]  # exclude 'S05' - less channels


def create_datasets(subjects, base_dir):
    train_datasets = []
    test_datasets = []
    for subject in subjects:
        dataset = BrennanDataset(
            root_dir=base_dir,
            phoneme_dir=base_dir / "phonemes",
            idx=subject,
            phoneme_dict_path=base_dir / "phoneme_dict.txt",
        )
        num_data_points = len(dataset)

        # Split indices into train and test sets
        split_index = int(num_data_points * 0.8)
        train_indices = list(range(split_index))
        test_indices = list(range(split_index, num_data_points))

        # Create Subset datasets using indices
        train_dataset = Subset(dataset, train_indices)
        test_dataset = Subset(dataset, test_indices)

        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
    return train_datasets, test_datasets


# ds = BrennanDataset(
#     root_dir=base_dir,
#     phoneme_dir=base_dir / "phonemes",
#     idx="S01",
#     phoneme_dict_path=base_dir / "phoneme_dict.txt",
# )
train_ds, test_ds = create_datasets(subjects_used, base_dir)
train_dataset = ConcatDataset(train_ds)
test_dataset = ConcatDataset(test_ds)
print(
    f"Train dataset length: {len(train_dataset)}, Test dataset length: {len(test_dataset)}"
)
import pdb

pdb.set_trace()
