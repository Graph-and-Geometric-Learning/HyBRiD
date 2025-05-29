import os
from collections import defaultdict
from functools import cached_property

import h5py
import numpy as np
import torch
from lightning import LightningDataModule
from scipy.stats import entropy as stats_entropy
from sklearn.model_selection import train_test_split
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from tqdm import tqdm


class MultiSourceDataset(Dataset):
    def __init__(self, datasets: dict[str, Dataset]):
        self.datasets: dict = datasets

    def __len__(self):
        return sum([len(d) for d in self.datasets.values()])

    def __getitem__(self, idx: tuple[str, int]):
        task, idx = idx
        dataset = self.datasets[task]
        return dataset[idx]


class MultiSourceBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        data_source: Dataset,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        def chunks(lst, n):
            """Yield successive n-sized chunks from lst."""
            for i in range(0, len(lst), n):
                yield lst[i : i + n]

        all_indices: list[list[tuple[str, int]]] = []
        for task, dataset in self.data_source.datasets.items():
            indices: list[int] = list(range(len(dataset)))
            if self.shuffle:
                np.random.shuffle(indices)
            for chunk in chunks(indices, self.batch_size):
                all_indices.append([(task, c) for c in chunk])
        if self.shuffle:
            np.random.shuffle(all_indices)

        yield from all_indices

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]


class BrainDataset(Dataset):
    def __init__(
        self,
        file: h5py.File,
        dataset_name: str,
        task_name: str,
        x_key: str = "x",
        y_key: str = "y",
    ):
        self.file = file
        self.dataset_name = dataset_name
        self.task_name = task_name
        self.in_memory = True

        self.x_key = x_key
        self.y_key = y_key

        assert hasattr(self, "dataset_name") and hasattr(self, "task_name")

        self.dataset = self._read_data()

    def __len__(self):
        return len(self.dataset)

    def process(self, data):
        x = data[self.x_key][()]
        x = torch.from_numpy(x).float()
        y = torch.tensor(data[self.y_key][()]).float()
        return x, (y - 60) / (121 - 60)

    def __getitem__(self, idx):
        if self.in_memory:
            x, y = self.dataset[idx]
        else:
            idx = self.idx_to_filename[idx]
            x, y = self.process(self.dataset[idx])

        x = torch.corrcoef(x)
        x = torch.nan_to_num(x, 0)

        meta = {
            "subject": idx,
            "dataset_name": self.dataset_name,
            "task_name": self.task_name,
        }
        return_data = {"x": x, "y": y, "meta": meta}
        return return_data

    def _read_data(self):
        data = self.file

        file_names = data.keys()
        self.idx_to_filename = file_names
        if self.in_memory:
            data = [self.process(self.file[fn]) for fn in tqdm(file_names)]

        print(f"# instances of {self.dataset_name} {self.task_name}: ", len(file_names))

        return data


class BrainDataModule(LightningDataModule):
    def __init__(
        self, dataset_keys: list[str], y_key: str, batch_size: int, num_workers: int
    ) -> None:
        super().__init__()
        self.y_key = y_key

        if os.environ.get("WANDB_MODE", "") == "disabled":
            num_workers = 0

        self.batch_size = batch_size
        self.num_workers = num_workers

        datasets = []
        for dataset_key in dataset_keys:
            dataset_name, task_name = dataset_key.split("-")
            file_path = os.path.join("data", dataset_name, task_name, "data.h5")
            file = h5py.File(file_path, "r")
            datasets.append(
                BrainDataset(
                    file=file,
                    y_key=self.y_key,
                    dataset_name=dataset_name,
                    task_name=task_name,
                )
            )

        self.datasets = datasets

    def setup(self, stage: str):
        if stage != "fit":
            return

        datasets = {"train": defaultdict(list), "val": defaultdict(list), "test": defaultdict(list)}
        for dataset in self.datasets:
            dataset_name, task_name = dataset.dataset_name, dataset.task_name
            train_ids, test_ids = self.get_train_test_split(dataset_name, task_name)

            values = []
            for idx in train_ids:
                value = dataset[idx]["y"]
                values.append(value)
            values = np.array(values)
            max_ = values.max()
            min_ = values.min()
            values = np.digitize(
                values, bins=np.linspace(min_, max_, 6)[1:], right=True
            )
            if False:
                """
                NOTE: The performance is really unstable under different random seeds due to the size of the dataset. Therefore, we select the optimal hyperparameters on the validation dataset, and re-train the model using both of the training dataset and the validation dataset.
                """
                train_ids, val_ids, _, _ = train_test_split(
                    train_ids, values, test_size=1 / 8, stratify=values
                )
            datasets["train"][task_name].append(Subset(dataset, train_ids))

        datasets = dict(datasets)
        for split in datasets.keys():
            for task, datasets_of_task in datasets[split].items():
                datasets[split][task] = ConcatDataset(datasets_of_task)
        self.train = MultiSourceDataset(datasets["train"])

    def train_dataloader(self):
        batch_sampler = MultiSourceBatchSampler(
            self.train, batch_size=self.batch_size, shuffle=True
        )
        return DataLoader(
            self.train, batch_sampler=batch_sampler, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return [
            DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
            )
            for dataset in self.datasets
        ]

    def get_train_test_split(self, dataset_name, task_name):
        base_path = os.path.join("data", dataset_name, task_name)
        subjects = list(h5py.File(os.path.join(base_path, "data.h5")).keys())

        def get_indices_of_subjects(path):
            subject_ids = (
                open(os.path.join(base_path, path), "r").read().rstrip("\n").split("\n")
            )
            indices = [subjects.index(sid) for sid in subject_ids]
            return indices

        train_ids, test_ids = map(
            get_indices_of_subjects, ("train.split", "test.split")
        )
        return train_ids, test_ids

    @cached_property
    def entropy(self) -> dict[str, np.ndarray]:
        # collect all x in datasets
        all_data = defaultdict(list)
        for dataset in self.datasets:
            for sample in dataset:
                task = sample["meta"]["task_name"]
                x = sample["x"]
                all_data[task].extend(list(x))

        # compute entropy for each fMRI task
        entropy_all_task = dict()
        for task, x in all_data.items():
            x = torch.stack(x, dim=0)
            x = x.transpose(0, 1)
            x = list(x.numpy())

            bins = np.linspace(-1.0, 1.0, 100)
            entropies = []
            # compute entropy for each node
            for x_node in x:
                hist = np.histogram(x_node, bins=bins, density=True)[0]
                entropy = stats_entropy(hist)
                entropies.append(entropy)

            entropy_all_task[task] = np.asarray(entropies)

        return entropy_all_task
