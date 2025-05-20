import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset.mgg_paired_dataset import PairedGraspDataset

def single_object_collate(batch):
    # With batch_size=1, return the single sample without extra nesting.
    return batch[0]

class GraspDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, hand1, hand2, batch_size, contact=True, gwh=False, jac=False):
        """
        Args:
            data_dir (str): root directory of your dataset.
            hand1, hand2 (str): directory names for the two hands.
            batch_size (int): the internal batch size for each object (used by your Dataset).
            num_workers (int): number of workers for the DataLoader.
            contact (bool): whether to load contact points.
        """
        super().__init__()
        self.data_dir = data_dir
        self.hand1 = hand1
        self.hand2 = hand2
        self.batch_size = batch_size  # used in dataset __init__
        self.contact = contact
        self.gwh = gwh
        self.jac = jac

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = PairedGraspDataset(
                root_dir=self.data_dir, hand1=self.hand1, hand2=self.hand2,
                split="train", contact=self.contact, gwh=self.gwh, jac=self.jac,
                batch_size=self.batch_size
            )
            self.val_dataset = PairedGraspDataset(
                root_dir=self.data_dir, hand1=self.hand1, hand2=self.hand2,
                split="val", contact=self.contact, gwh=self.gwh, jac=self.jac,
                batch_size=self.batch_size
            )
        if stage == 'test':
            self.test_dataset = PairedGraspDataset(
                root_dir=self.data_dir, hand1=self.hand1, hand2=self.hand2,
                split="test", contact=self.contact, gwh=self.gwh, jac=self.jac,
                batch_size=self.batch_size
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=1, shuffle=False,
            num_workers=1, collate_fn=single_object_collate,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=1,
            num_workers=1, collate_fn=single_object_collate,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=1,
            num_workers=1, collate_fn=single_object_collate,
            persistent_workers=True
        )

if __name__ == "__main__":
    from tqdm import tqdm
    # from utils.loss import set_random_seed
    # set_random_seed(0)
    data_dir = '/data/XXX/data/grasp_data'
    hand1 = "HumanHand"
    hand2 = "Allegro"
    batch_size = 256
    dm = GraspDataModule(data_dir, hand1, hand2, batch_size)
    dm.setup()
    dl = dm.train_dataloader()
    for epoch in range(10):
        for i, batch in tqdm(enumerate(dl), total=len(dl)):
            if i == 0:
                print(batch['indices1'][:5])
                print(batch["hand1_latent"].shape)
                print(batch["hand2_latent"].shape)
                print(batch["object_global_latent"].shape)
                print(batch["object_local_latent"].shape)
                if "contact_pc_1" in batch:
                    print(batch["contact_pc_1"].shape)
                    print(batch["contact_pc_2"].shape)
