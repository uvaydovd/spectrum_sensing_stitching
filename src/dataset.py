from torch.utils.data import Dataset
import torch
import h5py
class MultilabelDataset(Dataset):
    def __init__(self,h5file,norm=True) -> None:
        df = h5py.File(h5file)
        self.data = df['X']
        self.label = df['y']
        self.norm = norm
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        X = torch.from_numpy(self.data[idx].T)
        y = torch.from_numpy(self.label[idx][1:])
        return X, y
