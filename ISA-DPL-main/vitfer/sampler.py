import torch
import torch.utils.data
import random
is_torchvision_installed = True
try:
    import torchvision
except:
    is_torchvision_installed = False


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """
    def __init__(self, dataset, indices=None, num_samples=None):
        self.indices = list(range(len(dataset))) if indices is None else indices
        self.num_samples = len(self.indices) if num_samples is None else num_samples

        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]

        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        # ImageFolder专属
        # info = dataset.imgs
        # path, label = info[idx]
        # return label
        return dataset.label_list[idx]

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples
