from __future__ import division
import torch
import pickle
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Dataset

class TrajAugmenterWrapper(Dataset):
    def __init__(self, subset, augment=True, crop_ratio=0.7, views='OTR',
                 sub_ratio=0.15):
        self.subset = subset
        self.augment = augment
        self.crop_ratio = crop_ratio
        self.min_length = 1
        # Table IV: which augmented views to generate (letters T/R/S; O always)
        self.views = views.upper()
        self.sub_ratio = sub_ratio
        # Access original dataset attributes through subset.dataset
        self.padding_idx = subset.dataset.padding_idx
        self._max_len = subset.dataset.poi_list.shape[1]
        # Substitution view samples random valid POI ids in [1, vocab_hi]
        self.vocab_hi = int(subset.dataset.poi_list.max().item())

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, index):
        # Get original data
        poi_seq, user_label, seq_length, orig_mask = self.subset[index]

        # Convert to numpy for processing
        poi_np = poi_seq.numpy()
        valid_length = seq_length.item()

        # Generate augmented data
        if self.augment:
            item = {
                'orig': (poi_seq, orig_mask),
                'user': user_label,
                'lengths': (valid_length, valid_length, valid_length)
            }
            # Random cropping (maintain padding structure)
            if 'T' in self.views:
                crop_poi, crop_length = self._random_crop(poi_np, valid_length)
                crop_poi = torch.LongTensor(crop_poi)
                item['crop'] = (crop_poi, (crop_poi != self.padding_idx))
                item['crop_length'] = crop_length
            # Reverse augmentation (maintain original padding positions)
            if 'R' in self.views:
                reverse_poi = torch.LongTensor(self._reverse_with_pad(poi_np, valid_length))
                item['reverse'] = (reverse_poi, (reverse_poi != self.padding_idx))
            # Substitution augmentation: replace ~15% of valid positions with
            # random valid POI ids (revision R1.1/R2.5 third augmentation)
            if 'S' in self.views:
                subst_poi = torch.LongTensor(self._random_substitute(poi_np, valid_length))
                item['subst'] = (subst_poi, (subst_poi != self.padding_idx))
            return item
        else:
            return {
                'orig': (poi_seq, orig_mask),
                'user': user_label,
                'lengths': valid_length
            }

    def _random_crop(self, seq, valid_length):
        """Random cropping while maintaining padding structure"""
        if valid_length <= self.min_length:
            return seq.copy(), valid_length

        crop_length = max(int(valid_length * self.crop_ratio), self.min_length)
        start = np.random.randint(0, valid_length - crop_length + 1)

        # Create new sequence and preserve padding
        cropped = np.full_like(seq, self.padding_idx)
        cropped[:crop_length] = seq[start:start + crop_length]
        return cropped, crop_length

    def _reverse_with_pad(self, seq, valid_length):
        """Reverse only valid part, preserve padding"""
        # Ensure data type consistency
        seq = np.array(seq, dtype=np.int64)

        # Reverse valid part
        reversed_part = seq[valid_length - 1::-1]

        # Create padding part, ensure type consistency
        padding_part = np.full(len(seq) - valid_length, self.padding_idx, dtype=np.int64)

        # Concatenate arrays
        reversed_seq = np.concatenate([reversed_part, padding_part])

        return reversed_seq

    def _random_substitute(self, seq, valid_length):
        """Substitution view: replace ~sub_ratio of the valid positions with
        random valid POI ids (padding structure preserved)."""
        seq = np.array(seq, dtype=np.int64)
        out = seq.copy()
        if valid_length >= 2:
            n_sub = max(int(valid_length * self.sub_ratio), 1)
            pos = np.random.choice(valid_length, size=n_sub, replace=False)
            out[pos] = np.random.randint(1, self.vocab_hi + 1, size=n_sub)
        return out


def aug_collate_fn(batch):
    """Unified collate function for processing augmented data"""
    # Initialize containers
    orig_seq = []
    crop_seq = []
    reverse_seq = []
    subst_seq = []
    users = []
    masks = {'orig': [], 'crop': [], 'reverse': [], 'subst': []}
    lengths = {'orig': [], 'crop': [], 'reverse': [], 'subst': []}
    decoder_inputs = []  # Decoder input container

    # Unpack batch
    for item in batch:
        # Original data
        orig_poi = item['orig'][0]
        orig_seq.append(orig_poi)
        masks['orig'].append(item['orig'][1])
        _len = item['lengths']
        _vlen = _len[0] if isinstance(_len, (tuple, list)) else _len
        lengths['orig'].append(_vlen)  # Store int value directly

        # Generate decoder input (key correction)
        seq = orig_poi.tolist()
        valid_length = _vlen  # Get int value directly, no need for .item()
        decoder_seq = seq[:valid_length-1] + [0]*(len(seq)-(valid_length-1))
        decoder_inputs.append(torch.LongTensor(decoder_seq))

        # Augmented data (each view independent; Table IV subsets)
        if 'crop' in item:
            crop_seq.append(item['crop'][0])
            masks['crop'].append(item['crop'][1])
            lengths['crop'].append(item.get('crop_length', _len[1] if isinstance(_len, (tuple, list)) else _vlen))
        if 'reverse' in item:
            reverse_seq.append(item['reverse'][0])
            masks['reverse'].append(item['reverse'][1])
            lengths['reverse'].append(_vlen)
        if 'subst' in item:
            subst_seq.append(item['subst'][0])
            masks['subst'].append(item['subst'][1])
            lengths['subst'].append(_vlen)

        users.append(item['user'])

    # Convert to tensors
    batch_dict = {
        'orig_seq': torch.stack(orig_seq),
        'orig_mask': torch.stack(masks['orig']),
        'users': torch.tensor(users),
        'orig_length': torch.tensor(lengths['orig']),
        'decoder_input': torch.stack(decoder_inputs),
        'decoder_mask': torch.stack(masks['orig'])[:, 1:]
    }

    # Add augmented data
    if len(crop_seq) > 0:
        batch_dict['crop_seq'] = torch.stack(crop_seq)
        batch_dict['crop_mask'] = torch.stack(masks['crop'])
        batch_dict['crop_length'] = torch.tensor(lengths['crop'])
    if len(reverse_seq) > 0:
        batch_dict['reverse_seq'] = torch.stack(reverse_seq)
        batch_dict['reverse_mask'] = torch.stack(masks['reverse'])
        batch_dict['reverse_length'] = torch.tensor(lengths['reverse'])
    if len(subst_seq) > 0:
        batch_dict['subst_seq'] = torch.stack(subst_seq)
        batch_dict['subst_mask'] = torch.stack(masks['subst'])
        batch_dict['subst_length'] = torch.tensor(lengths['subst'])

    return batch_dict


def read_processed_tra(traj_path):
    with open(traj_path, 'rb') as f:
        trajectories = pickle.load(f)  # pickle.load() function deserializes data from file into Python objects
    return trajectories
def get_embedding_vector(vec_path, embed_size):
    # embed_size is the dimension size of embedding vectors. In this function, we use embed_size to initialize an empty list out_vec
    out_vec = []
    with open(vec_path, 'r') as f:
        for line in f.readlines():
            line_Arr = line.split()  # Split the line into a string list using space as delimiter
            if len(line_Arr) < embed_size + 1 or line_Arr[0] == '</s>':  # Skip header/invalid lines (POI id + embed_size dims)
                continue
            out_vec.append(list(map(float, line_Arr[1:])))  # Append float list starting from second element to out_vec
        vec_tensor = torch.tensor(out_vec)  # Function converts out_vec to PyTorch tensor and returns it
    return vec_tensor

def accuracy_at_k(predicted_labels, true_labels, k):
    if len(predicted_labels) != len(true_labels):
        raise ValueError("Predicted labels and true labels must have the same length.")

    total_samples = len(predicted_labels)
    correct_at_k = 0

    for i in range(total_samples):
        if isinstance(predicted_labels[i], (list, np.ndarray)):
            top_k_predictions = predicted_labels[i][:k]
        else:
            # Handle the case where predicted_labels[i] is an integer
            top_k_predictions = [predicted_labels[i]]

        if true_labels[i] in top_k_predictions:
            correct_at_k += 1

    accuracy = correct_at_k / total_samples
    return accuracy

def calculate_macro_metrics(predicted_labels, true_labels):
    # Calculate precision, recall and F1 score
    precision = precision_score(true_labels, predicted_labels, average='macro', zero_division=1)
    recall = recall_score(true_labels, predicted_labels, average='macro', zero_division=1)
    f1 = f1_score(true_labels, predicted_labels, average='macro', zero_division=1)
    return precision, recall, f1
def read_trajectories(data_file):
    trajectories = []
    users = []
    with open(data_file, 'r') as f:
        lines = f.readlines()  # Read file line by line
        for line in lines:
            trajectories.append(line.strip('\n').split()[1:])  # Store elements from second to last in trajectory, remove all spaces and newlines
            users.append(line.split()[0])  # First element after removing spaces as user
    return users, trajectories  # Return two lists