from __future__ import division
import torch
import pickle
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import Dataset

class TrajAugmenterWrapper(Dataset):
    def __init__(self, subset, augment=True):
        self.subset = subset
        self.augment = augment
        self.min_length = 1
        # Access original dataset attributes through subset.dataset
        self.padding_idx = subset.dataset.padding_idx
        self._max_len = subset.dataset.poi_list.shape[1]

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
            # Random cropping (maintain padding structure)
            crop_poi, crop_length = self._random_crop(poi_np, valid_length)
            # Reverse augmentation (maintain original padding positions)
            reverse_poi = self._reverse_with_pad(poi_np, valid_length)

            # Convert to Tensor
            crop_poi = torch.LongTensor(crop_poi)
            reverse_poi = torch.LongTensor(reverse_poi)

            # Generate new masks
            crop_mask = (crop_poi != self.padding_idx)
            reverse_mask = (reverse_poi != self.padding_idx)

            return {
                'orig': (poi_seq, orig_mask),
                'crop': (crop_poi, crop_mask),
                'reverse': (reverse_poi, reverse_mask),
                'user': user_label,
                'lengths': (valid_length, crop_length, valid_length)
            }
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

        crop_length = max(int(valid_length * 0.7), self.min_length)
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


def aug_collate_fn(batch):
    """Unified collate function for processing augmented data"""
    # Initialize containers
    orig_seq = []
    crop_seq = []
    reverse_seq = []
    users = []
    masks = {'orig': [], 'crop': [], 'reverse': []}
    lengths = {'orig': [], 'crop': [], 'reverse': []}
    decoder_inputs = []  # Decoder input container

    # Unpack batch
    for item in batch:
        # Original data
        orig_poi = item['orig'][0]
        orig_seq.append(orig_poi)
        masks['orig'].append(item['orig'][1])
        lengths['orig'].append(item['lengths'][0])  # Store int value directly

        # Generate decoder input (key correction)
        seq = orig_poi.tolist()
        valid_length = item['lengths'][0]  # Get int value directly, no need for .item()
        decoder_seq = seq[:valid_length-1] + [0]*(len(seq)-(valid_length-1))
        decoder_inputs.append(torch.LongTensor(decoder_seq))

        # Augmented data
        if 'crop' in item:
            crop_poi = item['crop'][0]
            crop_seq.append(crop_poi)
            masks['crop'].append(item['crop'][1])
            reverse_poi = item['reverse'][0]
            reverse_seq.append(reverse_poi)
            masks['reverse'].append(item['reverse'][1])
            lengths['crop'].append(item['lengths'][1])
            lengths['reverse'].append(item['lengths'][2])

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
        batch_dict.update({
            'crop_seq': torch.stack(crop_seq),
            'crop_mask': torch.stack(masks['crop']),
            'crop_length': torch.tensor(lengths['crop']),
            'reverse_seq': torch.stack(reverse_seq),
            'reverse_mask': torch.stack(masks['reverse']),
            'reverse_length': torch.tensor(lengths['reverse'])
        })

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
            if len(line_Arr) < 100 or line_Arr[0] == '</s>':  # If list length is less than 100 or first element is '</s>', skip this line and move to next
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