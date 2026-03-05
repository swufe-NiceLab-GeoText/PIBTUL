import torch
from torch.utils.data import TensorDataset



class TrajDataset(TensorDataset):
    def __init__(self, traj_data, traj_user, padding_idx, use_sos_eos):
        # Don't call super().__init__() here as we're building tensors manually
        # Read data files
        self.data = traj_data
        self.user = traj_user
        self.padding_idx = padding_idx
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.sos_eos = use_sos_eos

        # Build vocabulary and re-encode data
        trajectories = self.data
        userids = self.user
        voc_poi = self.build_dictionary(trajectories)  # Add unique POI points to voc_poi list
        voc_user = self.user_dictionary(userids)  # Add unique users to voc_user list
        # Two dictionaries (int_to_vocab and vocab_to_int) for converting between words and integers
        int_to_vocab, vocab_to_int = self.extract_words_vocab(voc_poi)
        int_to_vocab_user, vocab_to_int_user = self.extract_words_vocab(voc_user)

        trajectories = self.convert_data(trajectories, vocab_to_int)  # Convert words in text data to corresponding integer representation
        userids = self.convert_user(userids, vocab_to_int_user)  # Convert words in text data to corresponding integer representation
        # Get data labels
        self.voc_user = voc_user
        self.voc_poi = voc_poi
        # map() function applies a given function to each element in an iterable object and returns an iterator that produces results. In this example, map(int, self.user)
        # applies the int() function to each element in self.user and returns an iterator that generates results as integers
        # user_label = list(map(int, self.user))
        user_label = list(userids)

        # Data padding and start/end token setup
        max_id = max(list(int_to_vocab.keys()))  # Find the maximum value in all trajectories
        sos = [max_id + 1]
        eos = [max_id + 2]
        pad = self.padding_idx
        # Pad the data
        poi_list, lengths_list = self.pad_sentence_batch(trajectories, pad)


        # Add start and end tokens to data
        if self.sos_eos:
            poi_list = self.add_sos_eos(poi_list, sos, eos)
        # Set mask matrix
        mask = torch.tensor(poi_list).gt(0)
        # gt(0) operation sets values greater than 0 in poi_list to True and other values to False, generating a boolean tensor with the same dimensions as poi_list. This operation marks non-zero parts of the data as True and zero parts as False. For example, if poi_list is a 1D tensor [1, 0, 3, -2, 0, 5], then the mask tensor will be [True, False, True, False, False, True]

        # Move data to GPU
        self.user_label = torch.tensor(user_label)
        self.poi_list = torch.tensor(poi_list)
        self.lengths_list = torch.tensor(lengths_list)
        self.mask = torch.tensor(mask)

    def __len__(self):
        # Return the size of the dataset
        return len(self.data)

    # Support accessing objects in the class using index operator []. This method takes an index parameter as input and returns a tuple of four values
    def __getitem__(self, index):
        return self.poi_list[index], self.user_label[index], self.lengths_list[index], self.mask[index]

    def build_dictionary(self, data):  # Add unique POI points to voc_poi list
        voc_poi = []
        for traj_list in data:
            for poi in traj_list:
                if poi not in voc_poi:
                    voc_poi.append(poi)
        return voc_poi

    def user_dictionary(self, data):  # Add unique user IDs to voc_user list
        voc_user = []
        for user in data:
            if user not in voc_user:
                voc_user.append(user)
        return voc_user

    def extract_words_vocab(self, voc):
        int_to_vocab = {idx + 1: word for idx, word in enumerate(voc)}  # Use enumerate function and for loop to pair each word with an integer index
        vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}  # Dictionary mapping words to corresponding integers and dictionary mapping integers to words
        return int_to_vocab, vocab_to_int  # This method returns these two dictionaries (int_to_vocab and vocab_to_int). These dictionaries can be used for mutual conversion between words and integers

    # Function to convert words in a set of text data to corresponding integer representation
    def convert_data(self, DATA, vocab_to_int):
        new_DATA = list()
        for i in range(len(DATA)):  # TRAIN
            temp = list()
            for j in range(len(DATA[i])):
                temp.append(vocab_to_int[DATA[i][j]])
            new_DATA.append(temp)
        return new_DATA

    # Function to convert words in a set of text data to corresponding integer representation
    def convert_user(self, DATA, vocab_to_int):
        new_user = list()
        for i in range(len(DATA)):  # TRAIN
            # temp = list()
            # temp.append(vocab_to_int[DATA[i]])
            new_user.append(vocab_to_int[DATA[i]])
        return new_user

    def create_trajectories(self, data, voc_poi):
        int_to_vocab, vocab_to_int = self.extract_words_vocab(voc_poi)
        trajectories = self.convert_data(data, vocab_to_int)
        return trajectories

    # Pad input text sequence list to make them consistent in length, and return padded results and original sequence length list
    def pad_sentence_batch(self, sentence_batch, pad_idx):
        max_sentence = max([len(sentence) for sentence in sentence_batch])  # Get maximum length
        lengths_list = [len(sentence) for sentence in sentence_batch]
        return [sentence + [pad_idx] * (max_sentence - len(sentence)) for sentence in sentence_batch], lengths_list

    def add_sos_eos(self, trajectoreis, sos, eos):
        # Add start and end symbols
        for i in range(len(trajectoreis)):
            trajectoreis[i] = sos + trajectoreis[i] + eos
        return trajectoreis

    def get_label(self, trajectoreis):
        label = []
        for i in range(len(trajectoreis)):
            label.append(trajectoreis[i][-1])
        return label

    def del_label(self, trajectoreis):  # Remove the last POI from trajectory
        nolabel_trajectorties = []
        for trajectory in trajectoreis:
            nolabel_trajectorties.append(trajectory[:-1])
        return nolabel_trajectorties