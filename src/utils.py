from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np


class RatingsDataset(Dataset):
    def __init__(self, ratings_list, ratings_set, n_users = 4454, n_items = 3260, n_ng = 1, is_training=None):
        """
        Ratings is a scipy sparse matrix, list of keys (each of which have val of 1)
        """
        self.ratings_list = ratings_list
        self.ratings_set = ratings_set
        self.n_users = n_users
        self.n_items = n_items
        self.n_ng = n_ng
        self.is_training = is_training
        

    def __len__(self):
        return len(self.ratings_list)

    def __getitem__(self, idx):
        rating = self.ratings_list[idx]
        user, pos_item = rating


        neg_item = np.random.randint(self.n_items)
        while (user, neg_item) in self.ratings_set:
            neg_item = np.random.randint(self.n_items)
        return user, pos_item, neg_item
        

class MF(torch.nn.Module):
    def __init__(self, n_factors = 16, n_users = 4454, n_items = 3260):
        super(MF, self).__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)

        torch.nn.init.normal_(self.user_factors.weight, std=1)
        torch.nn.init.normal_(self.item_factors.weight, std=1)

        print(n_factors, n_users, n_items)
        
    def forward(self, user, item):
        return (self.user_factors(user) * self.item_factors(item)).sum(1)