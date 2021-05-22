import torch
import numpy as np
import random
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
from utils import RatingsDataset, MF

def randomized_split(data, ratio):
    random.shuffle(data)
    train_data = data[:int(ratio * len(data))]
    test_data = data[int(ratio * len(data)):]
    return train_data, test_data

def load_data(train_path):
    ratings_list = []
    n_users = 0
    n_items = 0

    # Read train data
    with open(train_path) as f:
        for no, line in enumerate(f):
            if no == 0: continue
            line = line.split(",")
            id = int(line[0])
            items = line[1].split()
            for item in items:
                ratings_list.append((id, int(item)))
                if id > n_users:
                    n_users  = id
                if int(item) > n_items:
                    n_items = int(item)

    # add one to adjust for 0-indexing
    n_users += 1
    n_items += 1
    return n_users, n_items, ratings_list

def init_dataloader(train_path, ratings_list, train_ratio):
    # Create dataset with train ratio of 0.9                
    train_ratings_list, val_ratings_list =  randomized_split(ratings_list, train_ratio)
    train_dataset = RatingsDataset(train_ratings_list, ratings_set, n_ng = n_ng, is_training = True)
    val_dataset = RatingsDataset(val_ratings_list, ratings_set, is_training = False)

    # Turn dataset into data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    return train_loader, val_loader

def predict(model, n_users, n_items, ratings_set, device):
    model.eval()
    answer = [[] for i in range(n_users)]
    for user in tqdm(range(n_users)):
        model_items = [item for item in range(n_items) if (user, item) not in ratings_set]
        model_users = [user] * len(model_items)
        
        preds = model(torch.tensor(model_users).to(device), torch.tensor(model_items).to(device))
        indices = torch.topk(preds, 50, sorted=True).indices.tolist()
        answer[user] = [model_items[i] for i in indices]
    return answer



if __name__ == "__main__":
    n_factors = 512
    learning_rate = 0.01
    wd =  1e-5
    batch_size = 256
    n_epochs = 100
    use_BPR = True
    n_ng = 1
    train_ratio = 0.95
    train_path = "train.csv"
    model_path = "model/best.pt"

    n_users, n_items, ratings_list = load_data(train_path)
    # Use set for faster lookup
    ratings_set = set(ratings_list)
    train_loader, val_loader = init_dataloader(train_path, ratings_list, train_ratio)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MF(n_factors = n_factors, n_users = n_users, n_items = n_items)
    model = model.to(device)
    
    model.load_state_dict(torch.load(model_path))
    pred = predict(model, n_users, n_items, ratings_set, device)

    with open("submission.csv", 'w') as f:
        f.write('UserId,ItemId\n')
        for user in range(len(pred)):
            f.write('{},{}\n'.format(str(user), ' '.join(str(a) for a in pred[user])))