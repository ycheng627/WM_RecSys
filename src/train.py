import torch
import scipy
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

def train_model(model, optimizer, use_BPR, train_loader, val_loader, device, n_ng):
    if not use_BPR:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = None
    model.to(device)
    best_loss = float('inf')

    t = trange(n_epochs, leave=True)
    for e in t:
        model.train()
        avg_loss = 0
        avg_loss_cnt = 0
        
        if not use_BPR:
            # Use BCE loss here
            for batch_num, batch in enumerate(train_loader):
                user, pos_item, neg_item = batch
                user = user.to(device)
                pos_item = pos_item.to(device)
                neg_item = neg_item.to(device)

                pos_label = torch.ones(pos_item.shape).to(device)
                neg_label = torch.zeros(neg_item.shape).to(device)
  
                    # BCE loss predict positive label
                optimizer.zero_grad() 
                pos_pred = model(user, pos_item)
                loss = criterion(pos_pred, pos_label)
                # print("pos loss", loss)
                avg_loss += loss
                avg_loss_cnt += 1
                loss.backward()
                optimizer.step()

                # BCE loss predict negative label
                optimizer.zero_grad() 
                neg_pred = model(user, neg_item)
                loss = criterion(neg_pred, neg_label)
                # print("neg loss", loss)
                avg_loss += loss
                avg_loss_cnt += 1
                loss.backward()
                optimizer.step()

        else:
            avg_loss = 0
            avg_loss_cnt = 0
            for batch_num, batch in enumerate(train_loader):
                user, pos_item, neg_item = batch
                user = user.to(device)
                pos_item = pos_item.to(device)
                neg_item = neg_item.to(device)
                
                optimizer.zero_grad()
                pos_pred = model(user, pos_item)
                neg_pred = model(user, neg_item)
                
                # BPR loss formula
                loss = -((pos_pred-neg_pred).sigmoid().log().sum())
                avg_loss += loss
                avg_loss_cnt += user.shape[0]
                
                loss.backward()
                optimizer.step()
                
        print("train loss: \t{:.4f}".format(avg_loss.item() / avg_loss_cnt), end = "\t")
        
        # Enter validation
        val_loss_sum, val_loss_cnt = 0, 0
        model.eval()
        for batch_num, batch in enumerate(val_loader):
            user, pos_item, neg_item = batch
            user = user.to(device)
            pos_item = pos_item.to(device)
            neg_item = neg_item.to(device)
            pos_pred = model(user, pos_item)
            neg_pred = model(user, neg_item)
            val_loss_sum += -((pos_pred-neg_pred).sigmoid().log().sum())
            val_loss_cnt += user.shape[0]
        loss = val_loss_sum.item() / val_loss_cnt
        print("val loss: \t{:.4f}".format(loss))

        torch.save(model.state_dict(), model_path)

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
    n_factors = 128
    learning_rate = 1
    wd =  0
    batch_size = 256
    n_epochs = 400
    use_BPR = False
    n_ng = 1
    train_ratio = 0.95
    train_path = "train.csv"
    model_path = "model/model-last-{}.pt".format(n_factors)
    predict_only = False

    n_users, n_items, ratings_list = load_data(train_path)
    # Use set for faster lookup
    ratings_set = set(ratings_list)
    train_loader, val_loader = init_dataloader(train_path, ratings_list, train_ratio)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MF(n_factors = n_factors, n_users = n_users, n_items = n_items)
    model = model.to(device)
    
    if not predict_only:
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=wd)
        train_model(model, optimizer, use_BPR, train_loader, val_loader, device, n_ng)
    
    model.load_state_dict(torch.load(model_path))
    pred = predict(model, n_users, n_items, ratings_set, device)

    with open("submission-{}.csv".format(n_factors), 'w') as f:
        f.write('UserId,ItemId\n')
        for user in range(len(pred)):
            f.write('{},{}\n'.format(str(user), ' '.join(str(a) for a in pred[user])))