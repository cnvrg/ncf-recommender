import pandas as pd
import numpy as np
import argparse
import NCF as ncf
import pytorch_lightning as pl
import torch
from transformers import logging
import json
import warnings
import math
import os
from pathlib import Path
import time
cnvrg_workdir = os.environ.get('CNVRG_WORKDIR', '/cnvrg')
np.random.seed(int(time.time()))

def add_timestamp_col(file_dir, filename):
    ''' Adds timestamp column to the dataframe and returns it, also stores unique item ids in a json file '''
    ratings = pd.read_csv(file_dir + "/" + filename)
    ratings['timestamp'] = np.random.randint(0, 50000, size=len(ratings))
    with open(file_dir + '/items_list.json', 'w') as outfile:
        json.dump(list(set(ratings['item_id'])), outfile)
    return ratings

def split_train_test(ratings):
    ''' Returns the train and test data based on rank
        Latest timestamp is categorised as the testing data 
    '''
    ratings['rank_latest'] = ratings.groupby(['user_id'])['timestamp'].rank(method='first', ascending=False)
    train_ratings = ratings[ratings['rank_latest'] != 1]
    test_ratings = ratings[ratings['rank_latest'] == 1]
    return train_ratings, test_ratings

def augment_data(train_ratings):
    ''' Augements the data if 'weights' are given in the input data '''
    print(len(train_ratings))
    added_lines = pd.DataFrame()
    for row in range(len(train_ratings)):
        for i in range(int(train_ratings.iloc[row]['weight'])):
            x = train_ratings.iloc[row].to_frame()
            added_lines = pd.concat([added_lines, x.T])
    added_lines.user_id = added_lines.user_id.astype(int)
    added_lines.item_id = added_lines.item_id.astype(int)
    train_ratings = pd.concat([train_ratings, added_lines])
    print(len(train_ratings))
    return train_ratings

def drop_colums(ratings, train_ratings, test_ratings):
    ''' Drop un-necessary columns '''
    if 'rating' in ratings.columns:
        train_ratings = train_ratings[['user_id', 'item_id', 'rating']]
        test_ratings = test_ratings[['user_id', 'item_id', 'rating']]
    else:
        train_ratings = train_ratings[['user_id', 'item_id']]
        test_ratings = test_ratings[['user_id', 'item_id']]
    return train_ratings, test_ratings

def get_data_attributes(ratings, train_ratings):
    ''' Returns data attributes '''
    train_ratings.loc[:, 'rating'] = 1
    num_users = ratings['user_id'].max() + 1
    num_items = ratings['item_id'].max() + 1
    all_item_ids = ratings['item_id'].unique()
    return train_ratings, num_users, num_items, all_item_ids

def fit_model(num_users, num_items, train_ratings, all_item_ids, batch_size, epochs, num_workers):
    ''' Calls the training function  '''
    model = ncf.NCF(num_users, num_items, train_ratings, all_item_ids, batch_size, num_workers)
    trainer = pl.Trainer(max_epochs=int(epochs), reload_dataloaders_every_n_epochs=True, enable_checkpointing=False, log_every_n_steps=1)
    # recommend_whole = pd.DataFrame(columns=['user_id', 'item_id', 'score'])
    trainer.fit(model)
    return model, trainer

def get_test_attributes(test_ratings, ratings):
    '''
        returns 
            (user:item) set
            (user: list(all items interacted by user)) dict 
    '''
    test_user_item_set = set(zip(test_ratings['user_id'], test_ratings['item_id']))
    # user1_movie_pred_whole = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'score', 'error'])
    user_interacted_items = ratings.groupby('user_id')['item_id'].apply(list).to_dict()
    return test_user_item_set, user_interacted_items

def get_recall(test_user_item_set, user_interacted_items, all_item_ids, model):
    ''' Calculates average recall '''
    recall = []
    for (u, i) in test_user_item_set:
        interacted_items = user_interacted_items[u]
        not_interacted_items = set(all_item_ids) - set(interacted_items)
        selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
        test_items = selected_not_interacted + [i]

        predicted_labels = np.squeeze(model(torch.tensor([u] * 100),
                                            torch.tensor(test_items)).detach().numpy())
        top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()]

        if i in top10_items:
            recall.append(1)
        else:
            recall.append(0)
    return np.average(recall)

def save_pt_model(model, file_dir, output_model_file):
    torch.save(model, file_dir + "/" + output_model_file)

if __name__ == "__main__":

    warnings.simplefilter(action='ignore', category=FutureWarning)
    logging.set_verbosity_error()

    parser = argparse.ArgumentParser(description="""Preprocessor""")
    parser.add_argument('-f', '--filename', action='store', dest='filename', required=True,
                        help="""string. csv recommender data""")

    parser.add_argument('--epochs', action='store', dest='epochs', default=5,
                        help="""int number of epochs""")

    parser.add_argument('--batch_size', action='store', dest='batch_size', default=512,
                        help="""int batch size""")

    parser.add_argument('--output_model_file', action='store', dest='output_model_file', default='model.pt',
                        help="""string. filename for saving the model""")

    parser.add_argument('--local_dir', action='store', dest='local_dir', default=cnvrg_workdir,
                        help="""string. local directory to store the data to """)

    parser.add_argument('--num_workers', action='store', dest='num_workers', default=4,
                        help="""string. num_workers paramter for pytorch model """)

    args = parser.parse_args()
    filename = args.filename
    epochs = int(args.epochs)
    batch_size = int(args.batch_size)
    output_model_file = args.output_model_file
    file_dir = args.local_dir
    num_workers = int(args.num_workers)

    ratings = add_timestamp_col(file_dir, filename)
    train_ratings, test_ratings = split_train_test(ratings)
    if 'weight' in ratings.columns:
        train_ratings = augment_data(train_ratings)

    train_ratings, test_ratings = drop_colums(ratings, train_ratings, test_ratings)
    # eval_metrics_whole = pd.DataFrame(columns=['user_id', 'rmse', 'precision', 'recall']) - is this used ?
 
    train_ratings, num_users, num_items, all_item_ids = get_data_attributes(ratings, train_ratings)
    model, trainer = fit_model(num_users, num_items, train_ratings, all_item_ids, batch_size, epochs, num_workers)
    test_user_item_set, user_interacted_items = get_test_attributes(test_ratings, ratings)

    average_recall = get_recall(test_user_item_set, user_interacted_items, all_item_ids, model)
    print("The hit ratio @ 10 is {:.2f}".format(average_recall))
    save_pt_model(model, file_dir, output_model_file)