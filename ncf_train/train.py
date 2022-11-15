# Copyright (c) 2022 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# SPDX-License-Identifier: MIT

import pandas as pd
import numpy as np

np.random.seed(123)
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

cnvrg_workdir = os.environ.get('CNVRG_WORKDIR', '/cnvrg')

args = parser.parse_args()
filename = args.filename
epochs = int(args.epochs)
batch_size = int(args.batch_size)
output_model_file = args.output_model_file

ratings = pd.read_csv(filename)
ratings['timestamp'] = np.random.randint(0, 50000, size=len(ratings))
with open(cnvrg_workdir + '/items_list.json', 'w') as outfile:
    json.dump(list(set(ratings['item_id'])), outfile)

ratings['rank_latest'] = ratings.groupby(['user_id'])['timestamp'].rank(method='first', ascending=False)

train_ratings = ratings[ratings['rank_latest'] != 1]
test_ratings = ratings[ratings['rank_latest'] == 1]
print(train_ratings.dtypes)

# If we have a weight column - duplicate each row x its weight times to make it have more/less impact on the model.
if 'weight' in ratings.columns:
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


# drop columns that we no longer  need
if 'rating' in ratings.columns:
    train_ratings = train_ratings[['user_id', 'item_id', 'rating']]
    test_ratings = test_ratings[['user_id', 'item_id', 'rating']]
else:
    train_ratings = train_ratings[['user_id', 'item_id']]
    test_ratings = test_ratings[['user_id', 'item_id']]

eval_metrics_whole = pd.DataFrame(columns=['user_id', 'rmse', 'precision', 'recall'])

train_ratings.loc[:, 'rating'] = 1
num_users = ratings['user_id'].max() + 1
num_items = ratings['item_id'].max() + 1
all_item_ids = ratings['item_id'].unique()

print(type(num_items))
print(type(num_users))
print(type(all_item_ids))
print(type(batch_size))

model = ncf.NCF(num_users, num_items, train_ratings, all_item_ids, batch_size)

trainer = pl.Trainer(max_epochs=int(epochs), gpus=1, reload_dataloaders_every_n_epochs=True, enable_checkpointing=False)
recommend_whole = pd.DataFrame(columns=['user_id', 'item_id', 'score'])

trainer.fit(model)

# User-item pairs for testing
test_user_item_set = set(zip(test_ratings['user_id'], test_ratings['item_id']))
user1_movie_pred_whole = pd.DataFrame(columns=['user_id', 'item_id', 'rating', 'score', 'error'])

# Dict of all items that are interacted with by each user
user_interacted_items = ratings.groupby('user_id')['item_id'].apply(list).to_dict()

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

print("The hit ratio @ 10 is {:.2f}".format(np.average(recall)))
torch.save(model, cnvrg_workdir + "/" + output_model_file)
