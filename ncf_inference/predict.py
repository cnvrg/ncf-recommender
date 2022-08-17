import numpy as np
np.random.seed(123)
import torch
import json
import pathlib
import sys
import os 

scripts_dir = pathlib.Path(__file__).parent.resolve()
sys.path.append(str(scripts_dir))

default_items_file = os.path.join(scripts_dir, 'items_list.json')
default_model_file = os.path.join(scripts_dir, 'model.pt')

items_file = os.environ.get('items_file', default_items_file)
model_file = os.environ.get('model_file', default_model_file)

if os.path.exists('/input/train'):
    model_path = '/input/train/model.pt'
    items_path = '/input/train/items_list.json'

else:
    model_path = model_file
    items_path = items_file

device = torch.device("cpu")
model = torch.load(model_path)

print(items_path)
items_file = open(items_path)
items_list = json.load(items_file)

def predict(data):
    user_num = int(data['user_id'])
    predicted_labels = np.squeeze(model(torch.tensor([user_num]*len(items_list)),
                                        torch.tensor(items_list)).detach().numpy())
    top5_items = [items_list[i] for i in np.argsort(predicted_labels)[::-1][0:5].tolist()]
    return {'recommendations': top5_items}