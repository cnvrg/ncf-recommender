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

import numpy as np
np.random.seed(123)
import torch
import json
import pathlib
import sys
import os 

def predict(data, model, items_list):
    user_num = int(data['user_id'])
    predicted_labels = np.squeeze(model(torch.tensor([user_num]*len(items_list)),
                                        torch.tensor(items_list)).detach().numpy())
    top5_items = [items_list[i] for i in np.argsort(predicted_labels)[::-1][0:5].tolist()]
    return {'recommendations': top5_items}

if __name__ == "__main__":
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