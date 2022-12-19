# NCF Recommender Train  
The NCF blueprint is a blueprint that trains a neural network based recommender system.

[![N|Solid](https://cnvrg.io/wp-content/uploads/2018/12/logo-dark.png)](https://nodesource.com/products/nsolid)

# Retrain
This library is used to retrain the neural network on a custom dataset.
As a result, we get a model file that can be used to recommend items to users. 
### Flow
- The user has to upload the training dataset which is a csv file that includes 2 columns: user_id and item_id.
 You can also train the model on weighted data. To do so, the training data should include an extra colum named 'weight'.
- The model is trained on the dataset and a model file is produced.

### Inputs
- `--filename` refers to the training dataset (required=True).
- `--epochs` refers to the number of epochs in the train process (required=False).
- `--batch_size` refers to the batch size in the train process (required=False).
- `--output_model_file` refers to the filename for saving the model (required=False).
- `--local_dir` refers to the local directory to store the data to (required=False).
- `--num_workers` refers to the num_workers paramter for pytorch model (required=False).

For more information regarding input arguments:
```
python3 train.py --help
```

 
## How to run
```
python3 train.py --filename <name of data file> --epochs <number of epochs>
```
Example:
```
python3 train.py --filename 'data.csv' --epochs 5
```
