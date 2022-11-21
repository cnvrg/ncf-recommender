import pandas
import unittest
import os
import sys
import json
import shutil
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/' + 'ncf_train')
from train import add_timestamp_col, split_train_test, augment_data, drop_colums, get_data_attributes, fit_model, get_test_attributes, get_recall, save_pt_model
from predict import predict

class test_predict(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        ''' Function used to set up test dataframe and unittesting parameters '''
        # Create data for testing and define data parameters
        df = pandas.DataFrame({ 
            'user_id': [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4],
            'item_id': [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4],
            'weight':  [3,20,2,3,20,2,3,20,2,3,20,2,3,20,2]
        })
        self.test_data_len = len(set(df['user_id']))
        self.train_data_len = len(df) - self.test_data_len
        self.test_num_users = len(set(df['user_id']))
        self.test_num_items = len(set(df['item_id']))
        self.test_all_item_ids = list(set(df['item_id']))
        self.epochs = 100
        self.batch_size = 512
        self.num_workers = 0

        # Make a Unit-testing directory
        self.unittest_dir = "unit_test_data"
        self.local_dir = os.path.dirname(os.path.abspath(__file__))
        self.data_path = os.path.join(self.local_dir, self.unittest_dir)
        os.mkdir(self.data_path)

        # Store the csv data in the unittesting directory
        df.to_csv(self.data_path + '/' + 'data.csv', index=False)

        # Create test data frame
        self.ratings = add_timestamp_col(self.data_path, 'data.csv')
        # Split data into train and test
        self.train_ratings, self.test_ratings = split_train_test(self.ratings)
        # Augment train data
        self.augment_train_ratings = augment_data(self.train_ratings)
        # Drop the un-necessary columns 
        self.drop_train_ratings, self.drop_test_ratings = drop_colums(self.ratings, self.train_ratings, self.test_ratings)
        # Get data attributes and final data processing
        self.final_train_ratings, self.num_users, self.num_items, self.all_item_ids = get_data_attributes(self.ratings, self.train_ratings)
        # Train the model on the above defined data
        self.model, self.trainer = fit_model(self.num_users, self.num_items, self.final_train_ratings, self.all_item_ids, self.batch_size, self.epochs, self.num_workers)
        # Get test data attributes
        self.test_user_item_set, self.user_interacted_items = get_test_attributes(self.drop_test_ratings, self.ratings)
        # Get the average recall for the model
        self.avg_recall = get_recall(self.test_user_item_set, self.user_interacted_items, self.all_item_ids, self.model)
        # Save the trained model
        self.output_model_file = 'model.pt'
        save_pt_model(self.model, self.data_path, self.output_model_file)
        # Define parameters for testing 'predict' function
        self.items_file = os.environ.get('items_file', self.data_path + '/' + 'items_list.json')
        self.model_file = os.environ.get('model_file', self.data_path + '/' + self.output_model_file)
        self.json_f = open(self.items_file)
        self.load_json_items = json.load(self.json_f)
        self.load_model = torch.load(self.model_file)
        
    @classmethod
    def tearDownClass(self):
        ''' Clean up function '''
        self.json_f.close()
        shutil.rmtree(self.data_path)

    def test_predict_function(self):
        ''' Tests out the prediction based on the test-data defined in the setUpClass above '''
        for i in range(self.test_data_len):
            prediction_recommendations = predict(self.drop_test_ratings.iloc[i:i+1], self.load_model, self.load_json_items)
            label = int(self.drop_test_ratings.iloc[i:i+1]['item_id'])
            if label != prediction_recommendations['recommendations']:
                return False
        return True