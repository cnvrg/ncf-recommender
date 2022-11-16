import pandas
import unittest
import os
import sys
import json
import shutil
from train import add_timestamp_col, split_train_test, augment_data, drop_colums, get_data_attributes, fit_model, get_test_attributes, get_recall, save_pt_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class test_train(unittest.TestCase):
    @classmethod
    def setUpClass(self):
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
        self.model, self.trainer = fit_model(self.num_users, self.num_items, self.final_train_ratings, self.all_item_ids, self.batch_size, self.epochs)
        # Get test data attributes
        self.test_user_item_set, self.user_interacted_items = get_test_attributes(self.test_ratings, self.ratings)
        # Get the average recall for the model
        self.avg_recall = get_recall(self.test_user_item_set, self.user_interacted_items, self.all_item_ids, self.model)
        # Save the trained model
        self.output_model_file = 'model.pt'
        save_pt_model(self.model, self.data_path, self.output_model_file)

    @classmethod
    def tearDownClass(self):
        shutil.rmtree(self.data_path)

    def test_return_df(self):
        self.assertIsInstance(
            self.ratings, pandas.core.frame.DataFrame
        )

    def test_add_timestamp_col(self):
        self.assertTrue(
            'timestamp' in self.ratings.columns
        )
    
    def test_json_item_list(self):
        test_item_list = [i for i in range(self.num_users)]
        with open(self.data_path + '/' + 'items_list.json', 'r') as json_file:
            item_list = json.load(json_file)
        self.assertListEqual(
            item_list,
            test_item_list
        )

    def test_train_test_len(self):
        self.assertEqual(
            len(self.train_ratings), self.train_data_len
        )
        self.assertEqual(
            len(self.test_ratings), self.test_data_len
        )
    
    def test_augment_data_len(self):
        self.assertTrue(
            len(self.augment_train_ratings) > len(self.train_ratings)
        )

    def test_drop_columns(self):
        self.assertEqual(
            len(self.drop_train_ratings.columns),
            (2 or 3)
        )
        self.assertEqual(
            len(self.drop_test_ratings.columns),
            (2 or 3)
        )
        if len(self.drop_train_ratings.columns) == 2 and len(self.drop_test_ratings.columns) == 2:
            self.assertListEqual(
                list(self.drop_train_ratings.columns),
                ['user_id', 'item_id']
            )
            self.assertListEqual(
                list(self.drop_train_ratings.columns),
                ['user_id', 'item_id']
            )
        else:
            self.assertListEqual(
                list(self.drop_train_ratings.columns),
                ['user_id', 'item_id', 'rating']
            )
            self.assertListEqual(
                list(self.drop_train_ratings.columns),
                ['user_id', 'item_id', 'rating']
            )

    def test_data_attributes(self):
        self.assertEqual(self.num_users, self.test_num_users)
        self.assertEqual(self.num_items, self.test_num_items)
        self.assertListEqual(sorted(list(self.all_item_ids)), self.test_all_item_ids)

    def test_model_trainer(self):
        self.assertTrue(
            self.model
        )
        self.assertTrue(
            self.trainer
        )

    def test_average_recall_value(self):
        self.assertTrue(
            0.75 <= self.avg_recall <= 1.0
        )

    def test_model_file_exists(self):
        self.assertTrue(
            os.path.isfile(self.data_path + '/' + self.output_model_file)
        )
