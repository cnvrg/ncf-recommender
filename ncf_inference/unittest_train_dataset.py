import pandas
import unittest
import os
import sys
from TrainDataset import TrainDataset
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class test_train(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        ''' Function used to set up test dataframe and unittesting parameters '''
        # Create a dataframe for testing
        self.ratings = pandas.DataFrame({ 
            'user_id': [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4],
            'item_id': [0,0,0,1,1,1,2,2,2,3,3,3,4,4,4],
        })
        # Retrieve all unique items in a list
        self.all_item_ids = list(set(self.ratings['item_id']))

        # Call the get_dataset function on the above defined data
        self.td = TrainDataset(self.ratings, self.all_item_ids)
        self.data_item_label_tuple = set()
        for user, item, label in zip(list(self.td.users), list(self.td.items), list(self.td.labels)):
            self.data_item_label_tuple.add((user, item, label))

    def data_labels_helper(self):
        ''' Checks if get_dataset function assigns correct labels to each user, item pair - based on the data defined in setUpClass above '''
        for user, item, label in self.data_item_label_tuple:
            if (user == item):
                if label == 0:
                    return False
            else:
                if label == 1:
                    return False
        return True

    def test_data_labels(self):
        ''' Checks if the get_dataset function produces the correct labels '''
        self.assertTrue(
            self.data_labels_helper()
        )