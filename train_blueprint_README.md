Use this blueprint to train a custom neural network recommender model, which can recommend similar items to customers according to their behaviors. For the model to learn each customer’s choices and predict recommendations, the blueprint requires data in the form of a customer’s preference on current items. Model predictions are based directly on scores, which are essentially predicted ratings for all items rather than just the ones the customer has already viewed and rated. This blueprint also establishes an endpoint that recommends similar items according to customer behavior based on the newly trained model.

To train this model with your data, provide one folder in the S3 Connector with a CSV file that includes two columns: `user_id` and `item_id`. The model can also be trained on weighted data. To do so, include an extra column named 'weight' in the training data CSV file.

Complete the following steps to train this NCF-recommender model:
1. Click the **Use Blueprint** button. The cnvrg Blueprint Flow page displays.
2. In the flow, click the **S3 Connector** task to display its dialog.
   - Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `bucketname` - Value: enter the data bucket name
     - Key: `prefix` - Value: provide the main path to the CVS file folder
   - Click the **Advanced** tab to change resources to run the blueprint, as required.
3. Return to the flow and click the **Train** task to display its dialog.
   - Within the **Parameters** tab, provide the following Key-Value pair information:
     - Key: `filename` − Value: provide the path to the CSV file including the S3 prefix in the following format: `/input/s3_connector/<prefix>/<csv file>`
     - Key: `epochs` − Value: set the number of times the model passes over the dataset
     - Key: `batch_size` − Value: set the number of times the model evaluates in each epoch
     NOTE: You can use the prebuilt example data paths provided.
   - Click the **Advanced** tab to change resources to run the blueprint, as required.
5. Click the **Run** button. The cnvrg software launches the training blueprint as set of experiments, generating a trained NCF-recommender model and deploying it as a new API endpoint.
6. Track the blueprint's real-time progress in its Experiments page, which displays artifacts such as logs, metrics, hyperparameters, and algorithms.
7. Click the **Serving** tab in the project, locate your endpoint, and complete one or both of the following options:
   - Use the Try it Live section with a relevant user ID to check the model's predictions.
   - Use the bottom Integration panel to integrate your API with your code by copying in your code snippet.

A custom NCF-recommender model and an API endpoint, which can recommend similar items to customers according to their behavior, have now been trained and deployed. To learn how this blueprint was created, click [here](https://github.com/cnvrg/ncf-recommender).









