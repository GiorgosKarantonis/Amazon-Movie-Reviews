# Amazon-Movie-Reviews

Before running any code make sure that the Datasets folder contains all the required datasets (train.csv and test.csv). 


## Sampling

In order to create the sampled dataset, run the samples.py file. 
This file samples the 0.1% of the training data points and the 0.1% of the testing data points to create a smaller dataset that can easily fit in the memory. 
The sampled datasets are saved in the Samples folder. 


## Preprocessing

The preprocessing of the dataset is performed by the preprocess.py file. As an outline, this file transforms the input data such that it can be used by machine learning models. 
Keep in mind that this file works with the sampled dataset. There are a couple of ways available to make it run to the full dataset, but the easiest is to edit the samples.py file by commenting out the following lines of code: 

train_df = train_df.sample(frac=0.001, random_state=1)
predict_df = predict_df.sample(frac=0.001, random_state=1)

Another easy way would be to change the frac parameter in both of the above lines of code to frac=1. 
Both of the ways proposed above are the easier to run but make take a bit of time. If you don't want to spend this time, replace the following lines of code in the preprocess.py file: 

train_df = pd.read_csv('./Samples/train.csv')
predict_df = pd.read_csv('./Samples/test.csv')

with: 

train_df, predict_df = df[df['Score'].notnull()], df[df['Score'].isnull()]

Regardless of the way you may choose, now you should be able to run preprocess.py to the full dataset. 
This file can also be used to run with google cloud, but all the lines that refer to fetching and storing to a google cloud bucket have been commented out. 


## Running the models

If you have followed the steps above, you should be able now to run the machine learning models defined in models.py. Keep in mind that this file was used just for testing in order to find the best model and write the report. 
To run the preferred model just uncomment it in the code and comment out the current uncommented model. 
When executed, this script prints out the accuracy and the predictions of the model. 
