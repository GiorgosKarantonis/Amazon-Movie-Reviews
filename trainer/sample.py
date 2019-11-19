import pandas as pd



def split_train_predict(df):
	'''
		Keeps as training set the points where Score has a value. 
		The rest are used as the test set. 
	'''
	return df[df['Score'].notnull()], df[df['Score'].isnull()]



df = pd.read_csv('./Datasets/train.csv')

train_df, predict_df = split_train_predict(df)

train_df = train_df.sample(frac=0.001, random_state=1)
predict_df = predict_df.sample(frac=0.001, random_state=1)

train_df.to_csv('./Samples/train.csv', index=False)
predict_df.to_csv('./Samples/test.csv', index=False)