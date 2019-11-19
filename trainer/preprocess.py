import os
import subprocess
import sys

# from google.cloud import storage

import pandas as pd
import numpy as np
import math

from scipy.stats import iqr, zscore
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import sklearn.decomposition

from sklearn.feature_extraction import FeatureHasher



def split_train_predict(df):
	'''
		Keeps as training set the points where Score has a value. 
		The rest are used as the test set. 
	'''
	return df[df['Score'].notnull()], df[df['Score'].isnull()]


def clean_data(df):
	'''
		Drops missing and inconsistent values. 
	'''
	df = df.dropna()
	df = df[(df[['HelpfulnessNumerator', 'HelpfulnessDenominator', 'Score', 'Time']] >= 0).all(1)]

	for i, (numerator, denominator) in enumerate(zip(df['HelpfulnessNumerator'], df['HelpfulnessDenominator'])):
		if numerator > denominator:
			df = df.drop(df.index[i])

	return df


def add_features(df, profile_users=True):
	'''
		Adds extra features to the dataset. 
	'''
	# add the percentage of helpfulness
	df['Helpfulness'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator']
	df['Helpfulness'] = df['Helpfulness'].fillna(0)

	if profile_users:
		# bin the timestamp and add each data point to each corresponding bin
		bin_width, n_bins = bin(df, 'Time')

		bin_ids = []
		time_min = df['Time'].min()
		
		print('Binning Time\n')
		for time_element in df['Time']:
			bin_ids.append(assign_to_bin(time_element, time_min, bin_width, n_bins))

		df['TimeBin'] = bin_ids

		print('Adding Statistics\n')
		for feature in ['UserId', 'ProductId']:
			df = add_statistics(df, feature)

		df = df.drop(columns='TimeBin')

	return df


def normalize(df):
	return zscore(df)

def bin(df, feature):
	'''
		Bins the timestamp based on the Freedman–Diaconis rule. 
	'''
	feature_length = len(df[feature])
	feature_iqr = iqr(df[feature])

	bin_width = 2 * feature_iqr / feature_length**(1/3)
	n_bins = math.ceil((df[feature].max() - df[feature].min()) / bin_width)

	return bin_width, n_bins


def assign_to_bin(element, starting_point, bin_width, n_bins):
	'''
		Assigns each time point to its corresponding bin. 
	'''
	for i in range(1, n_bins, 1):
		if (element - starting_point) / bin_width < i:
			return i

	return n_bins


def statistics(df):
	pos_votes = df['HelpfulnessNumerator'].sum()
	total_votes = df['HelpfulnessDenominator'].sum()

	if total_votes > 0:
		helpfulness = pos_votes / total_votes
	else:
		helpfulness = 0

	avg_rating = df['Score'].mean(skipna=True)
	std_rating = df['Score'].std(skipna=True)
	mode_rating = df['Score'].mode(dropna=True)
	skew_rating = df['Score'].skew(skipna=True)

	median_time = df['Time'].median(skipna=True)

	return pos_votes, total_votes, helpfulness, avg_rating, std_rating, mode_rating, skew_rating, median_time


def add_statistics(df, feature):
	feature_elements = df[feature].unique()
	
	for element in feature_elements:
		feature_data = df.loc[df[feature] == element]

		reviews = len(feature_data)
		
		pos_votes, total_votes, helpfulness, avg_rating, std_rating, mode_rating, skew_rating, median_time = statistics(feature_data)

		df.loc[(df[feature] == element), '{}PosVotes'.format(feature)] = pos_votes
		df.loc[(df[feature] == element), '{}TotalVotes'.format(feature)] = total_votes
		df.loc[(df[feature] == element), '{}Helpfulness'.format(feature)] = helpfulness
		df.loc[(df[feature] == element), '{}AvgRating'.format(feature)] = avg_rating
		df.loc[(df[feature] == element), '{}StdRating'.format(feature)] = std_rating
		df.loc[(df[feature] == element), '{}ModeRating'.format(feature)] = mode_rating
		df.loc[(df[feature] == element), '{}SkewRating'.format(feature)] = skew_rating
		df.loc[(df[feature] == element), '{}MedianTime'.format(feature)] = median_time

		feature_bins = feature_data['TimeBin'].unique()

		# for feature_bin in feature_bins:
		# 	feature_bin_data = feature_data.loc[df['TimeBin'] == feature_bin]

		# 	bin_pos_votes, bin_total_votes, bin_helpfulness, bin_avg_rating, bin_std_rating, bin_mode_rating, bin_skew_rating, bin_median_time = statistics(feature_bin_data)

		# 	df.loc[(df[feature] == element) & (df['TimeBin'] == feature_bin), '{}BinPosVotes'.format(feature)] = bin_pos_votes
		# 	df.loc[(df[feature] == element) & (df['TimeBin'] == feature_bin), '{}BinTotalVotes'.format(feature)] = bin_total_votes
		# 	df.loc[(df[feature] == element) & (df['TimeBin'] == feature_bin), '{}BinHelpfulness'.format(feature)] = bin_helpfulness
		# 	df.loc[(df[feature] == element) & (df['TimeBin'] == feature_bin), '{}BinAvgRating'.format(feature)] = bin_avg_rating
		# 	df.loc[(df[feature] == element) & (df['TimeBin'] == feature_bin), '{}BinStdRating'.format(feature)] = bin_std_rating
		# 	df.loc[(df[feature] == element) & (df['TimeBin'] == feature_bin), '{}BinModeRating'.format(feature)] = bin_mode_rating
		# 	df.loc[(df[feature] == element) & (df['TimeBin'] == feature_bin), '{}BinSkewRating'.format(feature)] = bin_skew_rating
		# 	df.loc[(df[feature] == element) & (df['TimeBin'] == feature_bin), '{}BinMedianTime'.format(feature)] = bin_median_time

	return df


def tf_idf(df, cool_coefs=False):
	text = df['Summary'] + '. ' + df['Text']
	df = df.drop(['Summary', 'Text'], axis=1)

	vectorizer = TfidfVectorizer(strip_accents='unicode', lowercase=True, stop_words='english')
	vectors = vectorizer.fit_transform(text.values.astype('U')).toarray()

	if cool_coefs:		
		n_docs = len(vectors)
		word_presence = np.where(vectors > 0, 1, 0)
		presence_counts = np.sum(word_presence, axis=0)

		word_score = np.array(df['Score']).reshape(-1, 1) * word_presence

		c1 = np.absolute(np.sin((n_docs / presence_counts) * math.pi)) + 1
		c1 = np.where(np.isnan(c1), 1, c1)
		
		c2 = np.exp(np.nanstd(word_score, axis=0))

		vectors = c1 * c2 * vectors

	return df.join(pd.DataFrame(vectors))


def get_cdf(variance):
	for i in range(len(variance)):
		variance[i] += variance[i-1]

	return variance


def apply_pca(df):
	pca = sklearn.decomposition.PCA()
	pca.fit(df)

	variance_retained = get_cdf(pca.explained_variance_ratio_)
	n_components = max(np.where(variance_retained <= 0.99)[0])

	pca = sklearn.decomposition.PCA(n_components=n_components)
	df_proj = pca.fit_transform(df)

	return df_proj



FILENAME = 'train'
FOLDER = 'Sample'

print('Fetching Data\n')

# # use for fetching the data from the gcloud bucket
# data = os.path.join(os.getcwd(), '{}.csv'.format(FILENAME))
# subprocess.check_call(['gsutil', 'cp', 'gs://cs-506-258209-mlengine/Dataset/{}.csv'.format(FILENAME), data], stderr=sys.stdout)
# df = pd.read_csv(data)

# use to run with local files
train_df = pd.read_csv('./Samples/train.csv')
train_df = clean_data(train_df)

predict_df = pd.read_csv('./Samples/test.csv')

df = pd.concat([train_df, predict_df])

y = np.array(df['Score'])
df = df.drop(columns='Id')


print('Adding Features\n')
df = add_features(df)
df = df.drop(columns='Score')

print('Calculating Tf-Idf\n')
df = tf_idf(df)


print('Creating One-Hot Features\n')
user_id_one_hot = pd.get_dummies(df['UserId'])
df = df.drop(columns='UserId')

product_id_one_hot = pd.get_dummies(df['ProductId'])
df = df.drop(columns='ProductId')

df = pd.concat([user_id_one_hot, product_id_one_hot, df], axis=1)

print(df)

df = df.dropna(axis=1)


print('Normalizing\n')
df = normalize(df)


print('Starting PCA\n')
X = apply_pca(np.nan_to_num(np.array(df)))


print('Getting Final Datasets\n')
X_train = X[np.where(np.isfinite(y))]

y_train = y[np.where(np.isfinite(y))]
y_train_one_hot = pd.get_dummies(y_train)

X_predict = X[np.where(np.isnan(y))]


print('Saving Datasets\n')
np.savetxt('X_train.csv', X_train, delimiter=',')
np.savetxt('y_train.csv', y_train, delimiter=',')
np.savetxt('y_train_one_hot.csv', y_train_one_hot, delimiter=',')
np.savetxt('X_predict.csv', X_predict, delimiter=',')

# # use to store on gcloud bucket
# client = storage.Client()
# bucket = client.get_bucket('cs-506-258209-mlengine')

# blob = bucket.blob('X_train.csv')
# blob.upload_from_filename('./X_train.csv')

# blob = bucket.blob('y_train.csv')
# blob.upload_from_filename('./y_train.csv')

# blob = bucket.blob('y_train_one_hot.csv')
# blob.upload_from_filename('./y_train_one_hot.csv')

# blob = bucket.blob('X_predict.csv')
# blob.upload_from_filename('./X_predict.csv')





