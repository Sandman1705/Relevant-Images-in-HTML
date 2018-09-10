import pandas as pd
import numpy as np
import time
import pickle
from data_preprocess import clean_non_relevant_data, clean_absolute_data
from sklearn import model_selection
from sklearn import metrics
from sklearn import preprocessing
from sklearn import svm
from matplotlib import pyplot as plt


def kernel_test(data, \
				kernels = np.array(['rbf', 'linear', 'poly', 'sigmoid']), \
				Cs = np.array([10**i for i in range(-3, 5)]), \
				gammas = np.array([10**i for i in range(-3, 3)]), \
				test_split_ratio = 0.25, \
				validation_split_ratio = 0.2, \
				random_state = 7 \
				):

	x, y = prepare_data(data)
	(x_train_validation, x_test, y_train_validation, y_test), \
		(x_train, x_validation, y_train, y_validation) = \
			split_and_scale_data(x, y, test_split_ratio, validation_split_ratio, random_state)

	best_score = 0
	best_params = {'C': 0, 'gamma': 0, 'kernel': 'none'}
	full_time = time.time()
	results = pd.DataFrame(index=np.arange(len(Cs)*len(gammas)*len(kernels)), columns=['kernel','C','gamma','f1','acc','time'])
	i=0
	for k in kernels:
		for C in Cs:
			for gamma in gammas:
				start_time = time.time()
				model = svm.SVC(C = C, gamma = gamma, kernel=str(k), class_weight='balanced')
				model.fit(x_train, y_train)
				y_predicted = model.predict(x_validation)
				score = metrics.f1_score(y_validation, y_predicted)
				score_acc = metrics.accuracy_score(y_validation, y_predicted)
				elapsed_time = time.time()-start_time
				print(str(i+1)+"/"+str(len(results)),k,"C:",C,"gamma:",gamma,"f1:",score,"acc:",score_acc,"time:",elapsed_time)
				if score>best_score:
					best_score = score
					best_params['C'] = C
					best_params['gamma'] = gamma
					best_params['kernel'] = str(k)
				results.loc[i]=[k,C,gamma,score,score_acc,elapsed_time]
				i=i+1
	print("Full train time:", time.time()-full_time)

	best_model = svm.SVC(**best_params, class_weight='balanced')
	start_time = time.time()
	best_model.fit(x_train_validation, y_train_validation)
	print("Best model train_validation time:", time.time()-start_time)
	y_predicted = model.predict(x_test)
	acc_score = metrics.accuracy_score(y_test, y_predicted)
	f1_score = metrics.f1_score(y_test, y_predicted)

	print("Best parameters:", best_params)
	print("acc:", acc_score, "f1:", f1_score)

	return results, best_params, (f1_score, acc_score)



def poly_kernel_degree_test(\
		data, \
		degrees = np.array([1,2,3,4,5]), \
		Cs = np.array([0.0001, 0.001, 0.01, 0.1]), \
		gammas = np.array([10.0, 10.0, 1.0, 1.0]), \
		test_split_ratio = 0.25, \
		validation_split_ratio = 0.2, \
		random_state = 7 \
		):

	params = pd.DataFrame()
	params["C"] = Cs
	params["gamma"] = gammas
	return poly_kernel_degree_test_with_params(data, degrees, params, test_split_ratio, validation_split_ratio, random_state)



def poly_kernel_degree_test_with_params(\
		data, \
		degrees = np.array([1,2,3,4,5]), \
		params = pd.DataFrame({'C': [0.0001, 0.001, 0.01, 0.1], 'gamma': [10.0, 10.0, 1.0, 1.0]}), \
		test_split_ratio = 0.25, \
		validation_split_ratio = 0.2, \
		random_state = 7 \
		):

	x, y = prepare_data(data)
	(x_train_validation, x_test, y_train_validation, y_test), \
		(x_train, x_validation, y_train, y_validation) = \
			split_and_scale_data(x, y, test_split_ratio, validation_split_ratio, random_state)

	full_time = time.time()
	results = pd.DataFrame(index=np.arange(len(degrees)*len(params)), columns=['degree','C','gamma','f1','acc','time'])
	best_score = 0
	best_params = {'C':0, 'gamma': 0, 'kernel': 'none', 'degree': 0}
	i=0
	for d in degrees:
		for index, row in params.iterrows():
			start_time = time.time()
			model = svm.SVC(C = row['C'], gamma = row['gamma'], kernel='poly', degree=d, class_weight='balanced')
			model.fit(x_train, y_train)
			y_predicted = model.predict(x_validation)
			score = metrics.f1_score(y_validation, y_predicted)
			score_acc = metrics.accuracy_score(y_validation, y_predicted)
			elapsed_time = time.time()-start_time
			print(str(i+1)+"/"+str(len(results)),"degree",d,"C",row['C'],"gamma",row['gamma'],"f1",score,"acc",score_acc,"time",elapsed_time)
			if score>best_score:
				best_score = score
				best_params['C'] = row['C']
				best_params['gamma'] = row['gamma']
				best_params['kernel'] = 'poly'
				best_params['degree'] = d
			results.loc[i]=[d,row['C'],row['gamma'],score,score_acc,elapsed_time]
			i=i+1
	print("Full time:", time.time()-full_time)

	best_params['class_weight'] = 'balanced'
	best_model = svm.SVC(**best_params)
	start_time = time.time()
	best_model.fit(x_train_validation, y_train_validation)
	print("Best model train_validation time:", time.time()-start_time)
	y_predicted = best_model.predict(x_test)
	acc_score = metrics.accuracy_score(y_test, y_predicted)
	f1_score = metrics.f1_score(y_test, y_predicted)

	print("Best parameters:", best_params)
	print("acc:", acc_score, "f1:", f1_score)

	return results, best_params, (f1_score, acc_score)


def poly_kernel_all_params_test(data, \
				Cs = np.array([0.0001, 0.001 , 0.01  , 0.1   ]), \
				gammas = np.array([ 0.1,  0.2,  0.3,  0.4,  0.5,  0.6,  0.7,  0.8,  0.9,  1. , \
								   2. , 3. ,  4. ,  5. ,  6. ,  7. ,  8. ,  9. , 10. , 11. ]), \
				degrees = [3], \
				test_split_ratio = 0.25, \
				validation_split_ratio = 0.2, \
				random_state = 7 \
				):

	x, y = prepare_data(data)
	(x_train_validation, x_test, y_train_validation, y_test), \
		(x_train, x_validation, y_train, y_validation) = \
			split_and_scale_data(x, y, test_split_ratio, validation_split_ratio, random_state)

	best_score = 0
	best_params = {'C': 0, 'gamma': 0, 'kernel': 'none', 'degree': 0}
	full_time = time.time()
	results = pd.DataFrame(index=np.arange(len(Cs)*len(gammas)*len(degrees)), columns=['degree','C','gamma','f1','acc','time'])
	i=0
	for d in degrees:
		for C in Cs:
			for gamma in gammas:
				start_time = time.time()
				model = svm.SVC(C = C, gamma = gamma, kernel='poly', degree=d, class_weight='balanced')
				model.fit(x_train, y_train)
				y_predicted = model.predict(x_validation)
				score = metrics.f1_score(y_validation, y_predicted)
				score_acc = metrics.accuracy_score(y_validation, y_predicted)
				elapsed_time = time.time()-start_time
				print(str(i+1)+"/"+str(len(results)),"degree:",d,"C:",C,"gamma:",gamma,"f1:",score,"acc:",score_acc,"time:",elapsed_time)
				if score>best_score:
					best_score = score
					best_params['C'] = C
					best_params['gamma'] = gamma
					best_params['kernel'] = 'poly'
					best_params['degree'] = d
				results.loc[i]=[d,C,gamma,score,score_acc,elapsed_time]
				i=i+1
	print("Full train time:", time.time()-full_time)

	best_params['class_weight'] = 'balanced'
	best_model = svm.SVC(**best_params)
	start_time = time.time()
	best_model.fit(x_train_validation, y_train_validation)
	print("Best model train_validation time:", time.time()-start_time)
	y_predicted = model.predict(x_test)
	acc_score = metrics.accuracy_score(y_test, y_predicted)
	f1_score = metrics.f1_score(y_test, y_predicted)

	print("Best parameters:", best_params)
	print("acc:", acc_score, "f1:", f1_score)

	return results, best_params, (f1_score, acc_score)



def random_state_test(\
		data, \
		random_states = np.arange(50), \
		params = {"C": 0.1, "gamma": 0.9, "kernel": 'poly', "degree": 3}, \
		test_size = 0.25\
		):

	full_time = time.time()
	results = pd.DataFrame(index=random_states, columns=['rand','f1','acc','time'])
	#best_score = 0
	#best_index = 0
	i=0

	x, y = prepare_data(data)
	for i in random_states:

		x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size = test_size, random_state = i, stratify = y )
		x_train, x_test = scale_data(x_train, x_test)

		start_time = time.time()
		params['class_weight'] = 'balanced'
		model = svm.SVC(**params)
		model.fit(x_train, y_train)
		y_predicted = model.predict(x_test)
		score = metrics.f1_score(y_test, y_predicted)
		score_acc = metrics.accuracy_score(y_test, y_predicted)
		elapsed_time = time.time()-start_time

		print(str(i+1)+"/"+str(len(random_states)),"f1",score,"acc",score_acc,"time",elapsed_time)
		#if score>best_score:
			#best_score = score
			#best_index = i
		results.loc[i]=[i,score,score_acc,elapsed_time]
	print("Full time: ", time.time()-full_time)
	print("Mean F1 score:", np.mean(results['f1']))
	print("Max F1 score:", np.max(results['f1']))
	print("Min F1 score:", np.min(results['f1']))
	return results



def cross_validation(data, params = {"C": 400, "gamma": 0.03, "kernel": 'rbf'}, \
					 number_of_folds=4, random_state=7):
	x, y = prepare_data(data)

	results_f1 = np.empty(number_of_folds)
	results_acc = np.empty(number_of_folds)
	results_prec = np.empty(number_of_folds)
	results_rec = np.empty(number_of_folds)
	i=0
	skf = model_selection.StratifiedKFold(n_splits=number_of_folds, random_state=random_state, shuffle=True)
	for train_index, test_index in skf.split(x, y):
		x_train, x_test = x.iloc[train_index], x.iloc[test_index]
		y_train, y_test = y.iloc[train_index], y.iloc[test_index]

		x_train, x_test = scale_data(x_train, x_test)

		start_time = time.time()
		params['class_weight'] = 'balanced'
		model = svm.SVC(**params)
		model.fit(x_train, y_train)
		y_predicted = model.predict(x_test)
		score = metrics.f1_score(y_test, y_predicted)
		score_acc = metrics.accuracy_score(y_test, y_predicted)
		score_prec = metrics.precision_score(y_test, y_predicted)
		score_rec = metrics.recall_score(y_test, y_predicted)
		elapsed_time = time.time()-start_time

		print("fold",i,"f1",score,"acc",score_acc,"time",elapsed_time)

		results_f1[i] = score
		results_acc[i] = score_acc
		results_prec[i] = score_prec
		results_rec[i] = score_rec
		i=i+1

	return results_f1, results_acc, results_prec, results_rec




def prepare_data(data):
	y = data['is_relevant']

	x = data.drop(columns='is_relevant')
	x = clean_non_relevant_data(x)
	x = clean_absolute_data(x)
	x = x.fillna(value=0.0)

	#x = x_full.drop(columns=['src', 'x', 'y', 'width', 'height', 'alt', 'title', 'smallest_rendered_ancestors_width',
	#   'smallest_rendered_ancestors_height', 'edit_distance_title_to_src', 'edit_distance_title_to_alt',
	#   'edit_distance_title_to_title', 'distance_from_edge_left', 'distance_from_edge_right', 'distance_from_edge_up',
	#   'distance_from_edge_down', 'center_x', 'center_y',])

	return x, y



def scale_data(x_train, x_test):
	scaler = preprocessing.StandardScaler()
	scaler.fit(x_train)
	x_train = scaler.transform(x_train)
	x_test = scaler.transform(x_test)
	return (x_train, x_test)


def split_and_scale_data(x, y, test_split_ratio, validation_split_ratio, random_state):
	x_train_validation, x_test, y_train_validation, y_test = \
		model_selection.train_test_split(x, y, test_size=test_split_ratio, stratify = y, random_state = random_state)
	x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x_train_validation, y_train_validation, test_size = \
		validation_split_ratio, stratify = y_train_validation, random_state = random_state)

	x_train, x_validation = scale_data(x_train, x_validation)
	x_train_validation, x_test = scale_data(x_train_validation, x_test)

	return (x_train_validation, x_test, y_train_validation, y_test), (x_train, x_validation, y_train, y_validation)



def train_svc_model_and_save(data, params, filename):
	x, y = prepare_data(data)

	scaler = preprocessing.StandardScaler()
	scaler.fit(x)
	x = scaler.transform(x)

	model = svm.SVC(**params)
	model.fit(x, y)

	relevant_finder = {}
	relevant_finder["trained_model"] = model
	relevant_finder["model_type"] = "sklearn.svm.SVC"
	relevant_finder["func"] = svm.SVC
	relevant_finder["parameters"] = params

	pickle.dump(relevant_finder, open(filename, "wb" ))
	print("saved model to:", filename)

	return relevant_finder


def load_trained_model(filename):
	return pickle.load(open(filename, "rb" ))
