import sys
sys.path.insert(0, '../src')
import os
dir_name = os.path.dirname(sys.argv[0])
if(len(dir_name)!=0):
	os.chdir(os.path.dirname(sys.argv[0]))

from svc_models import cross_validation
import pandas as pd
import numpy as np
import time

def main():

	data = pd.read_csv("../data/imgdata_train.csv", index_col='page')
	params = {"C": 400, "gamma": 0.03, "kernel": 'rbf', "class_weight": 'balanced'}
	np_of_rand_states = 50
	
	full_start_time = time.time()
	results = np.empty(np_of_rand_states, dtype=object)
	for i in range(np_of_rand_states):
		print("RANDOM STATE: ", i+1)
		start_time = time.time()
		results_f1, results_acc, results_prec, results_rec = cross_validation(data, params=params, random_state=i+1)
		print("f1: mean:", np.mean(results_f1), "min:", np.min(results_f1), "max:", np.max(results_f1))
		print("acc: mean:", np.mean(results_acc), "min:", np.min(results_acc), "max:", np.max(results_acc))
		print("prec: mean:", np.mean(results_prec), "min:", np.min(results_prec), "max:", np.max(results_prec))
		print("rec: mean:", np.mean(results_rec), "min:", np.min(results_rec), "max:", np.max(results_rec))
		results[i] = (results_f1, results_acc, results_prec, results_rec)
		print("Time:", time.time()-start_time)
	print("FULL TIME:", time.time()-full_start_time)
	
	f1_res = [f1 for f1,acc,prec,rec in results]
	acc_res = [acc for f1,acc,prec,rec in results]
	prec_res = [prec for f1,acc,prec,rec in results]
	rec_res = [rec for f1,acc,prec,rec in results]
	
	print("Results:")
	print("Mean:")
	print("F1:       ", np.mean(f1_res)  )
	print("Accuracy: ", np.mean(acc_res) )
	print("Precision:", np.mean(prec_res))
	print("Recall:   ", np.mean(rec_res) )
	print("Min:")
	print("F1:       ", np.min(f1_res)   )
	print("Accuracy: ", np.min(acc_res)  )
	print("Precision:", np.min(prec_res) )
	print("Recall:   ", np.min(rec_res)  )
	print("Max:")
	print("F1:       ", np.max(f1_res)   )
	print("Accuracy: ", np.max(acc_res)  )
	print("Precision:", np.max(prec_res) )
	print("Recall:   ", np.max(rec_res)  )
		  
	df = pd.DataFrame(columns=["rand","f1","acc","prec","rec"])
	df["rand"] = np.arange(np_of_rand_states)
	df["f1"] = [np.mean(e) for e in f1_res]
	df["acc"] = [np.mean(e) for e in acc_res]
	df["prec"] = [np.mean(e) for e in prec_res]
	df["rec"] = [np.mean(e) for e in rec_res]
	df.to_csv("results_train_6-svc_best_rbf_rand_state_4-fold_cross_validation.csv")

#if __name__ == "__main__":
#	main()

main()