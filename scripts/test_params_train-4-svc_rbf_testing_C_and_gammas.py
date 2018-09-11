import sys
sys.path.insert(0, '../src')
import os
dir_name = os.path.dirname(sys.argv[0])
if(len(dir_name)!=0):
	os.chdir(os.path.dirname(sys.argv[0]))

from svc_models import kernel_test
import pandas as pd
import numpy as np

def main():

	data = pd.read_csv("../data/imgdata_train.csv", index_col='page')
	Cs = np.concatenate(([90], np.linspace(100, 1000, 10), [1100]), axis=None)
	gammas = np.concatenate(([0.009], np.linspace(0.01, 0.1, 10), [0.11]), axis=None)
	results, best_params, (f1_score, acc_score) = kernel_test(\
            data, \
            kernels = np.array(['rbf']), \
            Cs = Cs, \
            gammas = gammas)
	#print("Best parameters:", best_params)
	#print("F1:", f1_score, "Accuracy:", acc_score)
	results.to_csv("results_train_4-svc_rbf_testing_C_and_gamma.csv")

#if __name__ == "__main__":
#	main()

main()