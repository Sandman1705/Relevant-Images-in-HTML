import sys
sys.path.insert(0, '../src')
import os
os.chdir(os.path.dirname(sys.argv[0]))

from svc_models import poly_kernel_all_params_test
import pandas as pd

def main():

	data = pd.read_csv("../data/imgdata_train.csv", index_col='page')
	results, best_params, (f1_score, acc_score) = poly_kernel_all_params_test(data)
	#print("Best parameters:", best_params)
	#print("F1:", f1_score, "Accuracy:", acc_score)
	results.to_csv("results_train_3-svc_poly_degree3_testing_gammas.csv")

#if __name__ == "__main__":
#	main()

main()