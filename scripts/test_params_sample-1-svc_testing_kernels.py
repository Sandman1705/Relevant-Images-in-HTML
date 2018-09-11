import sys
sys.path.insert(0, '../src')
import os
dir_name = os.path.dirname(sys.argv[0])
if(len(dir_name)!=0):
	os.chdir(os.path.dirname(sys.argv[0]))

from svc_models import kernel_test
import pandas as pd

def main():

	data = pd.read_csv("../data/imgdata_sample.csv", index_col='page')
	results, best_params, (f1_score, acc_score) = kernel_test(data)
	#print("Best parameters:", best_params)
	#print("F1:", f1_score, "Accuracy:", acc_score)
	results.to_csv("results_sample_1-svc_testing_kernels.csv")

#if __name__ == "__main__":
#	main()

main()