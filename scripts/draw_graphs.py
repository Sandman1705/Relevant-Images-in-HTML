import sys
sys.path.insert(0, '../src')
import os
dir_name = os.path.dirname(sys.argv[0])
if(len(dir_name)!=0):
	os.chdir(os.path.dirname(sys.argv[0]))

from graphs import make_all_kernel_plots, make_kernel_plots, draw_imshow
import pandas as pd

def main():

	result_data = pd.read_csv("../results/results_train_1-svc_testing_kernels.csv", index_col=[0])
	make_all_kernel_plots(result_data, horizontal='gamma', vertical='C', cmap='ocean', \
	                      round_values=True, save_png=True, save_pdf=False)

	poly_kernel_results = pd.read_csv("../results/results_train_3-svc_poly_degree3_testing_gammas.csv", index_col=[0])
	poly_kernel_results['kernel'] = 'poly'
	rbf_kernel_results = pd.read_csv("../results/results_train_4-rbf-testing_C_and_gamma.csv", index_col=[0])
	rbf_kernel_results['kernel'] = 'rbf'
	all_kernel_results = pd.concat([poly_kernel_results, rbf_kernel_results], sort=False)
	make_kernel_plots(all_kernel_results, horizontal='gamma', vertical='C', kernels=['poly','rbf'], \
					  round_values=True, cmap='ocean', save_png=True, save_pdf=False, \
					  filename_prefix="SVC_kernel_fine_")

	degree_results = pd.read_csv("../results/results_train_2-svc_poly_testing_degrees.csv", index_col=[0])
	draw_imshow(degree_results, "SVC with kernel='poly'", "degree", "C", "f1", round_values=True, \
			    save_png=True, save_pdf=False, filename="SVC_poly_degree")#, \
				# vmin = min_value, vmax = max_value)

#if __name__ == "__main__":
#	main()

main()