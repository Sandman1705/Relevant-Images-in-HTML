import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def column_maker(result_data, name_of_col_attr, name_of_col_value):
	col_attr_values = np.unique(result_data[name_of_col_attr])
	columns_data = np.empty(len(col_attr_values), dtype=object)
	for i in range(len(col_attr_values)):
		columns_data[i] = list(result_data[result_data[name_of_col_attr]==col_attr_values[i]][name_of_col_value])
	return columns_data

def matrix_maker(result_data, name_for_horizontal_data, name_for_vertical_data, name_for_value_data):
	columns_data = column_maker(result_data, name_for_vertical_data, name_for_value_data)
	mat = pd.DataFrame(index=np.unique(result_data[name_for_horizontal_data]))
	col_attr_values = np.unique(result_data[name_for_vertical_data])
	for col_name, col_data in zip(col_attr_values, columns_data):
		mat[str(col_name)] = col_data
	return mat

def draw_imshow(result_data, title, horizontal_column, vertical_column, value_column, \
				round_values = False, save_pdf = False, save_png = False, filename = None, \
				vmin = None, vmax = None, cmap='ocean'):

	if filename is None:
		filename = title
	horizontal = np.unique(result_data[horizontal_column])
	vertical = np.unique(result_data[vertical_column])
	horizontal_tick_names = list(map(str,horizontal))
	vertical_tick_names = list(map(str,vertical))
	if round_values:
		for i, e in enumerate(horizontal):
			horizontal_tick_names[i] = round(e,6)
		for i, e in enumerate(vertical):
			vertical_tick_names[i] = round(e,6)
	mat = matrix_maker(result_data, horizontal_column, vertical_column, value_column)

	plt.figure(figsize = ((len(horizontal)+2)/1.5, (len(vertical)+3)/1.5))
	plt.title(title)
	plt.xlabel(horizontal_column)
	plt.ylabel(vertical_column, rotation='horizontal')
	plt.xticks(range(len(horizontal)), horizontal_tick_names)
	plt.yticks(range(len(vertical)), vertical_tick_names, rotation='horizontal')
	plt.imshow(mat.T, cmap=cmap, vmin=vmin, vmax=vmax)
	plt.colorbar(orientation='horizontal').set_label(value_column)

	if save_png:
		plt.savefig(filename+".png")
		print("Graph saved to:", filename+".png")
	if save_pdf:
		plt.savefig(filename+".pdf", bbox_inches='tight')
		print("Graph saved to:", filename+".pdf")

	#plt.show()
	#return plt.gcf()

def make_all_kernel_plots(result_data, horizontal='gamma', vertical='C', value='f1', round_values=False, save_png=False, save_pdf=False, \
       titles=["SVC with kernel='poly'", "SVC with kernel='rbf'", "SVC with kernel='linear'", "SVC with kernel='sigmoid'"], \
       filenames=["SVC_kernel_poly","SVC_kernel_rbf","SVC_kernel_linear","SVC_kernel_sigmoid"], \
       vmin='auto', vmax='auto', cmap='ocean'):
    
    #result_data = pd.read_csv("results_train_1-testing_kernels.csv", index_col=[0])
    res_poly = result_data[result_data['kernel']=='poly']
    res_rbf = result_data[result_data['kernel']=='rbf']
    res_linear = result_data[result_data['kernel']=='linear']
    res_sigmoid = result_data[result_data['kernel']=='sigmoid']
    
    if vmin=='auto':
        min_value = np.min(result_data[value])
    else:
        min_value = 0.0
    if vmax=='auto':
        max_value = np.max(result_data[value])
    else:
        max_value = 1.0

    img1 = draw_imshow(res_poly, titles[0], horizontal, vertical, value, \
            round_values=round_values, save_png=save_png, save_pdf=save_pdf, filename=filenames[0], \
            vmin = min_value, vmax = max_value, cmap=cmap)
    img2 = draw_imshow(res_rbf, titles[1], horizontal, vertical, value, \
            round_values=round_values, save_png=save_png, save_pdf=save_pdf, filename=filenames[1], \
            vmin = min_value, vmax = max_value, cmap=cmap)
    img3 = draw_imshow(res_linear, titles[2], horizontal, vertical, value, \
            round_values=round_values, save_png=save_png, save_pdf=save_pdf, filename=filenames[2], \
            vmin = min_value, vmax = max_value, cmap=cmap)
    img4 = draw_imshow(res_sigmoid, titles[3], horizontal, vertical, value, \
            round_values=round_values, save_png=save_png, save_pdf=save_pdf, filename=filenames[3], \
            vmin = min_value, vmax = max_value, cmap=cmap)


def make_kernel_plots(result_data, horizontal='gamma', vertical='C', value='f1', round_values=False, save_png=False, \
					  save_pdf=False, kernels=['poly','rbf','linear','sigmoid'],title_prefix="SVC with kernel=", \
					  filename_prefix="SVC_kernel_",vmin='auto', vmax='auto', cmap='ocean'):

	no_of_kernels = len(kernels)
	res = np.empty(no_of_kernels, dtype=object)
	for i in range(no_of_kernels):
		res[i] = result_data[result_data['kernel']==kernels[i]]

	if vmin=='auto':
		min_value = np.min(result_data[value])
	else:
		min_value = 0.0
	if vmax=='auto':
		max_value = np.max(result_data[value])
	else:
		max_value = 1.0

	for i in range(no_of_kernels):
		draw_imshow(res[i], title_prefix+kernels[i], horizontal, vertical, value, round_values=round_values, \
					save_png=save_png, save_pdf=save_pdf, filename=filename_prefix+kernels[i], \
					vmin= min_value, vmax=max_value, cmap=cmap)
