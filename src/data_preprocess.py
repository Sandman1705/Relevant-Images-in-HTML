from html_parser import extract_img_info
from generate_data import generate_new_attributes
import json
import pandas as pd

import time

def read_labels(labels_file):
	'''Extracts data from .json file'''
	with open(labels_file) as f:
		data = json.load(f)
	return data

def process_data_set_and_save_to_csv(labels_path, labels_file, csv_filename, has_xpath=True):
	'''Runs process_data_set function and saves resulting pandas.DataFrame to csv'''
	data = process_data_set(labels_path, labels_file, has_xpath)
	data.to_csv(csv_filename)
	
def process_data_set(labels_path, labels_file, has_xpath=True):
	'''Extract all data for all documents in given .json file.
	
	Args:
		labels_path: path to labels.json file that contains list of all 
			available html documents in that same directory
		labels_file: name of .json file (usually labels.json)
		has_xpath: should be True if .json file containes "xpath" value for 
			each document listed and extracts 'is_relevant' attribute

	Returns:
		pandas.DataFrame object containing combined data of every document in 
			labels file
	'''

	labels = read_labels(labels_path+"/"+labels_file)
	num_of_pages = len(labels)

	start_time_extract = time.time()
	#frames = [ process_file(labels_path, label['id'], label['xpath']) for label in labels]
	frames = []
	for index, label in enumerate(labels):
		start_time = time.time()
		processed_data = pd.DataFrame()
		if has_xpath:
			processed_data = process_file(labels_path, label['id'], label['xpath'])# for label in labels
		else:
			processed_data = process_file(labels_path, label['id'])
		run_time = time.time() - start_time
		print("File", index+1, "/", num_of_pages, label['id'], "in time:", run_time, "no. of imgs", processed_data.shape[0])
		if has_xpath:
			print(processed_data['is_relevant'].value_counts())
		frames.append(processed_data)
		#print(processed_data.groupby('is_relevant').count())
		
	print("--- %s seconds of extract time ---" % (time.time() - start_time_extract))	
	start_time = time.time()
	all_data = pd.concat(frames)
	print("--- %s seconds of concat time ---" % (time.time() - start_time))
	
	#all_data.set_index("page", inplace=True, drop=True)
	print(all_data.shape)
	return all_data
		
def process_file(labels_path, labels_id, labels_xpath=None):
	'''Extract all attributes form a single HTML file'''
	
	#print("Processing file:", labels_path+"/"+labels_id)
	img_data, page_info = extract_img_info(labels_path, labels_id, labels_xpath)
	#print(page_info["title"])
	generated_data = generate_new_attributes(img_data, page_info)
	return img_data.join(generated_data)
	
def clean_drop_columns(data, list_of_column_names):
	'''Wrapper for droping certain attributes from data set'''
	return data.drop(list_of_column_names, axis=1)
	
#def clean_drop_columns_tolerant(data, list_of_column_names):
#	columns = [c for c in list_of_column_names if c in data.index]
#	return data.drop(columns, axis=1)
	
def clean_non_relevant_data(data):
	'''Removes all non numerical attribures'''
	non_relevant_attributes = ['src','alt','title']
	return clean_drop_columns(data, non_relevant_attributes)
	
def clean_absolute_data(data):
	'''Removes all attributes that are essentially duplicated during process of
	generating new and more appropriate attributes'''
	absolute_attributes = ['x', 'y', 'width', 'height', 'smallest_rendered_ancestors_width', 'smallest_rendered_ancestors_height', 'edit_distance_title_to_src', 'edit_distance_title_to_alt', 'edit_distance_title_to_title', 'distance_from_edge_left', 'distance_from_edge_right', 'distance_from_edge_up', 'distance_from_edge_down', 'center_x', 'center_y']
	return clean_drop_columns(data, absolute_attributes)
	
def change_bool_to_int(data, has_relevant=True):
	'''Changes all attribute represented as bool values into 0 or 1 value'''
	bool_attributes = ['is_hyperlink', 'classes_contain_header', 'classes_contain_content', 'classes_contain_nav', 'classes_contain_sidebar', 'classes_contain_footer', 'classes_contain_article', 'classes_contain_story', 'classes_contain_ad', 'classes_contain_advert', 'classes_contain_logo', 'classes_contain_main',	'is_external', 'in_bounds']
	if has_relevant:
		bool_attributes.append('is_relevant')
	for atr in bool_attributes:
		data[atr] = [ int(x) for x in data[atr]]
	return data
