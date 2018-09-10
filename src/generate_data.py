import pandas as pd
import numpy as np
from scipy import stats
from urllib.parse import urlparse

def generate_new_attributes(img_data, page_info):
	'''Generate new attributes based on attributes about all <img> elements
	extracted from a single HTML document and additional info about the document
	itself

	Args:
		img_data: pandas.DataFrame object containg attribute values
		page_info: padnas dict containg info about HTML document

	Returns:
		pandas.DataFrame object containg only new attributes
	'''

	new_data = pd.DataFrame()

	new_data["is_external"] = check_srcs_is_external(img_data["src"])

	new_data["in_bounds"] = [ is_in_bounds(page_info["render size"], img_data["x"][i], img_data["y"][i], img_data["width"][i], img_data["height"][i]) for i in range(len(img_data))]

	new_data["distance_from_edge_left"], new_data["distance_from_edge_right"], new_data["distance_from_edge_up"], new_data["distance_from_edge_down"] = calulate_distances_from_edges(page_info["render size"], img_data["x"], img_data["y"], img_data["width"], img_data["height"])

	new_data["same_sized_imgs"] = count_elements_of_same_size(img_data["width"],img_data["height"])

	new_data["standardized_edit_distance_title_to_src"] = edit_standardized(img_data["edit_distance_title_to_src"])
	new_data["standardized_edit_distance_title_to_alt"] = edit_standardized(img_data["edit_distance_title_to_alt"])
	new_data["standardized_edit_distance_title_to_title"] = edit_standardized(img_data["edit_distance_title_to_title"])

	new_data["center_x"] = center_coordinates(img_data["x"],img_data["width"])
	new_data["center_y"] = center_coordinates(img_data["y"],img_data["height"])

	render_width = page_info["render size"][2]
	render_height = page_info["render size"][3]
	if (render_width==0 or render_height==0):
		new_data["rel_x"], new_data["rel_y"], new_data["rel_width"], new_data["rel_height"], new_data["rel_distance_left"], new_data["rel_distance_right"], new_data["rel_distance_up"], new_data["rel_distance_down"], new_data["rel_ancestors_width"], new_data["rel_ancestors_height"] = [None] * 10
	else:
		new_data["rel_x"] = relative_coordinate(img_data["x"],render_width)
		new_data["rel_y"] = relative_coordinate(img_data["y"],render_height)
		new_data["rel_width"] = relative_coordinate(img_data["width"],render_width)
		new_data["rel_height"] = relative_coordinate(img_data["height"],render_height)
		new_data["rel_distance_left"] = relative_coordinate(new_data["distance_from_edge_left"],render_width)
		new_data["rel_distance_right"] = relative_coordinate(new_data["distance_from_edge_right"],render_width)
		new_data["rel_distance_up"] = relative_coordinate(new_data["distance_from_edge_up"],render_height)
		new_data["rel_distance_down"] = relative_coordinate(new_data["distance_from_edge_down"],render_height)
		new_data["rel_ancestors_width"] = relative_coordinate(img_data["smallest_rendered_ancestors_width"],render_width)
		new_data["rel_ancestors_height"] = relative_coordinate(img_data["smallest_rendered_ancestors_height"],render_height)

	return new_data


def is_absolute(url):
	return (url is None) or bool(urlparse(url).netloc) or url.lstrip().startswith("http")

def check_srcs_is_external(src_data):
	is_external = np.zeros(len(src_data),dtype=bool)
	for index, link in enumerate(src_data):
		is_external[index]= is_absolute(link)
	return is_external

def is_in_bounds(document_size, img_x, img_y, img_w, img_h):
	if (img_x is None or img_y is None or img_w is None or img_h is None):
		return False
	return not((img_x < document_size[0]) or \
			   (img_x+img_w>document_size[0]+document_size[2]) or \
			   (img_y<document_size[1]) or \
			   (img_y+img_h>document_size[1]+document_size[3]))

def distances_from_edges( doc_x, doc_y, doc_w, doc_h, img_x, img_y, img_w, img_h):
	#left, right, up, down
	if (img_x is None or img_y is None or img_w is None or img_h is None):
		return (None, None, None, None)
	return (img_x-doc_x, (doc_x+doc_w)-(img_x+img_w), img_y-doc_y, (doc_y+doc_h)-(img_y+img_h))

def calulate_distances_from_edges( document_size, data_x, data_y, data_w, data_h):
	number_of_imgs = len(data_x)
	distances_left = np.empty(number_of_imgs,dtype=object)
	distances_right = np.empty(number_of_imgs,dtype=object)
	distances_up = np.empty(number_of_imgs,dtype=object)
	distances_down = np.empty(number_of_imgs,dtype=object)
	x, y, w, h = document_size
	for i in range(number_of_imgs):
		left, right, up, down = distances_from_edges(x, y, w, h, data_x[i], data_y[i], data_w[i], data_h[i])
		distances_left[i] = left
		distances_right[i] = right
		distances_up[i] = up
		distances_down[i] = down
	return (distances_left, distances_right, distances_up, distances_down)

def count_elements_of_same_size(w_data,h_data):
	data = list(zip(w_data,h_data))
	return ( [data.count(x) for x in data]  )

def edit_standardized(edit_distance_data):
	if np.std(edit_distance_data) == 0:
		return np.zeros(len(edit_distance_data), dtype=float)
	return stats.zscore(edit_distance_data)

def edit_relative_size_to_largest(edit_distance_data):
	max_value = edit_distance_data.max()
	relative_values = np.empty(len(edit_distance_data),dtype=float)
	for index, value in enumerate(edit_distance_data):
		relative_values[index] = value/max_value
	return relative_values

def center_coordinates(corner_data, size_data):
	length = len(corner_data)
	center_data = np.empty(length,dtype=object)
	for i in range(length):
		if corner_data[i] is None or size_data[i] is None:
			center_data[i] = None
		else:
			center_data[i] = corner_data[i]+(size_data[i]//2)
	return center_data

def relative_coordinate(cor_data, max_value):
	return cor_data/max_value

