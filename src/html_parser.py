from bs4 import BeautifulSoup
from lxml import etree
import numpy as np
import pandas as pd
import re


def extract_img_info(path, html_file, xpath=None):
	'''Extracts all elements with <img> tag in given HTML file and determines
	values for all attributes. If a string that represents XPath is given it
	also provides an additional 'is_relevant' attribute.

	>>> extract_img_info("data/train", "example.html", xpath="/html/body/div/img")

	Returns: pandas.DataFrame object
	'''

	with open(path+"/"+html_file, 'rb') as fp:
		soup = BeautifulSoup(fp, 'html.parser')
	all_imgs = soup.find_all('img')

	data = pd.DataFrame()#index = [ html_file+"-"+str(i) for i in range(len(all_imgs))])
	page_info = {}
	page_info["render size"] = document_size(soup.body)
	page_info["title"] = soup.title.string

	#print(html_file)
	data["page"] = [ html_file+"-"+str(i) for i in range(len(all_imgs))]
	data["src"] = [img.get("src") for img in all_imgs]
	data["x"], data["y"], data["width"], data["height"] = get_coord(all_imgs)
	data["is_hyperlink"] = [is_hyperlink(img) for img in all_imgs]
	data["alt"] = [img.get("alt") for img in all_imgs]
	data["title"] = [img.get("title") for img in all_imgs]
	data["smallest_rendered_ancestors_width"], data["smallest_rendered_ancestors_height"] = smallest_rendered_ancestors(all_imgs)
	data["edit_distance_title_to_src"] = edit_distance_to_array_of_strings(soup.title.string, data["src"])
	data["edit_distance_title_to_alt"] = edit_distance_to_array_of_strings(soup.title.string, data["alt"])
	data["edit_distance_title_to_title"] = edit_distance_to_array_of_strings(soup.title.string, data["title"])

	words = ['header', 'content', 'nav', 'sidebar', 'footer', 'article', 'story', '(^|[^a-zA-Z])ad', 'advert', 'logo', 'main']
	contained_classes = pd.DataFrame([find_words_in_classes(img, words) for img in all_imgs])
	words[7] = "ad"
	contained_classes.columns = [ "classes_contain_"+word for word in words ]
	data = data.join(contained_classes)

	if xpath is not None:
		relevant_imgs = get_xpath_result(path+"/"+html_file, xpath)
		data["is_relevant"] = find_relevant_fast(all_imgs, relevant_imgs)

	return data, page_info


def is_hyperlink(tag):
	while(tag is not None and tag.name != 'html'):
		#print(tag.name)
		if (tag.name == 'a'):
			return True
		tag = tag.parent
	return False

def extract_coord_from_string(string):
	m1 = re.search('x=(\-?[0-9]+)',string)
	if m1 is None:
		return (None, None, None, None)
	m2 = re.search('y=(\-?[0-9]+)',string)
	m3 = re.search('w=(\-?[0-9]+)',string)
	m4 = re.search('h=(\-?[0-9]+)',string)
	return (int(m1.group(1)), int(m2.group(1)), int(m3.group(1)), int(m4.group(1)))

def get_coord(all_imgs):
	x_data = np.empty(len(all_imgs),dtype=object)
	y_data = np.empty(len(all_imgs),dtype=object)
	w_data = np.empty(len(all_imgs),dtype=object)
	h_data = np.empty(len(all_imgs),dtype=object)
	for index, img in enumerate(all_imgs):
		coord_string = img.get('_')
		if coord_string is not None:
			(x,y,w,h) = extract_coord_from_string(coord_string)
			x_data[index] = x
			y_data[index] = y
			w_data[index] = w
			h_data[index] = h
	return (x_data, y_data, w_data, h_data)

def smallest_rendered_ancestor(tag):
	tag = tag.parent
	while(tag.name != 'html'):
		render_data = tag.get('_')
		if render_data is not None:
			m1 = re.search('w=(\-?[0-9]+)',render_data)
			if m1 is None:
				return (None, None)
			m2 = re.search('h=(\-?[0-9]+)',render_data)
			return(int(m1.group(1)), int(m2.group(1)))
		tag = tag.parent
	return (None, None)

def smallest_rendered_ancestors(all_imgs):
	ancestor_width = np.empty(len(all_imgs),dtype=object)
	ancestor_height = np.empty(len(all_imgs),dtype=object)
	for index, img in enumerate(all_imgs):
		width, height = smallest_rendered_ancestor(img)
		ancestor_width[index] = width
		ancestor_height[index] = height
	return(ancestor_width, ancestor_height)

def levenshtein(s1, s2):
	if len(s1) < len(s2):
		return levenshtein(s2, s1)

	# len(s1) >= len(s2)
	if len(s2) == 0:
		return len(s1)

	previous_row = range(len(s2) + 1)
	for i, c1 in enumerate(s1):
		current_row = [i + 1]
		for j, c2 in enumerate(s2):
			insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
			deletions = current_row[j] + 1       # than s2
			substitutions = previous_row[j] + (c1 != c2)
			current_row.append(min(insertions, deletions, substitutions))
		previous_row = current_row

	return previous_row[-1]

def document_size(body_tag):
	render_data = body_tag.get('_')
	tag = body_tag
	tag_type = type(body_tag)
	while render_data is None:
		if (tag is None):
			return (0,0,0,0)
		tag = tag.next_element
		if (type(tag)==tag_type):
			render_data = tag.get('_')
	m1 = re.search('x=(\-?[0-9]+)',render_data)
	if m1 is None:
		return (0,0,0,0)
	m2 = re.search('y=(\-?[0-9]+)',render_data)
	m3 = re.search('w=(\-?[0-9]+)',render_data)
	m4 = re.search('h=(\-?[0-9]+)',render_data)
	return (int(m1.group(1)), int(m2.group(1)), int(m3.group(1)), int(m4.group(1)))

def edit_distance_to_array_of_strings(main_string, array):
	edit_distance = np.empty(len(array),dtype=int)
	for index, string in enumerate(array):
		if string is not None:
			edit_distance[index] = levenshtein(main_string, string)
		else:
			edit_distance[index] = levenshtein(main_string, "")
	return(edit_distance)

def find_words_in_classes(tag, words):
	found = np.zeros(len(words),dtype=bool)
	while(tag is not None and tag.name != 'body' and tag.name != 'html'):
		for index, word in enumerate(words):
			class_list = tag.get('class')
			if class_list is not None:
				res = [re.search(word, class_name, re.IGNORECASE) for class_name in tag.get('class')]
				if (res.count(None) != len(res)):
					found[index] = True
			if tag.get('id') is not None:
				if re.search(word, tag.get('id')) is not None:
					found[index] = True
			#print(tag.name, tag.get('class'), tag.get('id'))
		tag = tag.parent
	return found

def get_xpath_result(file, xpath):
	htmlparser = etree.HTMLParser(encoding="utf-8")
	with open(file, 'rb') as fb:
		tree = etree.parse(fb, htmlparser)
	result = tree.xpath(xpath)
	return ([img.get('src') for img in result], [img.get('_') for img in result])

def find_relevant(all_imgs, relevant_imgs):
	is_relevant = np.zeros(len(all_imgs), dtype=bool)
	for index, img in enumerate(all_imgs):
		for i in range(len(relevant_imgs[0])):
			if ((relevant_imgs[0][i]==img.get('src') or relevant_imgs[0][i] is None) and relevant_imgs[1][i]==img.get('_')):
				is_relevant[index] = True
				break
	return is_relevant

def find_relevant_fast(all_imgs, relevant_imgs):
	is_relevant = np.zeros(len(all_imgs), dtype=bool)
	j=0
	for i in range(len(relevant_imgs[0])):
		while (relevant_imgs[1][i]!=all_imgs[j].get('_')) or (relevant_imgs[0][i]!=all_imgs[j].get('src') and relevant_imgs[0][i] is not None):
			j=j+1
		is_relevant[j]=True
	return is_relevant


