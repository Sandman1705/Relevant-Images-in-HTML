import sys
sys.path.insert(0, '../src')
import os
dir_name = os.path.dirname(sys.argv[0])
if(len(dir_name)!=0):
	os.chdir(os.path.dirname(sys.argv[0]))

import data_preprocess
import pandas as pd

def main():

	data = data_preprocess.process_data_set("../data/test_public", "labels.json", has_xpath=False)
	data = data_preprocess.change_bool_to_int(data, has_relevant=False)
	data.to_csv("imgdata_test_public.csv", index=False)

#if __name__ == "__main__":
#	main()

main()