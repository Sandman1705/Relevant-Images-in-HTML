import sys
sys.path.insert(0, '../src')
import os
dir_name = os.path.dirname(sys.argv[0])
if(len(dir_name)!=0):
	os.chdir(os.path.dirname(sys.argv[0]))

import data_preprocess
import pandas as pd

def main():

	data = data_preprocess.process_data_set("../data/sample", "labels.json")
	data = data_preprocess.change_bool_to_int(data)
	data.to_csv("imgdata_sample.csv", index=False)
	print(data.groupby('is_relevant').count())

#if __name__ == "__main__":
#	main()

main()