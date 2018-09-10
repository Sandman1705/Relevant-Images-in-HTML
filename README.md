# Relevant-Images-in-HTML

This project represents a posible solution for [Diffbot's Machine Learning Challenge](https://www.diffbot.com/robotlab/DiffbotContest/)

Problem: Determining Relevant Images From HTML Source

Since this project was done years after the Diffbot chalenge ended the author did not focus on representing the output of data in the way that could be submitted (which cannot be done anymore). Instead the focus is on finding the best possible model as well as generating data that can be interpreted for later use.

## Usage:

This project was written entirely in Python. The following python libraries were used:
	(For Python 3)
	- bs4 [Beautiful Soup 4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
	- lxml [lxml - XML and HTML with Python](https://lxml.de/)
	- sklearn
	- scipy
As well as some other more common ones:
	- json
        - pandas
	- numpy
	- urllib
	- matplotlib
	- re
	- time
	- pickle

## Project structure:

+ **src** directory containes python modules for extracting and preprocessing data as well as functions for training and evaluating models
+ **scripts** directory containes multiple short scripts for running various tasks
	- Note: before running any *extract data* make sure to unpack data from data/all_data.zip into proper directories under /data 
	- See [scripts/README.md](../blob/master/scripts/README.md) for more info
+ **data** directory containes all data sets given by Diffbot (see [data/README.md](../blob/master/data/README.md) for more info)
+ **results** containes CSV files produced by scripts that test models on train data (provided as backup for reference in order to avoid long runtimes on train data set)
+ **doc** Contains documentation PDF (written in serbian since this was a student project for a faculty course)


## Other info:

This was a student project made for course <b>Machine Learning</b> in Faculty of Mathematics, University of Belgrade of academic year 2017/2018.

Ovo je studentski projekat urađen za kurs <b>Mašinsko učenje</b> na Matematičkom fakultetu u Beogradu akademske 2017/2018.

