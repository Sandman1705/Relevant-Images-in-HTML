# Relevant-Images-in-HTML

This project represents a possible solution for [Diffbot's Machine Learning Challenge](https://www.diffbot.com/robotlab/DiffbotContest/)

Problem: Determining Relevant Images From HTML Source

Since this project was done years after the Diffbot challenge ended the author did not focus on representing the output of data in the way that could be submitted (which cannot be done anymore). Instead the focus is on finding the best possible model as well as generating data that can be interpreted for later use.

## Requirements:

This project was written entirely in Python (Python 3). The following python libraries were used:
+ [bs4 - Beautiful Soup 4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
+ [lxml - XML and HTML with Python](https://lxml.de/)
+ [sklearn ](http://scikit-learn.org/stable/)
+ [scipy ](https://www.scipy.org/)

As well as some other more common ones:
+ json
+ pandas
+ numpy
+ urllib
+ matplotlib
+ re
+ time
+ pickle

## Project structure:

+ **src** directory contains python modules for extracting and preprocessing data as well as functions for training and evaluating models
+ **scripts** directory contains multiple short scripts for running various tasks
	- Note: before running any *extract data* script make sure to unpack data from data/all_data.zip into proper directories under /data/. See [scripts/README.md](../blob/master/scripts/README.md) for more info.
+ **data** directory contains all data sets given by Diffbot (see [data/README.md](../blob/master/data/README.md) for more info)
+ **results** directory contains CSV files produced by scripts that test models on train data (provided as backup for reference in order to avoid long runtime on train data set)
+ **doc** directory contains PDF documentation (written in serbian since this was a student project for a faculty course)


## Other info:

This was a student project made for course <b>Machine Learning</b> in Faculty of Mathematics, University of Belgrade of academic year 2017/2018.

Ovo je studentski projekat urađen za kurs <b>Mašinsko učenje</b> na Matematičkom fakultetu u Beogradu akademske 2017/2018.

