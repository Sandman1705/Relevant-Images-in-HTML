## Python scripts that can be used for running various tasks:

### Extracting data
The following scripts extract data from appropriate directory under /data/
Note: before running any of these scripts make sure to unpack data from data/all_data.zip into proper directories under /data 
+ extract_sample_data.py
+ extract_train_data.py
+ extract_test_public_data.py
The output is a single CSV file that combines data from all HTML documents in a data set. Runtime for *train* and *test_public* data sets can be somewhat longer so already preprocess data is provided in /data/ directory.

### Testing SVC models
The following scripts are used for testing various parameters of SVC model:
+ test_params_sample-1-svc_testing_kernels
+ test_params_sample-2-svc_poly_testing_degrees
+ test_params_sample-3-svc_poly_degree3_testing_gammas
+ test_params_sample-4-svc_rbf_testing_C_and_gammas
+ test_params_sample-5-svc_poly_cross_validation
+ test_params_sample-6-svc_rbf_cross_validation
+ test_params_train-1-svc_testing_kernels
+ test_params_train-2-svc_poly_testing_degrees
+ test_params_train-3-svc_poly_degree3_testing_gammas
+ test_params_train-4-svc_rbf_testing_C_and_gammas
+ test_params_train-5-svc_poly_cross_validation
+ test_params_train-6-svc_rbf_cross_validation
They are set to use already preprocessed data in /data/ directory. Same as before, because of the long runtime the resulting CSV files are already provided for *train* data set in /results/ directory. 

### Drawing graphs
This scripts is used for generating graphs used in documentation:
+ draw_graphs
It also relies on already generated data found in /results/ directory 

