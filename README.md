SIGNet requires python >=2.7 with cython, networkx and pickle package. You will also need C++ GSL library to build the code. In the setup.py you have to give the location of your GSL directory in include_dirs and extra_link_args.
Build the SIGNet using the following command: python setup.py build_ext --inplace
Then you can run it on the dataset by changing the value of dataset_name.
