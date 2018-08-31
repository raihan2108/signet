SIGNet requires python >=2.7 with cython, networkx and pickle package.
You will also need C++ GSL library to build the code.

Building the code: `python setup.py build_ext --inplace`

Running code: `python signet.py`

Make sure to add GSL library path in the `setup.py`. Headers files usually are in `/usr/local/include/` and library files are in `/usr/local/lib/` 

If you use the code please cite:
```
@inproceedings{islam2018signet,
  title={Signet: Scalable embeddings for signed networks},
  author={Islam, Mohammad Raihanul and Prakash, B Aditya and Ramakrishnan, Naren},
  booktitle={Pacific-Asia Conference on Knowledge Discovery and Data Mining},
  pages={157--169},
  year={2018},
  organization={Springer}
}
```
