### Extrinsic Evaluation: Method/Fragment-Level Vulnerability Detection

We leverage VulCNN as the automated tool for vulnerability detection.

Link to VulCNN code: [link](https://github.com/CGCL-codes/VulCNN)

Steps:
* First, convert DeepPDA model predictions to dot files.
* To do so, change file paths in `func_to_dot.py` and run with `$ python func_to_dot.py`
