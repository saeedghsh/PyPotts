# PyPotts
Task Assignment with Neural Network based on Potts Spin.

<p align="center">
	<img src="https://github.com/saeedghsh/PyPotts/master/docs/animation.gif" width="400">
</p>

The method is a *Neural Netork based on Potts Spin* for the task assignment problem.
The [original implementation](https://github.com/jenniferdavid/potts_spin) is by [Jennifer David](https://github.com/jenniferdavid) and in C++.
I implemented this is a python version under the supervision of Jennifer David.

# Dependencies and Download
Download, installing dependencies, and install package
```shell
# Download
$ git clone https://github.com/saeedghsh/PyPotts.git
$ cd PyPotts

# Install dependencies
$ pip install -r requirements.txt # python 2

# Install the package [optional]
python setup.py install # python 2
```

# Basic Use and API
* Solve assignment (From `D`, `T` to `V`):
```shell
$ not available yet...
```

* Parsing the V matrix (the assignments):
```shell
$ cd scripts/
$ python parse_V_mat.py --fname '../sample_data/3_6_Vmat__.txt' --m 3 --n 6 -verbose -visualize
```

# Laundry List
- [ ] The method is under developement! coordinate the update, developement, and debugging of the code with Jennifer David.
- [ ] Add a script to just check the validity of the V matrix, and if it is not valid, print what exactly is the problem.
- [ ] store the history of `V`, `E_task`, `P`, `R`, `L` in `.txt`.
- [ ] Add documentation and API examples.

# License
Distributed with a GNU GENERAL PUBLIC LICENSE;
see [LICENSE](https://github.com/saeedghsh/arrangement/blob/master/LICENSE).
```
Copyright (C) Saeed Gholami Shahbandi
```
