# PyPotts
Task Assignment with Neural Network based on Potts Spin.

<p align="center">
	<img src="https://github.com/saeedghsh/PyPotts/blob/master/docs/animation.gif" width="400">	
</p>

The method is a *Neural Netork based on Potts Spin* for the task assignment problem.
The [original implementation](https://github.com/jenniferdavid/potts_spin) is by [Jennifer David](https://github.com/jenniferdavid) and in C++.
I implemented this Python version under the supervision of Jennifer David.

# Dependencies and Download
Download, installing dependencies, and install package
```shell
# Download
$ git clone https://github.com/saeedghsh/PyPotts.git
$ cd PyPotts

# Install dependencies
$ pip install -r requirements.txt

# Install the package [optional]
python setup.py install
```

# Basic Use and API
* Solve assignment (From `D`, `T` to `V`):
```shell
$ not available yet...
```

* Parsing the `V` matrix (the assignments):
```shell
$ cd scripts/
$ python parse_V_mat.py --fname '../sample_data/3_6_Vmat__.txt' --m 3 --n 6 -verbose -visualize
```

# Laundry List (+ dev-log)
- [ ] write the dev/find_optimal.py script that finds the optimal solution to the input, kinda brute-force, exhausting all the possible solutiuons, but also considering the conditions for the validity of solution (is_V_mat_valid` and `parse_V_to_assignment`) as heuristic to limit the search space (how?).

- [ ] make the history-log-plot interactive, plot3D of the assignment
  at whatever iteration picked (to investigate the phase changing
  oscillation).

- [ ] The method is under developement! coordinate the update,
  developement, and debugging of the code with Jennifer David.

- [ ] Add a script to just check the validity of the V matrix, and if
  it is invalid, print what exactly is the problem.

- [ ] store the history of `V`, `E_task`, `P`, `R`, `L` in
  `.txt`. There is a problem here, there is an inconsistency among
  different configurations of the method.  The length of `E_task`,
  `P`, `R`, `L` change, so its not easy (straightforward) to store the
  results programatically (see examples below) I prefered storing in
  numpy.ndarray, because it would be easier to interact with the
  (e.g. plot and debug), and also file-IO is rather slow.  But then
  storing as array is memeory consuming.  Alternatively I could add a
  debugging mode, where these are stored to file, and add some
  functionality to read-load those required data.
  The first three that are consistent:
  ```python
  # synchronous mode
  config['synchronous'] = True
  # shape of V_log is (len(KT), ndim, ndim)
  # shape of E (local, loop, task, and P, R, L) is (len(KT), ndim, ndim)
  ```
  ,
  ```python
  # asynchronous-batch mode - without frequent update
  config['synchronous'] = False
  config['update_E_local_per_row_col'] = False
  # shape of V_log is (len(KT), ndim, ndim)
  # shape of E (local, loop, task, and P, R, L) is (len(KT), ndim, ndim)
  ```
  ,
  ```python
  # asynchronous-beta mode - one row/column per iteration
  config['synchronous'] = False
  config['update_temperature_after_each_row_col'] = True
  # shape of V_log is (len(KT), ndim, ndim)
  # shape of E (local, loop, task, and P, R, L) is (len(KT), ndim, ndim)
  ```
  but this config is not consistent with the previous ones:
  ```python
  # asynchronous-batch mode - with frequent update
  config['synchronous'] = False
  config['update_E_local_per_row_col'] = False
  # shape of V_log is (len(KT), ndim, ndim)
  # shape of E (local, loop, task, and P, R, L) is ((m+n)*len(KT), ndim, ndim)
  ```

- [ ] Implement with PyTorch or theano.
  * Started, but there is a bug. In the 1st (or mayby 2nd?) iteration
    `inf` is generated and the process terminates. Don't know where it
    is comming from yet.  I can't do the debugging right now, because
    I'm on the plain and the don't have access to internet, and the
    version of PyTorch on the machine I brought with me is too old and
    cannot even compile the torch version. my 0.1 version (vs the
    current 0.4) don't have the `torch.float` type defined.
  * So, it turns out I forgot to implement the method for converting
    the matrices to doubly connected stochastic... now the PyTorch
    version is working. Althought only the synchronous version is
    implemented, and more importantly, PyTorch is processing with CPU
    and takes more time than numpy version. Next step is to enable GPU
    processing and include asynchronous versions...

- [ ] Add documentation and API examples.


# License
Distributed with a GNU GENERAL PUBLIC LICENSE; see [LICENSE](https://github.com/saeedghsh/arrangement/blob/master/LICENSE).
```
Copyright (C) Saeed Gholami Shahbandi
```
