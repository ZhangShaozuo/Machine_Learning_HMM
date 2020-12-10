# Run Instruction
  
The code should be run in folder ***your_path/Machine_Learning_HMM***.
## Part 2

To run part 2 in console:

```
# training_file: labled data for training
# input_file: unlabeled data to be annotated
# output_file: annotated data file
# e.g. python part2.py EN/train EN/dev.in EN/dev.p2.out
python part2.py training_file input_file output_file
```
The output file can be found at the location of *output_file*.


## Part 3 

To run part 3 in console:

```
# <N>: running mode -- which folder to execute
# N=0 -> EN
# N=1 -> CN
# N=2 -> SG
# N=3 -> test
python part3.py <N>
```
The output file can be found in folder ***(EN, CN, SG, test)/part3***.


## Part 4

To run part 4 in console:

```
# <N>: running mode: which folder to execute
# N=0 -> EN
# N=1 -> CN
# N=2 -> SG
# N=3 -> test
python part4.py <N>
```
The output file can be found in folder ***(EN, CN, SG, test)/part4***.

## Part 5

To run part 5 in console:

```
# <N>: running mode: which folder to execute
# N=0 -> EN
# N=1 -> test
python part5.py <N>
```
The output file can be found in folder ***(EN, test)/part5***.
