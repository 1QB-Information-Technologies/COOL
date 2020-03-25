# An example of a 16x16 Spin Glass model:

## Training:

```bash
export LATTICE_L=16
export OMP_NUM_THREADS=8
python train.py --tag SG_16
```

## Testing:
For testing, we must specify a directory containing the Hamiltonian instances that
we have solved using the Spin Glass Server, as well as the tag that was used to train the model

```bash
export LATTICE_L=16
export OMP_NUM_THREADS=8
python test.py --tag SG_16 --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/16x16/validation/ 
```


## Baseline:
For benchmarking against traditional SA, we should use the same test Hamiltonian problems, but specify `--beta_init` and `--beta_end` for the linear schedule


```bash
export LATTICE_L=16
export OMP_NUM_THREADS=8
python sa_baseline.py --tag SA_BASELINE --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/16x16/validation/ --beta_init=0.1 --beta_end=3.0
```


## Destructive observation

To operate in destructive observation mode, add the `--destructive` flag, e.g.


```bash 
export LATTICE_L=16
export OMP_NUM_THREADS=16
## Train
python train.py --tag SG_16 --destructive
## Test
python test.py --tag SG_16 --hamiltonian_directory=${COOL_HOME}/latticefiles/toroidal2d/RND_J/16x16/validation/  --destructive
```


