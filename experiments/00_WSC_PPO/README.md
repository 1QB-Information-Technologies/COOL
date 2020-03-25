# Weak-strong clusters model

## Train the model:

```bash 
#!/bin/bash
export LATTICE_L=4
export OMP_NUM_THREADS=18
python train.py --tag WSC --hamiltonian_directory=${COOL_HOME}/latticefiles/weakstrongcluster/
```

## Test the model and print the performance:
```bash
#!/bin/bash
export LATTICE_L=4
export OMP_NUM_THREADS=18
python test.py --tag WSC --hamiltonian_directory=${COOL_HOME}/latticefiles/weakstrongcluster/ --betainit=0.5 
```

## SA benchmark:
```bash
#!/bin/bash
export LATTICE_L=4
export OMP_NUM_THREADS=18
###the tag below defines beta_init (e.g. 0.1) and beta_end (e.g. 3.0) of a linear schedule
python sa_baseline.py --tag WSC_SA-0.1-3.0 --hamiltonian_directory=${COOL_HOME}/latticefiles/weakstrongcluster/
```

