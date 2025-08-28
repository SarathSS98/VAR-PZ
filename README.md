# VAR-PZ: Constraining the Photometric Redshifts of Quasars using Variability

# Requirements
1. Python < = 3.11.5
2. Celerite
3. Numpy (1.25 or lesser) and autograd
4. Matplotlib
5. LRT (https://github.com/rjassef/LRT/tree/Sarath) (See requirements there)
6. Astropy
   



## Usage
Check out the example.ipynb in demo

Note: The default version of SED modeling uses LRT (Assef et al. 2010) which does not support parallel processing. But, one can split the sample into chunks and initiate the process.
