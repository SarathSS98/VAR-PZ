<img src="./Deep%20Space%20Blue%20Galaxy%20Logo.png" alt="Project Logo" width="300" height="200">
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

# The bandmag.dat file needs to be updated with the corresponding filters  

The file bandmag.dat needs to include all the filters you’re using for SED fitting in the format Filter name, Magnitude units (1= Vega mags, 2= IRAC mags, 3 = AB), Zero point

A list of available filter names is here:

👉 https://github.com/rjassef/LRT/tree/Sarath/Filters


Example (SDSS filters)
sdssu   3   3767.

sdssg   3   3631.

sdssr   3   3631.

sdssi   3   3631.

sdssz   3   3565.


 
If you need filters that aren’t listed, you can submit a pull request to the repo

