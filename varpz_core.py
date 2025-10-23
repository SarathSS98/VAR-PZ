import numpy as np
import pandas as pd
from scipy.optimize import minimize
import celerite
from celerite import terms
from M10Term import MacTerm, M10_model  # Import M10_model here

class VAR_PZ_Calculator:
    """Core VAR-PZ calculation that works with any light curve"""
    
    def __init__(self):
        """
        Initialize VAR-PZ calculator
        No need to pass sed_model_class anymore
        """
        self.sed_model_class = M10_model  # Use M10_model internally
    
    def neg_log_like(self, params, y, gp, z):
        """Negative log likelihood function for optimization"""
        PARAMS = np.array([z, params[0]])  
        gp.set_parameter_vector(PARAMS)      
        return -gp.log_likelihood(y)
    
    def grad_neg_log_like(self, params, y, gp, z):
        """Gradient of negative log likelihood function"""
        PARAMS = np.array([z, params[0]])
        gp.set_parameter_vector(PARAMS)  
        return -gp.grad_log_likelihood(y)[1][1]
    
    def log_like(self, y, gp, z):
        """Compute log likelihood and chi2 for given redshift"""
        PARAMS = np.array([z])
        gp.set_parameter_vector(PARAMS)
        
        # Compute log-likelihood
        loglike = gp.log_likelihood(y)
        
        # Compute chi2 from residuals
        resid = y - gp.mean.get_value(gp._t)
        chi2 = gp.solver.dot_solve(resid)
        
        return loglike, chi2
    
    def mle(self, y, gp, initial_params, bounds, z):
        """Maximum likelihood estimation for given redshift"""
        x0 = initial_params[1]  
        bounds = [bounds[1]]
        soln = minimize(self.neg_log_like, x0, jac=self.grad_neg_log_like, 
                       method="L-BFGS-B", bounds=bounds, args=(y, gp, z))
        return -soln.fun
    
    def calculate_band_log_likelihood(self, time, mag, mag_err, wavelength, 
                                    fluxes, flux_errors, z_values=None, survey='SDSS'):
        """
        Calculate log likelihood for a single band light curve
        
        Parameters:
        -----------
        time : array-like
            Time values of the light curve
        mag : array-like
            Magnitude values
        mag_err : array-like
            Magnitude errors
        wavelength : float
            Wavelength of the band in Angstrom
        fluxes : array-like
            Array of fluxes for SED fitting
        flux_errors : array-like
            Array of flux errors for SED fitting
        z_values : array-like, optional
            Redshift values to evaluate (default: 0.01 to 5.0 in steps of 0.01)
        survey : str, optional
            Survey name for SED model (default: 'SDSS')
            
        Returns:
        --------
        log_likelihoods : array
            Log likelihood values for each redshift
        chi2_values : array
            Chi-squared values for each redshift
        """
        if z_values is None:
            z_values = np.arange(0.01, 5.0, 0.01)
        
        # Initialize SED model for this band
        sed_model = self.sed_model_class(wavelength, fluxes, flux_errors, survey=survey)
        
        # Initialize kernel
        kernel = MacTerm(redshift=1.5, bounds=dict(redshift=(0.3, 5)))
        kernel.MacTerm_setup(wavelength, fluxes, flux_errors, survey=survey)
        
        # Add jitter term for uncertainties
        amplitude = np.max(mag + mag_err) - np.min(mag - mag_err)
        smin = -10
        smax = np.log(amplitude)
        log_s = np.mean([smin, smax])
        kernel += terms.JitterTerm(log_sigma=log_s, bounds=dict(log_sigma=(smin, smax)))
        
        # Initialize Gaussian Process
        gp = celerite.GP(kernel, mean=np.mean(mag), fit_mean=False)
        gp.compute(time, mag_err)
        
        initial_params = gp.get_parameter_vector()
        bounds = gp.get_parameter_bounds()
        
        log_likelihoods = []
        chi2_values = []
        
        for z in z_values:
            try:
                # Get log likelihood using MLE
                loglike = self.mle(mag, gp, initial_params, bounds, z)
                log_likelihoods.append(loglike)
                
                # Also get chi2 for completeness
                _, chi2 = self.log_like(mag, gp, z)
                chi2_values.append(chi2)
                
            except Exception as e:
                print(f"Warning: Failed at z={z:.2f}: {str(e)}")
                log_likelihoods.append(np.nan)
                chi2_values.append(np.nan)
        
        return np.array(log_likelihoods), np.array(chi2_values), z_values

# Utility functions
def mag_to_flux(mag, mag_err):
    """Convert magnitude to flux with proper error propagation"""
    flux = 10**(-0.4 * (mag + 48.6))  # Convert to Jy
    flux_err = flux * mag_err * 0.4 * np.log(10)  # Error propagation
    return flux, flux_err

def save_results(z_values, log_likelihoods, chi2_values, output_file):
    """Save results to file"""
    results_df = pd.DataFrame({
        'redshift': z_values,
        'log_likelihood': log_likelihoods,
        'chi2': chi2_values
    })
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

