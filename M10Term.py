import autograd.numpy as np
from celerite import terms
from SED_Model import lrt_model
from scipy.interpolate import interp1d

class M10_model(object):
    def __init__(self, lamda, fluxes, flux_errors, survey='SDSS', Mi_catalog=None, z_catalog=None):
        self.lamda = lamda
        fluxes = np.array(fluxes)
        flux_errors = np.array(flux_errors)
        
        # Define band information for different surveys
        if survey.upper() == 'SDSS':
            band_info = [
                {"range": (3000, 4000), "lamda": 3543, "jband_loc": 0},  # u-band
                {"range": (4000, 5000), "lamda": 4770, "jband_loc": 1},  # g-band
                {"range": (5000, 6500), "lamda": 6231, "jband_loc": 2},  # r-band
                {"range": (6500, 8000), "lamda": 7625, "jband_loc": 3},  # i-band
                {"range": (8000, 9500), "lamda": 9134, "jband_loc": 4}   # z-band
            ]
        elif survey.upper() == 'LSST':
            band_info = [
                {"range": (3000, 4000), "lamda": 3724, "jband_loc": 0},  # u-band
                {"range": (4000, 5000), "lamda": 4807, "jband_loc": 1},  # g-band
                {"range": (6000, 6500), "lamda": 6221, "jband_loc": 2},  # r-band
                {"range": (7000, 8000), "lamda": 7559, "jband_loc": 3},  # i-band
                {"range": (8000, 9000), "lamda": 8680, "jband_loc": 4},  # z-band
                {"range": (9000, 10000), "lamda": 9753, "jband_loc": 5}  # y-band
            ]
        else:
            raise ValueError(f"Survey '{survey}' not supported. Use 'SDSS' or 'LSST'.")

        # Match the wave with the correct band
        self.jband_loc = None
        for band in band_info:
            if band["range"][0] <= self.lamda <= band["range"][1]:
                self.jband_loc = band["jband_loc"]
                break
        
        if self.jband_loc is None:
            raise ValueError(f"Wave {self.lamda} is out of the expected range for {survey}.")

        # Initialize the object
        self.gal = lrt_model()
        self.gal.jyuse = np.ones(len(fluxes), dtype=np.int32)
        self.gal.jy = fluxes
        self.gal.ejy = flux_errors  # Error in flux for the current band
        self.gal.jyuse[self.gal.jy == 0] = 2  # Replace the jyuse=2 to use upperlimits: SSS
        min_ejy = 0.05 * self.gal.jy  # 5% of the flux value
        cond = self.gal.ejy < min_ejy  # Condition where error is below the minimum
        self.gal.ejy[cond] = min_ejy[cond]  # Enforce the noise floor

        # Perform fits for a range of redshifts
        self.redshifts = np.arange(0.3, 5.0, 0.01)
        self.Mi_values = []
        self.fnu_agn_values = []

        for z in self.redshifts:
            self.gal.zspec = z
            self.gal.kc_fit()

            if self.gal.comp[0] > 0:
                # Get total flux
                self.gal.get_model_fluxes()
                fnu_total = np.copy(self.gal.jymod)

                # Get AGN flux
                self.gal.comp[1:] = 0.
                self.gal.get_model_fluxes()
                fnu_agn = np.copy(self.gal.jymod)
                self.gal.ebv = np.array(0.)
                # Save Mi and fnu_agn
                self.Mi_values.append(self.gal.abs_mag[3])
                self.fnu_agn_values.append(fnu_agn[self.jband_loc] / fnu_total[self.jband_loc])
            else:
                self.Mi_values.append(np.nan)
                self.fnu_agn_values.append(np.nan)

        self.Mi_values = np.array(self.Mi_values)
        self.fnu_agn_values = np.array(self.fnu_agn_values)
        
        self.Mi_interp = interp1d(self.redshifts, self.Mi_values, bounds_error=False, fill_value=np.nan)
        self.fnu_agn_interp = interp1d(self.redshifts, self.fnu_agn_values, bounds_error=False, fill_value=np.nan)

    def get_SFinf_tau(self, redshift):
        # Interpolate Mi and fnu_agn for the given redshift
        Mi = self.Mi_interp(redshift)
        fnu_agn_ratio = self.fnu_agn_interp(redshift)
        
        if np.isnan(Mi) or np.isnan(fnu_agn_ratio):
            return 1e32, 1e-32  # Handle out-of-range cases

        # Estimate tau and SFinf
        tau = self._get_tau(Mi, redshift) * (1 + redshift)
        SFinf = self._get_SFinf(Mi, redshift) * fnu_agn_ratio
        return SFinf, tau

    def _get_tau(self, Mi, redshift):
        lam_RF = self.lamda / (1 + redshift)
        with open("coefficients.txt", "r") as f:
            lines = f.readlines()
            if lines:
                tau_values = lines[1].strip().split("{")[1].split("}")[0].split(",")
                A, B, C = map(float, map(str.strip, tau_values))
            else:
                raise ValueError("File is empty or not in expected format.")

        return 10. ** (A + B * np.log10(lam_RF / 4000.) + C * (Mi + 23))

    def _get_SFinf(self, Mi, redshift):
        lam_RF = self.lamda / (1 + redshift)
        with open("coefficients.txt", "r") as f:
            lines = f.readlines()
            if lines:
                sf_values = lines[0].strip().split("{")[1].split("}")[0].split(",")
                A, B, C = map(float, map(str.strip, sf_values))
            else:
                raise ValueError("File is empty or not in expected format.")

        return 10. ** (A + B * np.log10(lam_RF / 4000.) + C * (Mi + 23))

    @staticmethod
    def compute_SFinf_tau_from_Mi_fnu(lamda, Mi, fnu_agn_ratio, MBH, redshift, coeff_file="coefficient_bhmass.txt"):
        lam_RF = lamda / (1 + redshift)

        with open(coeff_file, "r") as f:
            lines = f.readlines()

            sf_values = lines[0].strip().split("{")[1].split("}")[0].split(",")
            A_sf, B_sf, C_sf, D_sf = map(float, map(str.strip, sf_values))

            tau_values = lines[1].strip().split("{")[1].split("}")[0].split(",")
            A_tau, B_tau, C_tau, D_tau = map(float, map(str.strip, tau_values))

        tau = 10. ** (A_tau + B_tau * np.log10(lam_RF / 4000.) + C_tau * (Mi + 23) + D_tau * np.log10(MBH / 1e9)) * (1 + redshift)
        SFinf = 10. ** (A_sf + B_sf * np.log10(lam_RF / 4000.) + C_sf * (Mi + 23) + D_sf * np.log10(MBH / 1e9)) * fnu_agn_ratio

        return SFinf, tau

def safe_to_float(x):
    """
    Converts a scalar or an autograd ArrayBox to a regular float.
    If x is an ArrayBox, it returns the unwrapped value.
    """
    if isinstance(x, (np.ndarray, float, int)):
        return float(x)
    elif hasattr(x, '_value'):
        return float(x._value)
    else:
        raise TypeError(f"Cannot convert type {type(x)} to float")

class MacTerm(terms.Term):
    parameter_names = ("redshift",)

    def MacTerm_setup(self, lamda, fluxes, flux_errors, survey='SDSS'):
        # Initialize the M10 object with survey specification
        self.M10obj = M10_model(lamda, fluxes, flux_errors, survey=survey)
        return 
    
    def __repr__(self):
        return "MacTerm({0.redshift})".format(self)

    def get_real_coefficients(self, params):
        redshift = params
        SFinf, tau = self.M10obj.get_SFinf_tau(safe_to_float(redshift))
        return (SFinf)**2, 1./tau
