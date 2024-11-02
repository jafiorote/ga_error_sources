import numpy as np
from scipy.stats import norm
from ga_error_sources.model.proteinSystemModel import ProteinSystemModel


class NormPSM(ProteinSystemModel):

    """
    Implements a normal distribution model of protein system model.

    """
        
    def __init__(self, M:int, i_0:float, i_nat:float, sigma2_0:float, n_bins:int, verbose:bool=False):

        """ 
        Parameters
        ----------
        - M (int): 
            Number of sequences in MSA;
        - i_0 (float): 
            Mean of I for systems arrangments with n = 0;
        - i_nat (float): 
            I for system native arrangment;
        - sigma2_0 (float): 
            Variance of I for systems arrangments with n = 0;
        """

        super().__init__(M, i_0, i_nat, sigma2_0, n_bins, verbose)


    def statistical_func(self, i: float, sigma2: float, expec: float):

        """
        Implements normal pmf for a I distribution in a protein system.
        
        Parameters
        ----------
        - i (float): I value to be fitted
        - sigma2 (float): variance of I
        - expec (float): expected value of I
        
        Returns
        -------
        float: Probability density function value at i
        """

        return norm.pdf(i, loc=expec, scale=np.sqrt(sigma2))
    

    def prob_interval(self, i1, i2, sigma2, expec):
        
        cdf_i2 = norm.cdf(i2, loc=expec, scale=np.sqrt(sigma2))
        cdf_i1 = norm.cdf(i1, loc=expec, scale=np.sqrt(sigma2))
        
        prob = cdf_i2 - cdf_i1
        
        return prob

