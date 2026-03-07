import torch
from typing import Optional


def get_normal_KL(
    mean_1: torch.Tensor,
    log_std_1: torch.Tensor,
    mean_2: Optional[torch.Tensor] = None,
    log_std_2: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    :Parameters:
    mean_1: means of normal distributions (1)
    log_std_1 : standard deviations of normal distributions (1)
    mean_2: means of normal distributions (2)
    log_std_2 : standard deviations of normal distributions (2)
    :Outputs:
    kl divergence of the normal distributions (1) and normal distributions (2)
    ---
    This function should return the value of KL(p1 || p2),
    where p1 = Normal(mean_1, exp(log_std_1) ** 2), p2 = Normal(mean_2, exp(log_std_2) ** 2).
    If mean_2 and log_std_2 are None values, we will use standard normal distribution.
    Note that we consider the case of diagonal covariance matrix.
    """
    # If mean_2 and log_std_2 are None, use standard normal distribution
    if mean_2 is None:
        mean_2 = torch.zeros_like(mean_1)
    if log_std_2 is None:
        log_std_2 = torch.zeros_like(log_std_1)
    
    # Convert to standard deviations and variances
    std_1 = torch.exp(log_std_1)
    std_2 = torch.exp(log_std_2)
    var_1 = std_1.pow(2)
    var_2 = std_2.pow(2)
    
    # Calculate KL divergence for normal distribution with diagonal covariance
    # Formula: KL(p1||p2) = 0.5 * [log(var_2/var_1) + var_1/var_2 + (mean_1-mean_2)^2/var_2 - 1]
    
    # Term by term calculation
    log_det_term = torch.log(var_2) - torch.log(var_1)  # log(var_2/var_1)
    var_term = var_1 / var_2  # var_1/var_2
    mean_term = (mean_1 - mean_2).pow(2) / var_2  # (mean_1-mean_2)^2/var_2
    
    # Complete formula
    kl = 0.5 * (log_det_term + var_term + mean_term - 1)
    
    return kl


def get_normal_nll(
    x: torch.Tensor, mean: torch.Tensor, log_std: torch.Tensor
) -> torch.Tensor:
    """
    This function should return the negative log likelihood log p(x),
    where p(x) = Normal(x | mean, exp(log_std) ** 2).
    Note that we consider the case of diagonal covariance matrix.
    """

    # Convert to variance
    var = torch.exp(2 * log_std)
    
    # Calculate squared difference
    squared_diff = (x - mean).pow(2)
    
    # NLL for univariate normal: 0.5 * [log(2π) + 2*log_std + (x-mean)²/var]
    log_2pi = torch.log(torch.tensor(2 * torch.pi))
    
    # Complete formula for scalar
    nll = 0.5 * (log_2pi + 2 * log_std + squared_diff / var)
    
    return nll