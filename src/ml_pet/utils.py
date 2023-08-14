import torch
import torch.nn as nn
from math import sqrt
import json
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve, auc


class CustomBCELoss(nn.Module):
    def __init__(self, weight=None):
        super(CustomBCELoss, self).__init__()
        self.weight = weight

    def forward(self, x, target):   
        # Add epsilon to avoid taking the logarithm of zero
        epsilon = 1e-4
        x = torch.clamp(x, epsilon, 1 - epsilon)
        
        # Calculate the element-wise loss
        loss = -((1 - target) * torch.log(1 - x) + target * torch.log(x))
        
        # Apply class weighting
        if self.weight is not None:
            loss = loss * (target * self.weight + (1 - target))
            
        # Compute the mean loss
        loss = loss.mean()
        
        return loss
    

def wilson(p, n, z = 1.96):
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z*z / (2*n)
    adjusted_standard_deviation = sqrt((p*(1 - p) + z*z / (4*n)) / n)
    
    lower_bound = (centre_adjusted_probability - z*adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z*adjusted_standard_deviation) / denominator
    
    lower_bound = round(lower_bound, 3)
    upper_bound = round(upper_bound, 3)
    
    return lower_bound, upper_bound

def calculate_p_value(lower1, upper1, lower2, upper2, n1, n2, alpha=0.05):
    confidence_interval1 = (lower1, upper1)  # 95% confidence interval for distribution 1
    confidence_interval2 = (lower2, upper2)  # 95% confidence interval for distribution 2
    alpha = 0.05  # Significance level

    # Calculate means based on the confidence intervals
    mean1 = (confidence_interval1[0] + confidence_interval1[1]) / 2
    mean2 = (confidence_interval2[0] + confidence_interval2[1]) / 2

    # Calculate standard errors based on the confidence intervals
    se1 = (confidence_interval1[1] - confidence_interval1[0]) / (2 * stats.norm.ppf(1 - alpha / 2))
    se2 = (confidence_interval2[1] - confidence_interval2[0]) / (2 * stats.norm.ppf(1 - alpha / 2))

    # Calculate degrees of freedom for Welch's t-test
    df = ((se1 ** 2 / n1) + (se2 ** 2 / n2)) ** 2 / (
        (se1 ** 2 / n1) ** 2 / (n1 - 1) + (se2 ** 2 / n2) ** 2 / (n2 - 1)
    )

    # Calculate t-statistic
    t_statistic = (mean1 - mean2) / np.sqrt((se1 ** 2 / n1) + (se2 ** 2 / n2))

    # Calculate p-value
    p_value = 2 * stats.t.cdf(-abs(t_statistic), df)

    print(f"t-statistic: {t_statistic:.2f}")
    print(f"p-value: {p_value:.4f}")

    # Check if the p-value is less than the significance level
    if p_value < alpha:
        print("Reject the null hypothesis: Distributions are significantly different.")
    else:
        print("Fail to reject the null hypothesis: Distributions are not significantly different.")