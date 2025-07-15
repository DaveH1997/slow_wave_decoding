function ll = boxcox_log_lik(y, lambda)
    % Number of observations
    n = length(y);
    
    % Apply Box-Cox transformation
    if lambda == 0
        y_transformed = log(y);
    else
        y_transformed = (y.^lambda - 1) / lambda;
    end
    
    % Calculate log-likelihood assuming normality
    % Using MLE, variance is estimated with denominator n
    variance = sum((y_transformed - mean(y_transformed)).^2) / n;
    
    % Add a small epsilon for numerical stability
    epsilon = 1e-10;
    variance = variance + epsilon;
    
    % Compute log-likelihood
    ll = (lambda - 1) * sum(log(y)) - (n / 2) * log(variance);
end