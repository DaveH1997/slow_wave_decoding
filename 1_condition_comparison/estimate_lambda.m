function lambda_opt = estimate_lambda(y)
    % Ensure all data are positive
    if any(y <= 0)
        error('All data points must be positive for Box-Cox transformation.');
    end
    
    % Define the negative log-likelihood function for Box-Cox
    neg_log_lik = @(lambda) -boxcox_log_lik(y, lambda);
    
    % Optimize lambda within a reasonable range, e.g., [-5, 5]
    lambda_opt = fminbnd(neg_log_lik, -5, 5);
end