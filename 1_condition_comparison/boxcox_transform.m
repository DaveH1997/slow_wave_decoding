function y_transformed = boxcox_transform(y, lambda)
    if any(y <= 0)
        error('All data points must be positive for Box-Cox transformation.');
    end
    
    % Threshold to handle lambda near zero
    threshold = 1e-6;
    
    if abs(lambda) < threshold
        y_transformed = log(y);
    else
        y_transformed = (y.^lambda - 1) / lambda;
    end
end