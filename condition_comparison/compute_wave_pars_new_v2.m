function [wave_pars] = compute_wave_pars_new_v2(top10_filtered_results, fs_original, options)

arguments
    top10_filtered_results struct;
    fs_original double;
    options.channels double = [];
end

wave_pars = struct();

if isempty(options.channels)

    % Computing slow-wave density for all channels
    wvspermin = arrayfun(@(x) length([x.maxnegpkamp{:}]) / (x.datalength / 1000 / 60), top10_filtered_results.channels);
    
    % Computing mean peak-to-peak amplitude for all channels
    p2pamp = arrayfun(@(x) mean(abs(double([x.maxnegpkamp{:}])) + double([x.maxpospkamp{:}])), top10_filtered_results.channels);
    
    % Computing mean downward slope for all channels
    dslope = arrayfun(@(x) mean(abs(double([x.maxnegpkamp{:}])) ./ ((double([x.maxnegpk{:}]) - double([x.negzx{:}])) / fs_original)), top10_filtered_results.channels);
    
    % Computing mean upward slope for all channels
    uslope = arrayfun(@(x) mean((abs(double([x.maxnegpkamp{:}])) + double([x.maxpospkamp{:}])) ./ ((double([x.maxpospk{:}]) - double([x.maxnegpk{:}])) / fs_original)), top10_filtered_results.channels);

else

    % Computing slow-wave density for all specified channels
    wvspermin = arrayfun(@(x) length([x.maxnegpkamp{:}]) / (x.datalength / 1000 / 60), top10_filtered_results.channels(options.channels));
    
    % Computing mean peak-to-peak amplitude for all specified channels
    p2pamp = arrayfun(@(x) mean(abs(double([x.maxnegpkamp{:}])) + double([x.maxpospkamp{:}])), top10_filtered_results.channels(options.channels));
    
    % Computing mean downward slope for all specified channels
    dslope = arrayfun(@(x) mean(abs(double([x.maxnegpkamp{:}])) ./ ((double([x.maxnegpk{:}]) - double([x.negzx{:}])) / fs_original)), top10_filtered_results.channels(options.channels));
    
    % Computing mean upward slope for all specified channels
    uslope = arrayfun(@(x) mean((abs(double([x.maxnegpkamp{:}])) + double([x.maxpospkamp{:}])) ./ ((double([x.maxpospk{:}]) - double([x.maxnegpk{:}])) / fs_original)), top10_filtered_results.channels(options.channels));

end

wave_pars.wvspermin = wvspermin;
wave_pars.p2pamp = p2pamp;
wave_pars.dslope = dslope;
wave_pars.uslope = uslope;

end