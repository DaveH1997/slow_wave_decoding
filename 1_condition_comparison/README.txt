#####################################################################################
Short Description of the Functions/Scripts on the Directory 'local_sleep/figure_1/1c'
#####################################################################################

############################################
Author: David Hof (last updated: 07-06-2025)
############################################


Functions:

– sw_pars_per_participant_v3: Computes the SW parameters per participant and condition (movie/
                              phone/overall), and (optionally) visualizes them in the form of
                              topographical plots. The detection and interpolation of SW
                              parameter outliers is another feature of the function.

– compute_wave_pars_new_v2: Helper function for 'sw_pars_per_participant_v3'.

– estimate_lambda: Helper function for 'sw_pars_per_participant_v3'. Estimates the lambda that
                   maximizes the log-likelihood function for the Box-Cox transformation.

– boxcox_log_lik: Helper function for 'estimate_lambda'. Computes the log-likelihood of the
                  transformed data assuming a normal distribution.

– boxcox_transform: Helper function for 'sw_pars_per_participant_v3'. Performs Box-Cox
                    transformation using the optimal lambda.

– aggregate_sw_pars_v2: Pools the SW parameters per channel and condition (movie/phone/overall)
                        across participants using a preferred aggregation function (e.g., mean,
                        median) and (optionally) visualizes the pooled SW parameters in the form
                        of topographical plots.

– condition_contrast_v2: Performs a paired t-test per SW parameter and per channel to compare
                         the movie and phone conditions. Subsequently performs a cluster-based
                         permutation test to determine significance. (Optionally) visualizes the
                         results in the form of topographical plots.

– visualize_wave_pars_new_v2: Helper function for 'sw_pars_per_participant_v3',
                              'aggregate_sw_pars_v2', and 'condition_contrast_v2'.


Scripts:

– run_paired_t_test_pipeline: Driver script for the paired t-test pipeline.
– LS_sw_pars_v3(.slurm): Batch script for the paired t-test pipeline.