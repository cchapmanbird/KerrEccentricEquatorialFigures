# KerrEccentricEquatorialFigures

Repository containing Figure production scripts and necessary data products for Chapman-Bird et al. (2025):

___The Fast and the Frame-Dragging: Efficient waveforms for asymmetric-mass
eccentric equatorial inspirals into rapidly-spinning black holes___

These scripts were used in conjunction with the `PaperProduction` tag of FEW (found [here](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms/releases/tag/PaperProduction)).

## Figure reproduction

Due to the large number of Figures in the paper, this repository is large and does not maintain consistent styling throughout. The following list describes how each Figure can be reproduced from the contents of the repository. Note that in some cases, required data products are too large to fit here, but are available on request.

From the `scripts` directory:

1. `Implementation/tf_waveform`: run `data_generation.py` followed by `produce_plot.py`.
2. `Implementation/dense_output`: run `data_generation.py` followed by `produce_plot.py`.
3. `Implementation/ODE_error_timing_dephasing`: run `generate_data.py` followed by `produce_plot.py`.
4. `Implementation/Amplitude_heatmap`: run `data_generation.py` followed by `produce_plot.py`.
5. `Validation/comparison_perturbation_club_flux`: run `flux_comparison_BHPC_data.py` followed by `flux_comparison_BHPC_plot.py`.
6. `Validation/Fluxes/FluxComparison`: run `sah_near_iso_comparison.py`.
7. `Validation/Fluxes/DownsampleComparison`: run `plot_comparison.py`.
8. `Validation/Trajectory/Flux_Interpolation_Resolution`: run `DownsampledFluxesDataGeneration.py` followed by `DownsampledFluxesPlotting.py`.
9. `Validation/Amplitude/GREMLINComparison`: run `generate-data.py` followed by `produce-plot.py`.
10. `Validation/Amplitude/BicubicTricubicComparison`: run `generate_data.py` followed by `produce_plot.py`.
11. `Validation/Waveform/`: run `plot_mismatch_downsample.py`.
12. Same as above.
13. `Validation/Waveform/AmplitudeComparison`: run `generate-data.py` followed by `make-plots.py`.
14. `Results/timing`: run `GeneratePlot.py`.
15. `Results/AAK_Kerr_Comparisons/process_data`: run `produce_plot.py`.
16. `Results/PE_studies/higher-modes`: execute the `Mahalonobis_approx_Fishers.ipynb` notebook.
17. `Results/horizon`: run `plot_horizon_data.py`.
18. `Results/low_e0_mismatch/data_for_plots`: execute the `plot_data.ipynb` notebook.
19. `Results/PE_studies`: run `plot_science_cases.py`.
20. A script for this Figure is not provided, but can be generated with the functions provided in `few.utils.mappings.kerrecceq`.
21. `flux_computational_time`: run `timing.py`. Note that this requires access to the underlying data.
22. `Validation/Cross_tests`: execute the `Phaseshift_plot_vs_a.ipynb` notebook.
23. `Validation/Cross_tests`: execute the `Amplitudes_cross_check.ipynb` notebook.
24. `Validation/Cross_tests`: execute the `mismatches_plot.ipynb` notebook.
25. `Validation/Fluxes`: run `ComparisonDataGeneration.py` followed by `ComparisonPlotData.py`.
26. `Validation/PN_amplitude_comparisons`: run `GenerateAmpComparisonData.py` followed by `GenerateAmpComparisonMagnitudePlot.py`.
27. `Validation/Trajectory/KerrPNTrajectoryComparison`: run `generate_data.py` followed by `produce_plot.py`.
28. `Results/PE_studies/mcmc_code`: execute the `paper_corner_plots.ipynb` notebook. Also produces Figures 29-32.
