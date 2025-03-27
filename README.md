# KerrEccentricEquatorialFigures
Repository containing figure scripts + associated data products for the FEW Kerr Eccentric Equatorial paper.

### Figure production guidelines
Figures should be produced with Computer Modern font to maintain style consistency and the seaborn "colorblind" palette. Scatter plots should be rasterized (via the ``rasterized=True`` keyword argument) and plots should be saved in pdf format.

For figures that contain analysis results, please have two separate scripts (one to produce the results, and one to plot them in the figure) as this will make it easier to tweak the figures later. 

To add your scripts etc., please fork this repo and add your script in the appropriately named directory (I've added an example to show what I mean) and then add to the main repo with a merge request. Please let me know if there are any issues with this. 

