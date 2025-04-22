import matplotlib.pyplot as plt
import numpy as np

def plot_heatmaps(array,a,x):

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern"]

    # Extract the p, e, z1=pdot rel diff, and z2=edot rel diff values from the array
    p = array[:, 0]
    e = array[:, 1]
    z1 = array[:, 6]
    z2 = array[:, 7]
    
    lmax=array[:,-4]
    deltaEinf=array[:,-3]
    deltaEh=array[:,-2]

    #Change x-axis from p to p-psep
    deltap=deltaEh=array[:,-1]
    
    # Create a figure with four subplots
    fig = plt.figure(figsize=(4.5,3), dpi=200)
    # Plot the pdot rel diff using scatter plot
    scatter1 = plt.scatter(deltap, e, s=5, c=z1, cmap='plasma',rasterized=True, vmin=-10, vmax=-2)
    plt.xlabel(r'$p-p_\mathrm{sep}$')
    plt.ylabel(r'$e$')
    
    plt.colorbar(scatter1, label=r'$ \log_{10} \left| 1 - \hat{f}_p^\mathrm{FEW}/ \hat{f}_p^\mathrm{BHPC} \right|$')

    if x==-1:
        figurename=f'FluxComparisonBHPC_{a}_retro.pdf'
    elif x==1:
        figurename=f'FluxComparisonBHPC_{a}_pro.pdf'
    else:
        print("Input should be x=1 for prograde or x=-1 for retrograde")

    plt.savefig(figurename, bbox_inches='tight')
    #plt.show()  

compare_a0p9pro=np.loadtxt('compare_a0p9pro.txt')
compare_a0p7pro=np.loadtxt('compare_a0p7pro.txt')
compare_a0p5pro=np.loadtxt('compare_a0p5pro.txt')
compare_a0p3pro=np.loadtxt('compare_a0p3pro.txt')
compare_a0p1pro=np.loadtxt('compare_a0p1pro.txt')

compare_a0p9ret=np.loadtxt('compare_a0p9ret.txt')
compare_a0p7ret=np.loadtxt('compare_a0p7ret.txt')
compare_a0p5ret=np.loadtxt('compare_a0p5ret.txt')
compare_a0p3ret=np.loadtxt('compare_a0p3ret.txt')
compare_a0p1ret=np.loadtxt('compare_a0p1ret.txt')

plot_heatmaps(compare_a0p9pro,0.9,1)
plot_heatmaps(compare_a0p7pro,0.7,1)
plot_heatmaps(compare_a0p5pro,0.5,1)
plot_heatmaps(compare_a0p3pro,0.3,1)
plot_heatmaps(compare_a0p1pro,0.1,1)

plot_heatmaps(compare_a0p9ret,0.9,-1)
plot_heatmaps(compare_a0p7ret,0.7,-1)
plot_heatmaps(compare_a0p5ret,0.5,-1)
plot_heatmaps(compare_a0p3ret,0.3,-1)
plot_heatmaps(compare_a0p1ret,0.1,-1)