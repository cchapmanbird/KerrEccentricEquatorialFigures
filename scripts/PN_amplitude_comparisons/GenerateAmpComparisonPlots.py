import matplotlib.pyplot as plt
import numpy as np

#Plot style 1

def plot_diffs(array,n):

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern"]

    label_fontsize = 14
    tick_fontsize = 14
    title_fontsize = 16

    # Extract the e, a, z1=real part amp rel diff, and z2=im part amp rel diff values from the array
    e = array[:, 0]
    a = array[:, 1]
    z1 = array[:, 2]
    z2 = array[:, 3]
    

 
    # Create a figure with four subplots
    fig, axes = plt.subplots(2, 1, figsize=(6,10))

    # Plot the pdot rel diff using scatter plot
    scatter1 = axes[0].scatter(e,a, c=z1, cmap='plasma',rasterized=True)
    axes[0].set_title(rf'$\log_{{10}} \left(| 1 - Re[A_{{22{n}}}]^{{FEW}}/ Re[A_{{22{n}}}]^{{PN}} | \right)$', fontsize=title_fontsize)
    axes[0].set_xlabel(r'Eccentricity $(e)$', fontsize=label_fontsize)
    axes[0].set_ylabel(r'Kerr spin $(a)$', fontsize=label_fontsize)
    
    axes[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    fig.colorbar(scatter1, ax=axes[0])

    # Plot the edot rel diff using scatter plot
    scatter2 = axes[1].scatter(e, a, c=z2, cmap='plasma',rasterized=True)
    axes[1].set_title(rf'$\log_{{10}} \left(| 1 - Im[A_{{22{n}}}]^{{FEW}}/ Im[A_{{22{n}}}]^{{PN}} | \right)$', fontsize=title_fontsize)
    axes[1].set_xlabel(r'Eccentricity $(e)$', fontsize=label_fontsize)
    axes[1].set_ylabel(r'Kerr spin $(a)$', fontsize=label_fontsize)
    
    axes[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    fig.colorbar(scatter2, ax=axes[1])

    # Display the plots
    plt.tight_layout()

    figurename=f'A22{n}Comparison5PNp100.pdf'
    plt.savefig(figurename)
    
    #plt.show()  

#Plot style 2

def plot_diffs2(array,n):

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern"]

    label_fontsize = 14
    tick_fontsize = 14
    title_fontsize = 16

    # Extract the e, a, z1=real part amp rel diff, and z2=im part amp rel diff values from the array
    p = array[:, 0]
    e = array[:, 1]
    z1 = array[:, 2]
    z2 = array[:, 3]
    

 
    # Create a figure with four subplots
    fig, axes = plt.subplots(2, 1, figsize=(6,10))

    # Plot the pdot rel diff using scatter plot
    scatter1 = axes[0].scatter(p, e, c=z1, cmap='plasma',rasterized=True)
    axes[0].set_title(rf'$\log_{{10}} \left(| 1 - Re[A_{{22{n}}}]^{{FEW}}/ Re[A_{{22{n}}}]^{{PN}} | \right)$', fontsize=title_fontsize)
    axes[0].set_xlabel(r'Semilatus rectum $(p)$', fontsize=label_fontsize)
    axes[0].set_ylabel(r'Eccentricity $(e)$', fontsize=label_fontsize)
    
    axes[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    fig.colorbar(scatter1, ax=axes[0])

    # Plot the edot rel diff using scatter plot
    scatter2 = axes[1].scatter(p, e, c=z2, cmap='plasma',rasterized=True)
    axes[1].set_title(rf'$\log_{{10}} \left(| 1 - Im[A_{{22{n}}}]^{{FEW}}/ Im[A_{{22{n}}}]^{{PN}} | \right)$', fontsize=title_fontsize)
    axes[1].set_xlabel(r'Semilatus rectum $(p)$', fontsize=label_fontsize)
    axes[1].set_ylabel(r'Eccentricity $(e)$', fontsize=label_fontsize)
    
    axes[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    fig.colorbar(scatter2, ax=axes[1])

    # Display the plots
    plt.tight_layout()


    figurename=f'A22{n}Comparison5PN.pdf'

    plt.savefig(figurename)
    #plt.show()  

#Plot style 3

def plot_diffs3(array,n):

    plt.rcParams["text.usetex"] = True
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Computer Modern"]

    label_fontsize = 14
    tick_fontsize = 14
    title_fontsize = 16

    # Extract the e, a, z1=real part amp rel diff, and z2=im part amp rel diff values from the array
    dp = array[:, 0]
    e = array[:, 1]
    z1 = array[:, 2]
    z2 = array[:, 3]
    

 
    # Create a figure with four subplots
    fig, axes = plt.subplots(2, 1, figsize=(6,10))

    # Plot the pdot rel diff using scatter plot
    scatter1 = axes[0].scatter(dp, e, c=z1, cmap='plasma',rasterized=True)
    axes[0].set_title(rf'$\log_{{10}} \left(| 1 - Re[A_{{22{n}}}]^{{FEW}}/ Re[A_{{22{n}}}]^{{PN}} | \right)$', fontsize=title_fontsize)
    axes[0].set_xlabel(r'Semilatus rectum $(p-p_{LSO})$', fontsize=label_fontsize)
    axes[0].set_ylabel(r'Eccentricity $(e)$', fontsize=label_fontsize)
    
    axes[0].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    fig.colorbar(scatter1, ax=axes[0])

    # Plot the edot rel diff using scatter plot
    scatter2 = axes[1].scatter(dp, e, c=z2, cmap='plasma',rasterized=True)
    axes[1].set_title(rf'$\log_{{10}} \left(| 1 - Im[A_{{22{n}}}]^{{FEW}}/ Im[A_{{22{n}}}]^{{PN}} | \right)$', fontsize=title_fontsize)
    axes[1].set_xlabel(r'Semilatus rectum $(p-p_{LSO})$', fontsize=label_fontsize)
    axes[1].set_ylabel(r'Eccentricity $(e)$', fontsize=label_fontsize)
    
    axes[1].tick_params(axis='both', which='major', labelsize=tick_fontsize)
    fig.colorbar(scatter2, ax=axes[1])

    # Display the plots
    plt.tight_layout()

    figurename=f'A22{n}Comparison5PNStrongField.pdf'

    plt.savefig(figurename)
    #plt.show() 

p1l2m2n0diffs=np.loadtxt('p1l2m2n0diffs.txt')
p1l2m2n1diffs=np.loadtxt('p1l2m2n1diffs.txt')
plot_diffs(p1l2m2n0diffs,0)
plot_diffs(p1l2m2n1diffs,1)

p2l2m2n0diffs=np.loadtxt('p2l2m2n0diffs.txt')
p2l2m2n1diffs=np.loadtxt('p2l2m2n1diffs.txt')
plot_diffs2(p2l2m2n0diffs,0)
plot_diffs2(p2l2m2n1diffs,1)

p3l2m2n0diffs=np.loadtxt('p3l2m2n0diffs.txt')
p3l2m2n1diffs=np.loadtxt('p3l2m2n1diffs.txt')
plot_diffs3(p3l2m2n0diffs,0)
plot_diffs3(p3l2m2n1diffs,1)
