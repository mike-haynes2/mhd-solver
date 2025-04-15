import numpy as np                       # type: ignore
import matplotlib.pyplot as plt          # type: ignore
import matplotlib.animation as animation # type: ignore
from mpl_toolkits.mplot3d import Axes3D  # type: ignore
import os

def aggregate_snapshots(data_dict):
    # Assumes dictionary keys contain a time identifier at the end separated by an underscore.
    # Modify the key extraction below to suit your key naming convention.
    sorted_keys = sorted(data_dict.keys(), key=lambda k: float(k.split('_')[-1]))
    return np.array([data_dict[k] for k in sorted_keys])


# Helper functions
def save_animation(anim, filename):
    # anim -- animation object to save
    # filename -- string for the filename to save the animation as
    # saves as "filename.gif"

    # saves animation as gif
    anim.save(filename, writer='pillow')
    print("Animation saved as " + filename)


# 1D Animation for scalar and vector fields
def generate_1d_animation(scalars, spatial_coords, time_coords,
                          save_stills, stills_frames, staggered_plot=True,ani_interval = 100):
    """
    scalars -- list of tuples (name, data) for scalar fields
    spatial_coords -- 1D array of spatial coordinates
    time_coords -- 1D array of time coordinates
    save_stills -- boolean to save stills
    stills_frames -- list of frames to save as stills
    stills_prefix -- prefix for stills filenames
    staggered_plot -- boolean to stagger the plot of quantities (default is True)
    """

    # Number of scalar fields to plot
    n_plots = len(scalars)

    # Create matplotlib objects
    fig, axs = plt.subplots(n_plots, 1, figsize=(8, 2 * n_plots), sharex=True)

    # Handle case where there's only one plot
    if n_plots == 1:
        axs = [axs]

    # Dictionary to hold line objects for each scalar field
    line_objects = {}

    # Initialize plots for each scalar field
    for idx, (name, data) in enumerate(scalars):
        line, = axs[idx].plot(spatial_coords, data[0, :], label=name)
        axs[idx].set_ylabel(name)
        axs[idx].legend(loc="upper right")
        line_objects[name] = line

    # Set common x-axis label
    axs[-1].set_xlabel("Spatial coordinate")

    # Function to update the plot for each frame of the animation
    def update(frame):
        for name, data in scalars:
            line_objects[name].set_ydata(data[frame, :])
        for idx, (name, _) in enumerate(scalars):
            axs[idx].set_title(f"{name} t={time_coords[frame]:.2f}")
        return []

    # Create the animation object
    ani = animation.FuncAnimation(fig, update, frames=len(time_coords), interval=ani_interval, blit=False)

    # Save stills if requested
    if save_stills:
        for frame in stills_frames:
            stills_prefix = "stills"
            update(frame)
            fig.savefig(f"plots/{stills_prefix}_frame{frame}.png")
            print(f"Saved still: plots/{stills_prefix}_frame{frame}.png")

        # Handle staggered plotting
        if staggered_plot:
            stills_prefix = "stagered"
            comp_fig, comp_axs = plt.subplots(n_plots, 1, figsize=(8, 2 * n_plots), sharex=True)

            # Handle case where there's only one plot
            if n_plots == 1:
                comp_axs = [comp_axs]

            # Choose a colormap and generate colors for each frame
            cmap = plt.get_cmap('plasma')
            n_frames = len(stills_frames)
            
            # Avoid division by zero for a single frame
            if n_frames > 1:
                colors = [cmap(i / (n_frames - 1)) for i in range(n_frames)]
            else:
                colors = [cmap(0.5)]

            # Overplot all the scalar data
            for i, (name, data) in enumerate(scalars):
                
                for j, frame in enumerate(stills_frames):
                    comp_axs[i].plot(spatial_coords,
                                     data[frame, :],
                                     label=f"t={time_coords[frame]:.2f}",
                                     color=colors[j],
                                     alpha=0.5)
                
                # Add plot metadata
                comp_axs[i].set_ylabel(name)
                comp_axs[i].legend(loc="upper right")

            comp_axs[-1].set_xlabel("Spatial coordinate")
            comp_fig.savefig(f"plots/{stills_prefix}.png")
            print(f"Saved staggered plot: plots/{stills_prefix}.png")

    return ani


# Main script function
def visualize_mhd_data(filename, 
                        rho_plot=True, energy_plot=False, ux_plot=False, uy_plot=False, uz_plot=False, by_plot=False, bz_plot=False,
                       
                       spatial_coords=None, time_coords=None,
                       save=True, stills_frames=None):
    
    
    B_y, B_z, u_x, u_y, u_z, en, rho = getScalarsFromFile(filename)
    
    

    
    # converts dictionaries to numpy arrays
    rho = aggregate_snapshots(rho)
    en = aggregate_snapshots(en)
    u_x = aggregate_snapshots(u_x)
    u_y = aggregate_snapshots(u_y)
    u_z = aggregate_snapshots(u_z)
    B_y = aggregate_snapshots(B_y)
    B_z = aggregate_snapshots(B_z)

    # if asked to draw data, add it to list
    scalars = []
    if rho_plot is True:
        scalars.append(("rho", rho))
    if energy_plot is True:
        scalars.append(("Energy", en))
    if ux_plot is True:
        scalars.append(("Ux", u_x))
    if uy_plot is True:
        scalars.append(("Uy", u_y))
    if uz_plot is True:
        scalars.append(("Uz", u_z))
    if by_plot is True:
        scalars.append(("By", B_y))
    if bz_plot is True:
        scalars.append(("Bz", B_z))
        
    

    # Use provided time coordinates or default to index range
    if time_coords is None:
        time_coords = np.arange(scalars[0][1].shape[0])  # Access the numpy array within the first tuple

    # Provide default spatial_coords if not provided
    if spatial_coords is None:
        spatial_coords = np.arange(scalars[0][1].shape[1]) 

    # If no snapshot timeframes are provided, use the first and last time step as snapshot times
    if stills_frames is None:
        stills_frames = [0, len(time_coords)-1]
        

    ani1d = generate_1d_animation(scalars, spatial_coords, time_coords,save, stills_frames)
        
    # Save animations as GIFs
    if (ani1d is not None) and (save is True):
        ani1d.save("plots/1D-animation.gif", writer='pillow')
        print("Animation saved.")
        
    plt.show()
    


def getScalarsFromFile(filename):
    B_y = {}
    B_z = {}
    rho = {}
    en = {}
    u_x = {}
    u_y = {}
    u_z = {}
    
    directory = filename
    
    
    for file in os.listdir(directory):
        full_path = os.path.join(directory, file)
        split_name = file.split('_')
        if split_name[0] == 'sigmoid':
            case = split_name[0]
            var = split_name[1]
            t = split_name[3]  # val will be either alpha or sigma depending on what directory is commented
        else:
            case = split_name[1]
            var = split_name[3]
            t = split_name[5]

        data = np.load(full_path)

        B_y[f'{var}_{t}'] = data[f'B_y_{var}_{t}']
        B_z[f'{var}_{t}'] = data[f'B_z_{var}_{t}']
        u_x[f'{var}_{t}'] = data[f'u_x_{var}_{t}']
        u_y[f'{var}_{t}'] = data[f'u_y_{var}_{t}']
        u_z[f'{var}_{t}'] = data[f'u_z_{var}_{t}']
        en[f'{var}_{t}'] = data[f'en_{var}_{t}']
        rho[f'{var}_{t}'] = data[f'rho_{var}_{t}']
        
    return B_y, B_z, u_x, u_y, u_z, en, rho

    
    
    
    
    

# Example usage
if __name__ == "__main__":


        

    # Example usage of the visualize_mhd_data function
    # plot data for rho and energy
    # stills at frames 0, 10, 20, 30
    # animate
    # generate staggered graph
    visualize_mhd_data(
        
        rho_plot=True,
        energy_plot=True,
        
        filename = 'alpha_test_Brio&Wu_13-47-45',
        
        save=True,
        stills_frames=[10, 12, 15, 17,19],
    )