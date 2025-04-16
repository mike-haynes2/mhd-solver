import numpy as np                       # type: ignore
import matplotlib.pyplot as plt          # type: ignore
import matplotlib.animation as animation # type: ignore
from mpl_toolkits.mplot3d import Axes3D  # type: ignore


# Helper functions
def save_animation(anim, filename):
    # anim -- animation object to save
    # filename -- string for the filename to save the animation as
    # saves as "filename.gif"

    # saves animation as gif
    anim.save(filename, writer='pillow')
    print("Animation saved as " + filename)


# 1D Animation for scalar and vector fields
def generate_1d_animation(scalars, vectors, spatial_coords, time_coords,
                          save_stills, stills_frames, stills_prefix, staggered_plot=True):

    # scalars  -- list of tuples (name, data) for scalar fields
    # vectors  -- list of tuples (name, data) for vector fields
    # spatial_coords -- 1D array of spatial coordinates
    # time_coords -- 1D array of time coordinates
    # save_stills -- boolean to save stills
    # stills_frames -- list of frames to save as stills
    # stills_prefix -- prefix for stills filenames

    # staggered_plot -- boolean to stagger the plot of quantities (default is False)

    # identify how many things we need to plot
      
    
    n_plots = len(scalars) + len(vectors)
    
    n_plots_scalars = len(scalars) 
    n_plots_vectors = 3*len(vectors)
    
    # create matplotlib objects
    fig, axs = plt.subplots(n_plots, 1, figsize=(8, 2 * n_plots), sharex=True)
    
    # exception handling for the case theres only one thing to plot 
    if n_plots == 1: 
        axs = [axs]


    line_objects = {} # array to hold all the ax.plot objects generated
    idx = 0

    # Plot scalar data
    # iterates through all data in scalars
    for name, data in scalars: # grab data and metadata from scalars list

        # plot 
        line, = axs[idx].plot(spatial_coords, data[0, :], label=name)
        
        # add plot metadata
        axs[idx].set_ylabel(name)
        axs[idx].legend(loc = "upper right")
        
        # add ax.plot object to container
        line_objects[name] = line
        
        idx += 1

    
    # Plot vector data (three components per vector)
    for name, data in vectors: # grab data and metadata from vectors list  
        
        comp_lines = [] # array to hold ax.plot objects for each component of the vector
        
        # plot each component of the vector
        for comp in range(3):
            line, = axs[idx].plot(spatial_coords, data[0, :, comp], label=f"{name}_{comp}")
            comp_lines.append(line)

        # add plot metadata
        axs[idx].set_ylabel(name)
        axs[idx].legend(loc = "upper right")

        # add ax.plot objects to container
        # this is a list of ax.plot objects, one for each component of the vector
        line_objects[name] = comp_lines
        idx += 1

    # Set common x-axis label
    axs[-1].set_xlabel("Spatial coordinate")

    # function to update the plot for each frame of the animation
    def update(frame): 

        # Update scalar data
        for name, data in scalars:
            line_objects[name].set_ydata(data[frame, :])
        
        # Update vector data
        for name, data in vectors:
            for comp in range(3):
                line_objects[name][comp].set_ydata(data[frame, :, comp])
        
        # Update titles with current time value
        i = 0
        for name, _ in scalars: # grab data and metadata from scalars list
            axs[i].set_title(f"{name} t={time_coords[frame]:.2f}")
            i += 1
        for name, _ in vectors: # grab data and metadata from vectors list
            axs[i].set_title(f"{name} t={time_coords[frame]:.2f}")
            i += 1

        return []
    
    # Create the animation object
    ani = animation.FuncAnimation(fig, update, frames=len(time_coords), interval=100, blit=False)
    
    # save stills if requested
    if save_stills:

        # iterate through the stills frames and save them
        for frame in stills_frames:
            update(frame) # update the plot to the correct frame
            fig.savefig(f"{stills_prefix}_frame{frame}.png") # save the still
            print(f"Saved still: {stills_prefix}_frame{frame}.png") # print confirmation and filename

        if staggered_plot:
            
            if n_plots_scalars != 0:
                comp_fig, comp_axs =  plt.subplots(n_plots_scalars, 1, figsize=(8, 2 * n_plots_scalars), sharex=True)
            
            
            if n_plots_vectors != 0: 
                comp_fig_vec, comp_axs_vec = plt.subplots(n_plots_vectors, 1, figsize=(8, 2 * n_plots_vectors), sharex=True)

            if n_plots == 1: 
                comp_axs = [comp_axs]


            # Choose a colormap and generate colors for each frame
            cmap = plt.get_cmap('plasma')
            n_frames = len(stills_frames)
            # In case of a single frame avoid division by zero:
            if n_frames > 1:
                colors = [cmap(i / (n_frames - 1)) for i in range(n_frames)]
            else:
                colors = [cmap(0.5)]
            
            idx = 0

              
            
            # overplot all the scalar data
            for i, (name, data) in enumerate(scalars): # grab data and metadata from scalars list
                for j, frame in enumerate(stills_frames):

                    # plot the data at the current frame
                    comp_axs[i].plot(spatial_coords, 
                                        data[frame, :], 
                                        label=f"t={time_coords[frame]:.2f}", 
                                        color = colors[j],
                                        alpha=0.5)
                    
                    # add plot metadata
                    comp_axs[i].set_ylabel(name)
                    comp_axs[i].legend(loc = "upper right")
                    idx += 1

            # Plot vector data (three components per vector)


            for j, (name, data) in enumerate(vectors): # grab data and metadata from vectors list
                
                ax_index = j # index for the vector data
                for comp in range(3):
                    for k, frame in enumerate(stills_frames):
                        comp_axs_vec[ax_index].plot(spatial_coords, 
                                                data[frame, :, comp], 
                                                label=f"t={time_coords[frame]:.2f}",
                                                color = colors[k], 
                                                alpha=0.5)
                        
                        comp_axs_vec[ax_index].set_ylabel(name)
                        comp_axs_vec[ax_index].legend(loc = "upper right")
                        

            comp_axs[-1].set_xlabel("Spatial coordinate")
            comp_fig.savefig(f"{stills_prefix}_comp.png") # save the still
            print(f"Saved still: {stills_prefix}_comp.png") # print confirmation and filename

    # return the animation object        
    return ani


# 3D Animation for B field
# responsible for plotting stills and animations the same way as 1D but for the B field
def generate_3d_animation_B(B, spatial_coords, time_coords, save_stills, stills_frames, stills_prefix):
    # B -- 3D array of B field data (time, spatial, components)
    # spatial_coords -- 1D array of spatial coordinates
    # time_coords -- 1D array of time coordinates
    # save_stills -- boolean to save stills
    # stills_frames -- list of frames to save as stills
    # stills_prefix -- prefix for stills filenames



    
    # create matplotlib objects and a 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot the initial frame as a scatter plot
    # B[0, :, 1] is By and B[0, :, 2] is Bz
    sc = ax.scatter(spatial_coords, B[0, :, 1], B[0, :, 2], c='b')

    # set labels and title
    ax.set_xlabel("Spatial coordinate")
    ax.set_ylabel("By")
    ax.set_zlabel("Bz")
    ax.set_title(f"B 3D t={time_coords[0]:.2f}")
    

    # helper function to update the plot for each frame of the animation
    def update(frame):

        # B[frame, :, 1] is By and B[frame, :, 2] is Bz
        xs = spatial_coords
        ys = B[frame, :, 1]
        zs = B[frame, :, 2]

        sc._offsets3d = (xs, ys, zs) # update the scatter plot data
        ax.set_title(f"B 3D t={time_coords[frame]:.2f}") # update the title with the current time value
        return [sc] # return the scatter plot object
    
    # Create the animation object
    ani3d = animation.FuncAnimation(fig, update, frames=len(time_coords), interval=100, blit=False)
    

    # save stills if requested
    if save_stills:
        for frame in stills_frames: # iterate through the stills frames and save them
            update(frame) # update the plot to the correct frame
            fig.savefig(f"{stills_prefix}_frame{frame}.png") # save the still
            print(f"Saved still: {stills_prefix}_frame{frame}.png") # print confirmation and filename
    
    # return the animation object
    return ani3d

# Main script function
def visualize_mhd_data(rho=None, p=None, energy=None, V=None, B=None,
                       spatial_coords=None, time_coords=None,
                       animate=False, save_stills=True, stills_frames=None):
    
    # rho -- 3D array of density data (time, spatial)
    # p -- 3D array of pressure data (time, spatial)
    # energy -- 3D array of energy data (time, spatial)

    # V -- 3D array of velocity data (time, spatial, components)
    # B -- 3D array of magnetic field data (time, spatial, components)

    # spatial_coords -- 1D array of spatial coordinates
    # time_coords -- 1D array of time coordinates
    # animate -- boolean to animate the plots
    # save_stills -- boolean to save stills
    # stills_frames -- list of frames to save as stills


    # Organize scalar and vector data into lists
    scalars = []
    if rho is not None:
        scalars.append(("rho", rho))
    if p is not None:
        scalars.append(("p", p))
    if energy is not None:
        scalars.append(("energy", energy))
        
    vectors = []
    if V is not None:
        vectors.append(("V", V))
    if B is not None:
        vectors.append(("B", B))
    
    # Use provided time coordinates or default to index range
    if time_coords is None:
        if rho is not None:
            time_coords = np.arange(rho.shape[0])
        else:
            time_coords = np.arange(V.shape[0])


    #Provide default spatial_coords if not provided
    if spatial_coords is None:
        if p is not None:
            spatial_coords = np.arange(p.shape[1])
        elif energy is not None:
            spatial_coords = np.arange(energy.shape[1])
        elif rho is not None:
            spatial_coords = np.arange(rho.shape[1])
        elif V is not None:
            spatial_coords = np.arange(V.shape[1])
        elif B is not None:
            spatial_coords = np.arange(B.shape[1])

    # if no snapshot timeframes are provided, use the first and last time step as snapshot times
    if stills_frames is None:
        stills_frames = [0, len(time_coords)-1]
        

    # check to see if user wants to animate plots
    if animate:
        
        ani1d = None
        ani3d = None
        
        #check if there is 1D data to animate
        if len(scalars) != 0:
            ani1d = generate_1d_animation(scalars, vectors, spatial_coords, time_coords,
                                          save_stills, stills_frames, "1d")

        
        if B is not None:
            ani3d = generate_3d_animation_B(B, spatial_coords, time_coords,
                                            save_stills, stills_frames, "3d")
        
        
        
        # Save animations as GIFs
        if ani1d is not None:
            save_animation(ani1d, "1d_animation.gif")
        if ani3d is not None:
            save_animation(ani3d, "3d_animation.gif")
            
        plt.show()
        return ani1d, ani3d
    
    
    
    # if not animating, just generate and show plots at timesteps
    else:
        
        if B is not None:
            generate_3d_animation_B(B, spatial_coords, time_coords,
                                    save_stills, stills_frames, "3d")
        if scalars:
            name, data = scalars[0]
            generate_1d_animation(scalars, vectors, spatial_coords, time_coords,
                                  save_stills, stills_frames, "1d")
            
        plt.show()



# Example usage
if __name__ == "__main__":

    # N timesteps and spatial steps
    n_timesteps = 100
    n_spatialsteps = 200

    # spatial coords and time coords (can be left out, will default to the range of the data)
    spatial_coords = np.linspace(0, 10, n_spatialsteps)
    time_coords = np.linspace(0, 5, n_timesteps)
    
    # Generate random data for rho, p, energy, V, and B
    # rho, p, energy are 2D arrays (time, spatial)
    rho = np.array([np.sin(spatial_coords) * np.cos(0.1 * t) for t in range(n_timesteps)])
    p = np.array([np.cos(spatial_coords) * np.sin(0.1 * t) for t in range(n_timesteps)])
    energy = np.array([np.sin(spatial_coords + 0.2 * t) for t in range(n_timesteps)])
    
    
    
    # V,B is a 3D array (time, spatial, components)
    V = np.array([np.column_stack((np.sin(spatial_coords + 0.1 * t),
                                    np.cos(spatial_coords + 0.1 * t),
                                    np.sin(spatial_coords + 0.1 * t) * np.cos(spatial_coords)))
                  for t in range(n_timesteps)])
    B = np.array([np.column_stack((np.full(n_spatialsteps, 1.0),  # Constant Bx
                                   np.cos(spatial_coords + 0.1 * t),
                                   np.sin(spatial_coords + 0.1 * t)))
                  for t in range(n_timesteps)])
    
    # example usage of the visualize_mhd_data function
    # Animate all plots and save stills at frames 0, 50, and 99.
    visualize_mhd_data(rho = rho, p = p, energy = None, V = None, B = None, spatial_coords = None, time_coords = None,
                       animate=True, save_stills=True, stills_frames=[10,12,14,16,18,20,22,24,26,28,30])
