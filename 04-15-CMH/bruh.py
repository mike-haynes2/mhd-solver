import os
import numpy as np
from datavis import visualize_mhd_data
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
plt.style.use('dark_background')
import matplotlib.animation as animation
#plt.rcParams['animation.html'] = 'html5' # this is used to display animations in jupyter notebooks
#%config InlineBackend.figure_format = 'svg'

B_y = {};B_z = {}
rho = {};en = {}
u_x = {};u_y = {};u_z = {}
# directory = 'sigmoid_test_10-38-35'# YOU WILL NEED TO CHANGE THIS BASED ON WHAT YOU NEED
directory = 'alpha_test_alfven2_21-32-29'
# SEE LINE 103 AND 107 FOR FORMAT OF THE NAMES IF YOU WANT TO FIND A CLEVER WAY TO DO IT
t_vals =[]
var_vals = []
for file in os.listdir(directory):
    full_path = os.path.join(directory, file)
    split_name = file.split('_')
    split_name
    if split_name[0] == 'Plots': continue
    if split_name[0] == 'sigmoid':
        case = split_name[0]; var = split_name[1]; t = split_name[3]  # val will be either alpha or sigma depending on what directory is commented
    elif split_name[0] == 'case':
        case = split_name[1]; var = split_name[3]; t = split_name[5]

    t_vals.append(t)
    var_vals.append(var)
    data = np.load(full_path, allow_pickle=True)

    B_y[f'{var}_{t}'] = data[f'B_y_{var}_{t}']
    B_z[f'{var}_{t}'] = data[f'B_z_{var}_{t}']
    u_x[f'{var}_{t}'] = data[f'u_x_{var}_{t}']
    u_y[f'{var}_{t}'] = data[f'u_y_{var}_{t}']
    u_z[f'{var}_{t}'] = data[f'u_z_{var}_{t}']
    en[f'{var}_{t}'] = data[f'en_{var}_{t}']
    rho[f'{var}_{t}'] = data[f'rho_{var}_{t}']

var_list = [rho, u_x, u_y, u_z, B_y, B_z, en]
var_names = ['rho', 'u_x', 'u_y', 'u_z', 'B_y', 'B_z', 'en']
# color_dict = mcolors.CSS4_COLORS
# color_list = list(color_dict.keys())

t_vals = np.unique(t_vals)
var_vals = np.unique(var_vals)

# should introduce a parameter to determine whether the vertical axis limits are fixed
def plot_variables(var_dict, name, t_vals, var_vals, save=False, inbuilt_animate=False):

    # initialize empty array to store animation material
    fig = plt.figure(figsize=[12,10])
    var_plots = []
    color_assign = {}
    colors = ['salmon', 'peru', 'gold', 'olive', 'greenyellow', 'forestgreen', 'aquamarine', 'teal', 'deepskyblue', 'lightsteelblue', 'navy', 'mediumpurple', 
            'fuchsia', 'crimson', 'pink', 'grey', 'lightcoral', 'firebrick', 'sandybrown', 'goldenrod', 'darkkhaki', 'chartreuse', 'darkgreen', 'turqoise', 'aqua']
    if len(var_vals) == 1:
        colors = ['cyan']
    linestyles = []

    for i, var in enumerate(var_vals):
        color_assign[var] = colors[2*i]


    for t_val in t_vals:
        for count, (key, value) in enumerate(var_dict.items()):
            var, t = key.split('_')
            #### time plotting ####
            # print(t)
            # print(var)
            if t == t_val:
                var_lab = f"{float(var):.4f}"
                plot_obj = plt.plot(value, label=var_lab, lw=2, color=color_assign[var])
        plt.grid()
        t_val= f"{float(t_val):.4f}"
        title = f'Variable: {name} Time: ' + t_val
        plt.title(title)
        plt.xlabel('Position')
        plt.ylabel(name)
        # plt.legend()
        handles, labels = plt.gca().get_legend_handles_labels()
        plt.legend(*zip(*sorted(zip(handles, labels), key=lambda x: x[1])))
        var_plots.append(plot_obj)
        if save:
            plt.savefig(directory + '/Plots/' + name + '/' + title + '.png')
            plt.close()
            if inbuilt_animate:
                ani = animation.ArtistAnimation(fig, var_plots, interval=100, blit=True, repeat_delay=100)
                ani.save(f'Variable:{name}-animated.svg')
        else: 
            plt.show()

if __name__ == '__main__':

    os.mkdir(directory + '/Plots')
    # plot_variables(var_list[0], var_names[0], t_vals, var_vals, save=False)
    for count, variable in enumerate(var_list):   
        os.mkdir(directory + '/Plots/' + var_names[count]) 
        plot_variables(variable, var_names[count], t_vals, var_vals, save=True)