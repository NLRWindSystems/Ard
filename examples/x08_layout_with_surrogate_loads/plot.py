import matplotlib.pyplot as plt
import openmdao.api as om
import numpy as np
import dill as pickle
import optiwindnet.plotting

plt.rcParams['font.size'] = 20

# Instantiate your CaseReader
# cr = om.CaseReader("optimization_demo_out/gridfarm_9turbs_50iters_COBYLA/opt_results.sql")
case_dir = "ard_prob_out/test_out/cases.sql"
# case_dir = "optimization_demo_out/gridfarm_9turbs_7500maxcablelength_200iters_COBYLA_max_AEP"
cr = om.CaseReader(case_dir + "/opt_results.sql")

# Get driver cases (do not recurse to system/solver cases)
driver_cases = cr.get_cases('driver', recurse=False)

# Get a list of cases for the objective component
layout2aep_cases = cr.list_cases('root.layout2aep', out_stream=None)

# layout = cr.list_cases('root.layout', out_stream=None)

optiwindnet_coll = cr.list_cases('root.optiwindnet_coll', out_stream=None)

# Plot the path the design variables took to convergence
# Note that there are two lines in the right plot because "Z"
# contains two variables that are being optimized
# dv_x_values = []
# dv_z_values = []
# for case in driver_cases:
#     dv_x_values.append(case['tower_base_load'])
    # dv_x_values.append(case['AEP_farm'])
    # dv_z_values.append(case['AEP_farm'])

objectives = []
AEPs = []
for case in driver_cases:
    objectives.append(case.outputs['tower_base_load'])
    # objectives.append(case.outputs['AEP_farm'])

AEPs = []
tower_loads = []
spacing_effective_primary = []
spacing_effective_secondary = []
for case_id in layout2aep_cases:
    case = cr.get_case(case_id)
    AEPs.append(case['AEP_farm'])
    tower_loads.append(case['tower_base_load'])
    spacing_effective_primary.append(case['spacing_effective_primary'])
    spacing_effective_secondary.append(case['spacing_effective_secondary'])

# print(AEPs[-1])
# print(objectives[-1])
# lkj

# x_locations = []
# y_locations = []
# for case_id in layout:
#     case = cr.get_case(case_id)
#     x_locations.append(case['x_locations'])
#     y_locations.append(case['y_locations'])

tot_length_cables = []
for case_id in optiwindnet_coll:
    case = cr.get_case(case_id)
    tot_length_cables.append(case['optiwindnet_coll.total_length_cables'])

# for i, graph in enumerate(graphs):
#     optiwindnet.plotting.gplot(graph)
#     plt.savefig('optimization_demo_out/plots/iter_' + str(i) + '.png')

n_iters = 50

# for i in range(n_iters):
for i in [50]:

    fig, axes= plt.subplots(2, 3, figsize=(21, 12))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

    axes[0,0].plot(np.arange(len(objectives[0:i])), np.array(objectives[0:i]))
    axes[0,0].set(xlabel='Iterations', ylabel='Weighted Sum Tower Base DEL', title='Optimization History')
    axes[0,0].grid()

    axes[0,1].plot(np.arange(len(AEPs[0:i])), np.array(AEPs[0:i]) / 1E9)
    axes[0,1].set(xlabel='Iterations', ylabel='AEP [GWh]', title='Optimization History')
    axes[0,1].grid()

    # axes[0,0].plot(np.arange(len(tower_loads[0:i])), np.array(tower_loads[0:i]))
    # axes[0,0].set(xlabel='Iterations', ylabel='Weighted Sum Tower Base Load', title='Optimization History')
    # axes[0,0].grid()

    # axes[0,1].plot(np.arange(len(objectives[0:i])), -1 * np.array(objectives[0:i]) / 1E9)
    # axes[0,1].set(xlabel='Iterations', ylabel='AEP [GWh]', title='Optimization History')
    # axes[0,1].grid()

    axes[0,2].plot(np.arange(len(tot_length_cables[0:i])), np.array(tot_length_cables[0:i]))
    axes[0,2].set(xlabel='Iterations', ylabel='Total Cable Length [m]', title='Optimization History')
    axes[0,2].grid()

    axes[1,0].plot(
        np.arange(len(spacing_effective_primary[0:i])),
        np.array(spacing_effective_primary[0:i])
    )
    axes[1,0].set(xlabel='Iterations', ylabel='Primary Spacing [D]', title='Optimization History')
    axes[1,0].grid()

    axes[1,1].plot(
        np.arange(len(spacing_effective_secondary[0:i])),
        np.array(spacing_effective_secondary[0:i])
    )
    axes[1,1].set(xlabel='Iterations', ylabel='Secondary Spacing [D]', title='Optimization History')
    axes[1,1].grid()

    file_name = case_dir + '/plots/iter_' + str(i) + '_graph.p'
    with open(file_name, "rb") as file:
        graph = pickle.load(file)
    optiwindnet.plotting.gplot(graph, ax=axes[1,2])

    fig.tight_layout()
    plt.savefig(case_dir + '/plots/iter_' + str(i) + '_plot.png')
    plt.close()

# fig, (ax1, ax2) = plt.subplots(1, 2)
# fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=None)

# ax1.plot(np.arange(len(dv_x_values)), np.array(dv_x_values))
# ax1.set(xlabel='Iterations', ylabel='Design Var: X', title='Optimization History')
# ax1.grid()

# ax2.plot(np.arange(len(dv_z_values)), np.array(dv_z_values))
# ax2.set(xlabel='Iterations', ylabel='Design Var: Z', title='Optimization History')
# ax2.grid()

# plt.show()