#!/usr/bin/env python

# import and set up packages
from pathlib import Path

import tqdm

import numpy as np
import matplotlib.pyplot as plt

import ard
import floris

import test_rotational_workbench

plt.style.use(ard.get_house_style(use_tex=True))

## configuration settings

path_floris_config = Path.cwd() / "precooked.yaml"
show_plots = False
save_plots = True
save_data = True
run_debug_mode = False

## generate demo plots for the testing workbench setup

if show_plots or save_plots:
    # demo wraparound gaussian pulse resource for nearly single-directional wind
    (WD, WS, FREQ), df_wr = test_rotational_workbench.f_windrose(-5.0, 8.0)
    fig, ax = plt.subplots()
    ct0 = ax.contourf(WD, WS, FREQ)
    ax.set_xlabel("wind direction, (°)")
    ax.set_ylabel("wind speed, (m/s)")
    cb = fig.colorbar(ct0)
    cb.set_label("probability density (-)")
    if save_plots:
        fig.savefig("wind_resource_demo.png", dpi=300, bbox_inches="tight")
    plt.show() if show_plots else plt.close(fig)

    # demo the layout generator w/o rotational symmetries
    fig, ax = plt.subplots()
    ax.scatter(
        *[
            v.flat
            for v in test_rotational_workbench.layout_generator(
                diameter=130.0, orientation=0.0
            )
        ]
    )
    ax.set_xlabel("turbine x location/relative easting, (m)")
    ax.set_ylabel("turbine y location/relative northing, (m)")
    if save_plots:
        fig.savefig("layout_demo.png", dpi=300, bbox_inches="tight")
    plt.show() if show_plots else plt.close(fig)

## set up the aerodynamic models and call functions

# generate the FLORIS model w/ the pre-cooked IEA 3.4 RWT model
floris_model = floris.FlorisModel(path_floris_config)


# a function to run either wind direction or orientation sweeps
def run_floris(wd_val=0.0, orientation=0.0):
    WD, WS, FREQ = test_rotational_workbench.f_windrose(
        wd_center=wd_val,
        ws_center=8.0,
    )[0]
    wind_rose = floris.wind_data.WindRose(
        wind_directions=WD[:, 0],
        wind_speeds=WS[0, :],
        ti_table=0.06,
        freq_table=FREQ,
    )

    X, Y = test_rotational_workbench.layout_generator(D_rotor, orientation)
    floris_model.set(
        wind_data=wind_rose,
        layout_x=X.flatten(),
        layout_y=Y.flatten(),
    )
    floris_model.run()
    return floris_model.get_farm_AEP() / 1e9  # report in GWh


# extract FLORIS data for FLOWERS s.t. the two models are 1:1
turbine_def = floris_model.core.farm.turbine_definitions[0]
D_rotor = turbine_def["rotor_diameter"]
# thrust and power curves
u_cp = u_ct = np.array(turbine_def["power_thrust_table"]["wind_speed"][1:])
cp = (
    np.array(turbine_def["power_thrust_table"]["power"][1:])
    * 1e3
    / (
        0.5
        * turbine_def["power_thrust_table"]["ref_air_density"]
        * (np.pi * D_rotor**2 / 4)
        * u_cp**3
    )
)
ct = np.array(turbine_def["power_thrust_table"]["thrust_coefficient"][1:])
# detect cutin and cutout
U_cutin = float(
    u_cp[
        np.squeeze(
            np.argwhere(
                np.isclose(cp, 0.0) & (np.array(list(range(len(cp)))) < len(cp) / 2)
            )[0]
        )
        + 1
    ]
)
U_cutout = float(
    u_cp[
        np.squeeze(
            np.argwhere(
                np.isclose(cp, 0.0) & (np.array(list(range(len(cp)))) > len(cp) / 2)
            )[0]
        )
        - 1
    ]
)
print(f"detected cut in {U_cutin} and cut out {U_cutout}. verify.")

# pack up flowers model
flowers_turbine = {
    "D": D_rotor,
    "cp": cp,
    "u_cp": u_cp,
    "ct": ct,
    "u_ct": u_ct,
    "U": U_cutout,
}

## run an experiment: move the pulse wind resource around the compass

# generate and run the cases
wd_vec = np.arange(0.0, 360.001, 1.0 if not run_debug_mode else 10.0)
AEP_FLOWERS_e1_vec = np.zeros_like(wd_vec)
AEP_FLORIS_e1_vec = np.zeros_like(wd_vec)
for idx, wd_val in enumerate(tqdm.tqdm(wd_vec)):
    AEP_FLOWERS_e1_vec[idx] = test_rotational_workbench.run_FLOWERS(
        flowers_turbine, wd_val=wd_val
    )
    AEP_FLORIS_e1_vec[idx] = run_floris(wd_val=wd_val)

# plot the results of the experiment
if show_plots or save_plots:
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(wd_vec, AEP_FLOWERS_e1_vec, label="FLOWERS")
    axes[0].plot(wd_vec, AEP_FLORIS_e1_vec, label="FLORIS")
    axes[0].legend()
    axes[1].hist((AEP_FLOWERS_e1_vec - AEP_FLORIS_e1_vec) / AEP_FLORIS_e1_vec * 100)
    axes[1].set_xlabel("percent difference in FLOWERS w.r.t. FLORIS, (\\%)")
    if save_plots:
        fig.savefig("compare_AEP_vs_WD.png", dpi=300, bbox_inches="tight")
    plt.show() if show_plots else plt.close(fig)

## run an experiment: rotate the farm around the compass w/ fixed resource

# generate and run the cases
orientation_vec = np.arange(0.0, 360.001, 1.0 if not run_debug_mode else 10.0)
AEP_FLOWERS_e2_vec = np.zeros_like(orientation_vec)
AEP_FLORIS_e2_vec = np.zeros_like(orientation_vec)
for idx, orientation_val in enumerate(tqdm.tqdm(orientation_vec)):
    AEP_FLOWERS_e2_vec[idx] = test_rotational_workbench.run_FLOWERS(
        flowers_turbine, orientation=orientation_val
    )
    AEP_FLORIS_e2_vec[idx] = run_floris(orientation=orientation_val)

# plot the results of the experiment
if show_plots or save_plots:
    fig, axes = plt.subplots(2, 1)
    axes[0].plot(orientation_vec, AEP_FLOWERS_e2_vec, label="FLOWERS")
    axes[0].plot(orientation_vec, AEP_FLORIS_e2_vec, label="FLORIS")
    axes[0].legend()
    axes[1].hist((AEP_FLOWERS_e2_vec - AEP_FLORIS_e2_vec) / AEP_FLORIS_e2_vec * 100)
    axes[1].set_xlabel("percent difference in FLOWERS w.r.t. FLORIS, (\\%)")
    if save_plots:
        fig.savefig("compare_AEP_vs_orientation.png", dpi=300, bbox_inches="tight")
    plt.show() if show_plots else plt.close(fig)

## save the output data
if save_data:
    np.savez_compressed(
        "rotational_consistency",
        wd_vec=wd_vec,
        AEP_FLOWERS_e1_vec=AEP_FLOWERS_e1_vec,
        AEP_FLORIS_e1_vec=AEP_FLORIS_e1_vec,
        orientation_vec=orientation_vec,
        AEP_FLOWERS_e2_vec=AEP_FLOWERS_e2_vec,
        AEP_FLORIS_e2_vec=AEP_FLORIS_e2_vec,
        flowers_turbine=flowers_turbine,
    )
