import floris.tools as wfct
import numpy as np
from scipy.interpolate import NearestNDInterpolator
import time

import flowers.flowers_model as flow
import flowers.optimization.optimization_interface as opt
import flowers.tools.tools as tl

### THE FOLLOWING CODE IS LEGACY FLOWERS CODE AND IS NOT ACTIVELY MAINTAINED OR
### TESTED IN THE MOST RECENT ARD DEPLOYMENT OF THE FLOWERS METHOD


class AEPInterface:
    """
    AEPInterface is a high-level user interface to compare AEP estimates between
    the FLOWERS (analytical) AEP model and the Conventional (numerical) AEP model.

    Args:
        wind_rose (pandas.DataFrame): A dataframe for the wind rose in the FLORIS
            format containing the following information:
                - 'ws' (float): wind speeds [m/s]
                - 'wd' (float): wind directions [deg]
                - 'freq_val' (float): frequency for each wind speed and direction
        layout_x (numpy.array(float)): x-positions of each turbine [m]
        layout_y (numpy.array(float)): y-positions of each turbine [m]
        num_terms (int, optional): number of Fourier modes for the FLOWERS model
        k (float, optional): wake expansion rate in the FLOWERS model
        conventional_model (str, optional): underlying wake model:
                - 'jensen' (default)
                - 'gauss'
        turbine (str, optional): turbine type:
                - 'nrel_5MW' (default)

    """

    ###########################################################################
    # Initialization tools
    ###########################################################################

    def __init__(
        self,
        wind_rose,
        layout_x,
        layout_y,
        num_terms=0,
        k=0.05,
        conventional_model=None,
        turbine=None,
    ):

        self._wind_rose = wind_rose
        self._model = conventional_model

        # Initialize FLOWERS
        self.flowers_interface = flow.FlowersInterface(
            wind_rose, layout_x, layout_y, num_terms=num_terms, k=k, turbine=turbine
        )

        wd_array = np.array(wind_rose["wd"].unique(), dtype=float)
        ws_array = np.array(wind_rose["ws"].unique(), dtype=float)
        wd_grid, ws_grid = np.meshgrid(wd_array, ws_array, indexing="ij")
        freq_interp = NearestNDInterpolator(
            wind_rose[["wd", "ws"]], wind_rose["freq_val"]
        )
        freq = freq_interp(wd_grid, ws_grid)
        self._freq_2D = freq / np.sum(freq)

        self.floris_interface.reinitialize(
            layout_x=layout_x.flatten(),
            layout_y=layout_y.flatten(),
            wind_directions=wd_array,
            wind_speeds=ws_array,
            time_series=False,
        )

    def reinitialize(
        self,
        wind_rose=None,
        layout_x=None,
        layout_y=None,
        num_terms=None,
        wd_resolution=0.0,
        ws_avg=False,
    ):

        # Reinitialize FLOWERS interface
        self.flowers_interface.reinitialize(
            wind_rose=wind_rose,
            layout_x=layout_x,
            layout_y=layout_y,
            num_terms=num_terms,
        )

        # Reinitialize FLORIS interface
        if wind_rose is not None:
            self._wind_rose = wind_rose

            wd_array = np.array(wind_rose["wd"].unique(), dtype=float)
            ws_array = np.array(wind_rose["ws"].unique(), dtype=float)
            wd_grid, ws_grid = np.meshgrid(wd_array, ws_array, indexing="ij")
            freq_interp = NearestNDInterpolator(
                wind_rose[["wd", "ws"]], wind_rose["freq_val"]
            )
            freq = freq_interp(wd_grid, ws_grid)
            self._freq_2D = freq / np.sum(freq)

            self.floris_interface.reinitialize(
                wind_directions=wd_array, wind_speeds=ws_array, time_series=False
            )

        if layout_x is not None and layout_y is not None:
            self.floris_interface.reinitialize(
                layout_x=layout_x.flatten(),
                layout_y=layout_y.flatten(),
                time_series=(np.shape(self._freq_2D)[1] == 1),
            )
        elif layout_x is not None and layout_y is None:
            self.floris_interface.reinitialize(
                layout_x=layout_x.flatten(),
                time_series=(np.shape(self._freq_2D)[1] == 1),
            )
        elif layout_x is None and layout_y is not None:
            self.floris_interface.reinitialize(
                layout_y=layout_y.flatten(),
                time_series=(np.shape(self._freq_2D)[1] == 1),
            )

        if wd_resolution > 0.0 or ws_avg:
            if wd_resolution > 1.0:
                wr = tl.resample_wind_direction(
                    self._wind_rose, wd=np.arange(0, 360, wd_resolution)
                )
            else:
                wr = self._wind_rose

            if ws_avg:
                wr = tl.resample_average_ws_by_wd(wr)
                freq = wr.freq_val.to_numpy()
                freq /= np.sum(freq)
                self._freq_2D = np.expand_dims(freq, 1)
                self.floris_interface.reinitialize(
                    wind_directions=wr.wd, wind_speeds=wr.ws, time_series=True
                )
            else:
                wr = tl.resample_wind_speed(wr, ws=np.arange(1.0, 26.0, 1.0))
                wd_array = np.array(wr["wd"].unique(), dtype=float)
                ws_array = np.array(wr["ws"].unique(), dtype=float)
                wd_grid, ws_grid = np.meshgrid(wd_array, ws_array, indexing="ij")
                freq_interp = NearestNDInterpolator(wr[["wd", "ws"]], wr["freq_val"])
                freq = freq_interp(wd_grid, ws_grid)
                self._freq_2D = freq / np.sum(freq)
                self.floris_interface.reinitialize(
                    wind_directions=wd_array, wind_speeds=ws_array, time_series=False
                )

    ###########################################################################
    # AEP methods
    ###########################################################################

    def compute_flowers_aep(self, timer=False):
        """
        Compute farm AEP using the FLOWERS model.

        Args:
            timer (bool, optional): indicate whether wall timer should be
                used for calculation.

        Returns:
            aep (float): farm AEP [Wh]
            elapsed (float, optional): wall time of AEP calculation [s]

        """

        # Time AEP calculation
        if timer:
            elapsed = 0
            for _ in range(5):
                t = time.time()
                aep = self.flowers_interface.calculate_aep()
                elapsed += time.time() - t
            elapsed /= 5
            return aep, elapsed
        else:
            aep = self.flowers_interface.calculate_aep()
            return aep

    def compute_floris_aep(self, timer=False):
        """
        Compute farm AEP using the FLORIS model.

        Args:
            timer (bool, optional): indicate whether wall timer should be
                used for calculation.

        Returns:
            aep (float): farm AEP [Wh]
            elapsed (float, optional): wall time of AEP calculation [s]

        """

        # Time AEP calculation
        if timer:
            elapsed = 0
            for _ in range(5):
                t = time.time()
                self.floris_interface.calculate_wake()
                aep = np.sum(
                    self.floris_interface.get_farm_power() * self._freq_2D * 8760
                )
                elapsed += time.time() - t
            elapsed /= 5
            return aep, elapsed

        else:
            self.floris_interface.calculate_wake()
            aep = np.sum(self.floris_interface.get_farm_power() * self._freq_2D * 8760)
            return aep

    def compare_aep(self, timer=True, display=True):
        """
        Compute farm AEP using both models and compare. The calculation is
            repeated an optional number of instances to average computation time.
            A table of relevant information is printed to the terminal.

        Args:
            iter (int, optional): the number of times AEP should be computed
                to average the wall time of each calculation.
            num_terms (int, optional): for FLOWERS, the number of Fourier modes
                to compute AEP in the range [1, ceiling(num_wind_directions/2)]
            ws_avg (bool, optional): for FLORIS, to indicate whether wind speed
                should be averaged for each wind direction bin.
            wd_resolution (float, optional): for FLORIS, the width of the discrete
                wind direction bins to compute AEP

        """

        if timer:
            aep_flowers, time_flowers = self.compute_flowers_aep(timer=True)
            aep_floris, time_floris = self.compute_floris_aep(timer=True)
        else:
            aep_flowers = self.compute_flowers_aep(timer=False)
            aep_floris = self.compute_floris_aep(timer=False)

        if display:
            print("============================")
            print("    AEP Results    ")
            print("    FLORIS Model:  " + str(self._model).capitalize())
            print(
                "    Number of Turbines: {:.0f}".format(
                    len(self.flowers_interface.get_layout()[0])
                )
            )
            print(
                "    FLOWERS Terms: {:.0f}".format(
                    self.flowers_interface.get_num_modes()
                )
            )
            print(
                "    FLORIS Bins:   [{:.0f},{:.0f}]".format(
                    len(self._freq_2D[:, 0]), len(self._freq_2D[0, :])
                )
            )
            print("----------------------------")
            print("FLOWERS AEP:      {:.3f} GWh".format(aep_flowers / 1.0e9))
            print("FLORIS  AEP:      {:.3f} GWh".format(aep_floris / 1.0e9))
            print(
                "Percent Difference:  {:.1f}%".format(
                    (aep_flowers - aep_floris) / aep_floris * 100
                )
            )
            if timer:
                print("FLOWERS Time:       {:.3f} s".format(time_flowers))
                print("FLORIS Time:        {:.3f} s".format(time_floris))
            print("Factor of Improvement: {:.1f}x".format(time_floris / time_flowers))
            print("============================")

        if timer:
            return (aep_flowers, aep_floris), (time_flowers, time_floris)
        else:
            return (aep_flowers, aep_floris)


class WPLOInterface:
    """
    WPLOInterface is a high-level user interface to initialize and run wind plant
    layout optimization studies with the FLOWERS (analytical) AEP model and the
    Conventional (numerical) AEP model as the objective function.

    Args:
        wind_rose (pandas.DataFrame): A dataframe for the wind rose in the FLORIS
            format containing the following information:
                - 'ws' (float): wind speeds [m/s]
                - 'wd' (float): wind directions [deg]
                - 'freq_val' (float): frequency for each wind speed and direction
        layout_x (numpy.array(float)): x-positions of each turbine [m]
        layout_y (numpy.array(float)): y-positions of each turbine [m]
        boundaries (list(tuple(float, float))): (x,y) position of each boundary point [m]
        num_terms (int, optional): number of Fourier modes for the FLOWERS model
        k (float, optional): wake expansion rate in the FLOWERS model
        conventional_model (str, optional): underlying wake model:
                - 'jensen' (default)
                - 'gauss'
        turbine (str, optional): turbine type:
                - 'nrel_5MW' (default)

    """

    def __init__(
        self,
        wind_rose,
        layout_x,
        layout_y,
        boundaries,
        num_terms=10,
        k=0.05,
        conventional_model=None,
        turbine=None,
    ):

        self._initial_x = layout_x
        self._initial_y = layout_y
        self._model = conventional_model
        self._boundaries = boundaries

        if conventional_model is None or conventional_model == "gauss":
            self.floris_interface = wfct.floris_interface.FlorisInterface(
                "./input/gauss.yaml"
            )
        elif conventional_model == "jensen":
            self.floris_interface = wfct.floris_interface.FlorisInterface(
                "./input/jensen.yaml"
            )

        # Initialize FLOWERS interface
        self.flowers_interface = flow.FlowersInterface(
            wind_rose, layout_x, layout_y, num_terms=num_terms, k=k, turbine=turbine
        )

        # Initialize FLORIS interface
        wr = tl.resample_wind_direction(wind_rose, wd=np.arange(0, 360, 5.0))
        wr = tl.resample_average_ws_by_wd(wr)
        freq = wr.freq_val.to_numpy()
        freq /= np.sum(freq)
        self._freq_1D = np.expand_dims(freq, 1)
        self.floris_interface.reinitialize(
            wind_directions=wr.wd,
            wind_speeds=wr.ws,
            layout_x=layout_x.flatten(),
            layout_y=layout_y.flatten(),
            time_series=True,
        )

        # Initialize post-processing interface
        self.post_processing = wfct.floris_interface.FlorisInterface(
            "./input/post.yaml"
        )
        wind_rose = tl.resample_wind_speed(wind_rose, ws=np.arange(1.0, 26.0, 1.0))
        wd_array = np.array(wind_rose["wd"].unique(), dtype=float)
        ws_array = np.array(wind_rose["ws"].unique(), dtype=float)
        wd_grid, ws_grid = np.meshgrid(wd_array, ws_array, indexing="ij")
        freq_interp = NearestNDInterpolator(
            wind_rose[["wd", "ws"]], wind_rose["freq_val"]
        )
        freq = freq_interp(wd_grid, ws_grid)
        self._freq_2D = freq / np.sum(freq)
        self.post_processing.reinitialize(
            layout_x=layout_x.flatten(),
            layout_y=layout_y.flatten(),
            wind_directions=wd_array,
            wind_speeds=ws_array,
            time_series=False,
        )

        # Calculate initial AEP
        self.post_processing.calculate_wake()
        self._aep_initial = np.sum(
            self.post_processing.get_farm_power() * self._freq_2D * 8760
        )

    def run_optimization(
        self,
        optimizer,
        gradient="analytic",
        solver="SLSQP",
        scale=1e3,
        tol=1e-2,
        timer=None,
        history="hist.hist",
        output="out.out",
    ):
        """
        Run a Wind Plant Layout Optimization study with either the FLOWERS
        or Conventional optimizer.

        Args:
            optimizer (str): the objective function to use in the study:
                - "flowers"
                - "conventional"
            solver (str, optional): the optimization algorithm to use:
                - "SLSQP" (default)
                - "SNOPT"
            timer (int, optional): time limit [s]
            history (str, optional): file name for pyoptsparse history file
            output (str, optional): file name for solver output file

        Returns:
            solution (dict): relevant information from the optimization solution:
                - "init_x" (numpy.array(float)): initial x-positions of each turbine [m]
                - "init_y" (numpy.array(float)): initial y-positions of each turbine [m]
                - "init_aep" (float): initial plant AEP [Wh]
                - "opt_x" (numpy.array(float)): optimized x-positions of each turbine [m]
                - "opt_y" (numpy.array(float)): optimized y-positions of each turbine [m]
                - "opt_aep" (float): optimized plant AEP [Wh]
                - "hist_x" (numpy.array(float,float)): x-positions of each turbine at each solver iteration [m]
                - "hist_y" (numpy.array(float,float)): y-positions of each turbine at each solver iteration [m]
                - "hist_aep" (numpy.array(float)): plant AEP at each solver iteration [Wh]
                - "iter" (int): number of major iterations taken by the solver
                - "obj_calls" (int): number of AEP function evaluations
                - "grad_calls" (int): number of gradient evaluations
                - "total_time" (float): total solve time
                - "obj_time" (float): time spent evaluating objective function
                - "grad_time" (float): time spent evaluating gradients
                - "solver_time" (float): time spent solving optimization problem

        """

        # Instantiate optimizer class with user inputs
        if optimizer == "flowers":
            prob = opt.FlowersOptimizer(
                self.flowers_interface,
                self._initial_x,
                self._initial_y,
                self._boundaries,
                grad=gradient,
                solver=solver,
                scale=scale,
                tol=tol,
                timer=timer,
                history_file=history,
                output_file=output,
            )
        elif optimizer == "conventional":
            prob = opt.ConventionalOptimizer(
                self.floris_interface,
                self._freq_1D,
                self._initial_x,
                self._initial_y,
                self._boundaries,
                grad=gradient,
                solver=solver,
                scale=scale,
                tol=tol,
                timer=timer,
                history_file=history,
                output_file=output,
            )

        # Solve optimization problem
        print("Solving layout optimization problem.")
        sol = prob.optimize()
        print("Optimization complete: " + str(sol.optInform["text"]))

        # Define solution dictionary and gather data
        self.solution = dict()
        self.solution["init_x"] = self._initial_x
        self.solution["init_y"] = self._initial_y
        self.solution["opt_x"], self.solution["opt_y"] = prob.parse_sol_vars(sol)
        self.solution["total_time"] = float(sol.optTime)
        self.solution["obj_time"] = float(sol.userObjTime)
        self.solution["grad_time"] = float(sol.userSensTime)
        self.solution["solver_time"] = float(sol.optCodeTime)
        self.solution["obj_calls"] = int(sol.userObjCalls)
        self.solution["grad_calls"] = int(sol.userSensCalls)
        self.solution["exit_code"] = sol.optInform["text"]
        self.solution["init_aep"] = self._aep_initial

        # Get number of iterations
        with open(output, "r") as fp:
            for line in fp:
                if "No. of major iterations" in line:
                    self.solution["iter"] = int(line.split()[4])
                    break

        # Compute optimal AEP
        self.post_processing.reinitialize(
            layout_x=self.solution["opt_x"].flatten(),
            layout_y=self.solution["opt_y"].flatten(),
            time_series=False,
        )
        self.post_processing.calculate_wake()
        self._aep_final = np.sum(
            self.post_processing.get_farm_power() * self._freq_2D * 8760
        )
        self.solution["opt_aep"] = self._aep_final

        return self.solution
