#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class methods defining multicompartment neuron populations in the hybrid scheme
"""
import os
# import glob
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpi4py import MPI
# from .gdf import GDF
from . import helpers
from .helperfun import _calc_radial_dist_to_cell, _get_all_SpCells
import LFPy
# import neuron
import scipy.signal as ss
from time import time
import pandas as pd 
import random
import re
from matplotlib.collections import LineCollection

'''
TOSEE: used te fix points of the script that need further explanation.




'''







# ################ Initialization of MPI stuff ############################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()


# ########### class objects ###############################################
class PopulationSuper(object):
    """
    Main population class object, let one set up simulations, execute, and
    compile the results. This class is suitable for subclassing for
    custom cell simulation procedures, inherit things like gathering of
    data written to disk.

    
    PopulationSuper handles all the generic mechanics of creating a population 
    of 3D morphologies in space and managing their data, but it does not include any synapses 
    or network activity.



    Note that `PopulationSuper.cellsim` do not have any stimuli,
    its main purpose is to gather common methods for inherited Population
    objects.

    


    NOTES:

    1) cellindex is fundamentally equivalent to the local_idx you generated in your extract_macro_populations function. 
        It is a unique integer strictly bounded by the size of the specific macro-population currently being simulated.
        If you have a population of 10 cells (cellindex 0 through 9) and are running on 4 MPI ranks (SIZE = 4), the indices are assigned like this:
        Rank 0: Processes cellindex 0, 4, 8
        Rank 1: Processes cellindex 1, 5, 9
        Rank 2: Processes cellindex 2, 6
        Rank 3: Processes cellindex 3, 7






    Parameters
    ----------
    cellParams : dict
        Parameters for class `LFPy.Cell`
    rand_rot_axis : list
        Axis of which to randomly rotate morphs.
    simulationParams : dict
        Additional args for `LFPy.Cell.simulate()`.
    populationParams : dict
        Constraints for population and cell number.
    y : str
        Population identifier string.
    layerBoundaries : list of lists
        Each element is a list setting upper and lower layer boundary (floats)
    probes: list
        list of `LFPykit.models.*` like instances for misc. forward-model
        predictions
    savelist : list
        `LFPy.Cell` arguments to save for each single-cell simulation.
    savefolder : str
        path to where simulation results are stored.
    dt_output : float
        Time resolution of output, e.g., LFP, CSD etc.
    recordSingleContribFrac : float
        fraction  in [0, 1] of individual neurons in population which output
        will be stored
    POPULATIONSEED : int/float
        Random seed for population, for positions etc.
    verbose : bool
        Verbosity flag.
    output_file : str
        formattable string for population output, e.g., '{}_population_{}'


    Returns
    -------
    hybridLFPy.population.PopulationSuper object


    See also
    --------
    Population, LFPy.Cell, LFPy.RecExtElectrode

    """

    def __init__(self,
                 cellParams={
                     'morphology': 'morphologies/ex.hoc',
                     'Ra': 150,
                     'cm': 1.0,
                     'e_pas': 0.0,
                     'lambda_f': 100,
                     'nsegs_method': 'lambda_f',
                     'rm': 20000.0,
                     'dt': 0.1,
                     'tstart': 0,
                     'tstop': 1000.0,
                     'v_init': 0.0,
                     'verbose': False},
                 rand_rot_axis=[],
                 simulationParams={},
                 populationParams={
                     'min_cell_interdist': 1.0,
                     'number': 400,
                     'radius': 100,
                     'z_max': -350,
                     'z_min': -450,
                     'r_z': [[-1E199, 1E99], [10, 10]]},
                 Pop='EX',
                 layerBoundaries=[[0.0, -300], [-300, -500]],

                SubPopulations_list = ['xxxxx.hoc','xxxxx.hoc'],
                layer_names =  ['L1', 'L2/3', 'L4', 'L5', 'L6'],
                Pop_to_Syntype = {'L4':'exc'},
                synapse_base_path = '',
                SubPop_positions = [1, 10.5, -20.2, -150.0],

                cell_mtypes = ['L1','L1', 'L23'],
                mtype_fast_lookup = {},

                input_dict = {},

                local_to_raw_map = {},
                Cell_afferences = {},
                Cell_coords = [],
                global_cell_coords = [],
                voxel_size = None,
                grid_extent = None,
                SpanTree_path = '',

                SynTau =  {}, 
                SynWeigths = {},




                 probes=[],
                 savelist=['somapos'],
                 savefolder='simulation_output_example_brunel',
                 dt_output=1.,
                 recordSingleContribFrac=0,
                 POPULATIONSEED=123456,
                 verbose=False,
                 output_file='{}_population_{}'
                 ):
        """
        Main population class object, let one set up simulations, execute, and
        compile the results. This class is suitable for subclassing for
        custom cell simulation procedures, inherit things like gathering of
        data written to disk.

        Note that `PopulationSuper.cellsim` do not have any stimuli,
        its main purpose is to gather common methods for inherited Population
        objects.


        Parameters
        ----------
        cellParams : dict
            Parameters for class `LFPy.Cell`
        rand_rot_axis : list
            Axis of which to randomly rotate morphs.
        simulationParams : dict
            Additional args for `LFPy.Cell.simulate()`.
        populationParams : dict
            Constraints for population and cell number.
        y : str
            Population identifier string.
        layerBoundaries : list of lists
            Each element is a list setting upper and lower layer boundary as
            floats

        SubPopulations_list: list
            list of the current subpopulation's (self.Pop) morpholgy paths

        layer_names: list 
            Layer names

        synapse_base_path: string
            path to the folder where the per-neuron synapses csv files are held

    
        Pop_to_Syntype : dict
            The keys are the sub-populations while the values are the synapse types 'exc' or 'inh'

        SubPop_positions: array N_SubPop x4
            in the 1st column contains the raw identification number of the subpop neurons as they appear in the ADJ matrix
            the other columns contains the positions in um.


        local_to_raw_map : dict
            Keys are the local id while the values are the same neuron's raw id


        cell_mtypes : 1xN_cells array
            For each cell (in raw indicies) defines its subtype in base of the NMC guidelines.

        mtype_fast_lookup : dict
            For each sub-type (the ones found in cell_mtypes) defines the layer of belonging and the synaptic type

        Cell_afferences : dict
            The key is the post_syn neuron's local index, the value is the list of all the pre_syn neurons' raw indicies along with the 
            specific number of synapses to be distributed on the pre-synaptic arbour.
            It's subpopulation specific which means that the indicies have been reset.


        Cell_coords : Nx3 array
            Position (um) of the cell's somata (idx are associated to the local indexing)

        global_cell_coords: N_tot x3 array
            Position (um) of ALL the cells with adj-related global indexing


        voxel_size : float [um]
            Voxel's edge dimension of the discretized volume to evaluate spanning tree probabilities 

        grid_extent : float [um]
            Cube edge extent in which the spanning tree probabilities have been evaluated 

        SynTau: dict
            Nested dictionaries: at the top level the key is the presynaptic macro-population while the second key is the specific 
            post-synaptic population: The specific value is the decaying time constant 

        SynWeigths: dict 
            Nested dictionaries: at the top level the key is the presynaptic macro-population while the second key is the specific 
            post-synaptic population: The specific value is the synaptic weight


        SpanTree_path: string
            Path to the spanning tree distribution across macro-populations

        input_dict: dict
            dictionary contining infos about layers, cells and column radius

        probes: list
            list of `LFPykit.models.*` like instances for misc. forward-model
            predictions
        savelist : list
            `LFPy.Cell` attributes to save for each single-cell simulation.
        savefolder : str
            path to where simulation results are stored.
        recordSingleContribFrac : float
            fraction  in [0, 1] of individual neurons in population which
            output will be stored
        POPULATIONSEED : int/float
            Random seed for population, for positions etc.
        verbose : bool
            Verbosity flag.
        output_file : str
            formattable string for population output, e.g., '{}_population_{}'


        Returns
        -------
        hybridLFPy.population.PopulationSuper object


        See also
        --------
        Population, LFPy.Cell, LFPy.RecExtElectrode

        """




        self.cellParams = cellParams
        self.dt = self.cellParams['dt']
        self.rand_rot_axis = rand_rot_axis
        self.simulationParams = simulationParams
        self.populationParams = populationParams
        self.Pop = Pop
        self.layerBoundaries = np.array(layerBoundaries)
        self.Subpop = SubPopulations_list
        self.ImplementedMorph = {self.Pop : {}}
        self.PlacedSynapses = {}
        self.layer_names = layer_names
        self.Pop_to_Syntype = Pop_to_Syntype
        self.synapse_base_path = synapse_base_path
        self.SubPop_positions = SubPop_positions



        self.local_to_raw_map = local_to_raw_map
        self.Cell_afferences = Cell_afferences
        self.pop_soma_pos = Cell_coords
        self.global_cell_coords = global_cell_coords
        self.mtype_fast_lookup = mtype_fast_lookup
        self.cell_mtypes = cell_mtypes


        self.grid_extent = grid_extent
        self.voxel_size = voxel_size

        self.input_dict = input_dict
        self.SpanTree_path = SpanTree_path

        self.SynWeigths= SynWeigths
        self.SynTau = SynTau


        self.probes = probes
        self.savelist = savelist
        self.savefolder = savefolder
        self.dt_output = dt_output
        self.recordSingleContribFrac = recordSingleContribFrac
        self.output_file = output_file

        # check that decimate fraction is actually a whole number
        try:
            assert int(self.dt_output / self.dt) == self.dt_output / self.dt
        except AssertionError:
            raise AssertionError('dt_output not an integer multiple of dt')

        self.decimatefrac = int(self.dt_output / self.dt)
        self.POPULATIONSEED = POPULATIONSEED
        self.POPULATION_SIZE = len(self.pop_soma_pos)
        self.verbose = verbose

        # set the random seed for reproducible populations, synapse locations,
        # presynaptic spiketrains
        np.random.seed(self.POPULATIONSEED)

        # using these colors and alphas:
        self.colors = []
        for i in range(self.POPULATION_SIZE):
            i *= 256.
            if self.POPULATION_SIZE > 1:
                i /= self.POPULATION_SIZE - 1.
            else:
                i /= self.POPULATION_SIZE

            try:
                self.colors.append(plt.cm.rainbow(int(i)))
            except BaseException:
                self.colors.append(plt.cm.gist_rainbow(int(i)))

        self.alphas = np.ones(self.POPULATION_SIZE)

        # self.pop_soma_pos = self.set_pop_soma_pos()
        self.rotations = self.set_rotations()

        self._set_up_savefolder()

        self.CELLINDICES = np.arange(self.POPULATION_SIZE)
        self.RANK_CELLINDICES = self.CELLINDICES[self.CELLINDICES % SIZE
                                                 == RANK]

        # container for single-cell output generated on this RANK
        self.output = dict((i, {}) for i in self.RANK_CELLINDICES)

    def _set_up_savefolder(self):
        """
        Create catalogs for different file output to clean up savefolder.

        Non-public method


        Parameters
        ----------
        None


        Returns
        -------
        None

        """
        if self.savefolder is None:
            return

        self.cells_path = os.path.join(self.savefolder, 'cells')
        if RANK == 0:
            if not os.path.isdir(self.cells_path):
                os.mkdir(self.cells_path)

        self.figures_path = os.path.join(self.savefolder, 'figures')
        if RANK == 0:
            if not os.path.isdir(self.figures_path):
                os.mkdir(self.figures_path)

        self.populations_path = os.path.join(self.savefolder, 'populations')
        if RANK == 0:
            if not os.path.isdir(self.populations_path):
                os.mkdir(self.populations_path)

        COMM.Barrier()

    def run(self):
        """
        Distribute individual cell simulations across ranks.

        This method takes no keyword arguments.


        Parameters
        ----------
        None


        Returns
        -------
        None

        """
        for cellindex in self.RANK_CELLINDICES:
            self.cellsim(cellindex)

        COMM.Barrier()




    def cellsim(self, cellindex, return_just_cell=False):
        """
        Single-cell `LFPy.Cell` simulation without any stimulus, mostly for
        reference, as no stimulus is added


        Parameters
        ----------
        cellindex : int
            cell index between 0 and POPULATION_SIZE-1.
        return_just_cell : bool
            If True, return only the LFPy.Cell object
            if False, run full simulation, return None.


        Returns
        -------
        None
            if `return_just_cell is False
        cell : `LFPy.Cell` instance
            if `return_just_cell` is True


        See also
        --------
        LFPy.Cell, LFPy.Synapse, LFPy.RecExtElectrode

        """
        tic = time()

        ### ADD MORPHOLOGY CHECK

        # electrode = LFPy.RecExtElectrode(**self.electrodeParams)

        cellParams = self.cellParams.copy()
        cell = LFPy.Cell(**cellParams)
        cell.set_pos(**self.pop_soma_pos[cellindex])
        cell.set_rotation(**self.rotations[cellindex])

        if return_just_cell:
            return cell
        else:


            ### ADD CELLINDEX SPECIFIC MORPHOLGY




            # set LFPykit.models instance cell attribute
            for probe in self.probes:
                probe.cell = cell

            if 'rec_imem' in self.simulationParams.keys():
                try:
                    assert self.simulationParams['rec_imem']
                    cell.simulate(**self.simulationParams)
                    for probe in self.probes:
                        M = probe.get_transformation_matrix()
                        probe.data = M @ cell.imem
                    del cell.imem
                except AssertionError:
                    cell.simulate(probes=self.probes, **self.simulationParams)
            else:
                cell.simulate(probes=self.probes, **self.simulationParams)

            # downsample probe.data attribute and unset cell
            for probe in self.probes:
                probe.data = ss.decimate(probe.data, q=self.decimatefrac)
                probe.cell = None

            # put all necessary cell output in output dict
            for attrbt in self.savelist:
                attr = getattr(cell, attrbt)
                if isinstance(attr, np.ndarray):
                    self.output[cellindex][attrbt] = attr.astype('float32')
                else:
                    try:
                        self.output[cellindex][attrbt] = attr
                    except BaseException:
                        self.output[cellindex][attrbt] = str(attr)
                self.output[cellindex]['srate'] = 1E3 / self.dt_output

            # collect probe output
            for probe in self.probes:
                if cellindex == self.RANK_CELLINDICES[0]:
                    self.output[probe.__class__.__name__] = \
                        probe.data.copy()
                else:
                    self.output[probe.__class__.__name__] += \
                        probe.data.copy()
                probe.data = None

            # clean up hoc namespace
            cell.__del__()

            print('cell %s population %s in %.2f s' % (cellindex, self.Pop,
                                                       time() - tic))

    def set_pop_soma_pos(self):
        """
        Set `pop_soma_pos` and `pop_neuron_ids` using either predefined 
        coordinates or random generation.

        This method takes no keyword arguments.

        Parameters
        ----------
        None

        Returns
        -------
        numpy.ndarray
            (x,y,z) coordinates of each neuron in the population

        See also
        --------
        PopulationSuper.draw_rand_pos
        """

        tic = time()
        if RANK == 0:
            if self.predefined_positions is not None:
                # Initialize containers
                pop_soma_pos = []
                
                # Column 0 is your raw neuron ID (matched with adjacency matrix)
                self.pop_neuron_ids = self.predefined_positions[:, 0].astype(int)

                # Columns 1, 2, and 3 are x, y, z
                for row in self.predefined_positions:
                    pop_soma_pos.append({
                        'x': row[1],
                        'y': row[2],
                        'z': row[3]
                    })
            else:
                # Default behavior if no predefined positions are provided
                pop_soma_pos = self.draw_rand_pos(**self.populationParams)
                self.pop_neuron_ids = np.arange(self.POPULATION_SIZE)

        else:
            # Initialize as None for all other ranks to prevent MPI AttributeErrors
            pop_soma_pos = None
            self.pop_neuron_ids = None

        if RANK == 0:
            print('found cell positions in %.2f s' % (time() - tic))

        # Synchronize both coordinates and IDs across all MPI ranks
        self.pop_neuron_ids = COMM.bcast(self.pop_neuron_ids, root=0)

        return COMM.bcast(pop_soma_pos, root=0)





    def set_rotations(self):
        """
        Applies a biologically realistic 'jiggle' to the cells.
        Pyramidal apical dendrites are tilted slightly off the vertical Z-axis,
        and cells are randomly spun around their vertical axis.
        """
        tic = time()
        if RANK == 0:
            rotations = []
            
            # Define the maximum tilt in radians (e.g., 15 degrees)
            max_tilt_deg = 10.0
            max_tilt_rad = np.radians(max_tilt_deg)

            for i in range(self.POPULATION_SIZE):
                
                # 1. The "Jiggle" (Tilt)
                # Rotate slightly around X and Y axes to deflect the apical trunk
                # np.random.uniform(low, high) ensures we stay within the +/- limit
                tilt_x = np.random.uniform(-max_tilt_rad, max_tilt_rad)
                tilt_y = np.random.uniform(-max_tilt_rad, max_tilt_rad)

                # 2. The "Spin" (Yaw)
                # The cell can face any direction horizontally
                spin_z = np.random.uniform(0, 2 * np.pi)

                # Optional: If you want interneurons to tumble completely chaotically
                # while keeping pyramidal cells mostly upright, you can check self.Pop here.
                # if 'inh' in self.Pop:
                #     tilt_x = np.random.uniform(0, 2 * np.pi) 
                #     tilt_y = np.random.uniform(0, 2 * np.pi)

                rotations.append({'x': tilt_x, 'y': tilt_y, 'z': spin_z})
        else:
            rotations = None

        if RANK == 0:
            print('applied 15-degree apical jiggle in %.2f s' % (time() - tic))

        return COMM.bcast(rotations, root=0)
    



    def calc_min_cell_interdist(self, x, y, z):
        """
        Calculate cell interdistance from input coordinates.


        Parameters
        ----------
        x, y, z : numpy.ndarray
            xyz-coordinates of each cell-body.


        Returns
        -------
        min_cell_interdist : np.nparray
            For each cell-body center, the distance to nearest neighboring cell

        """
        min_cell_interdist = np.zeros(self.POPULATION_SIZE)

        for i in range(self.POPULATION_SIZE):
            cell_interdist = np.sqrt((x[i] - x)**2
                                     + (y[i] - y)**2
                                     + (z[i] - z)**2)
            cell_interdist[i] = np.inf
            min_cell_interdist[i] = cell_interdist.min()

        return min_cell_interdist




    # def draw_rand_pos(self, radius, z_min, z_max,
    #                   min_r=np.array([0]), min_cell_interdist=10.,
    #                   **args):
    #     """
    #     Draw some random location within radius, z_min, z_max,
    #     and constrained by min_r and the minimum cell interdistance.
    #     Returned argument is a list of dicts with keys ['x', 'y', 'z'].


    #     Parameters
    #     ----------
    #     radius : float
    #         Radius of population.
    #     z_min : float
    #         Lower z-boundary of population.
    #     z_max : float
    #         Upper z-boundary of population.
    #     min_r : numpy.ndarray
    #         Minimum distance to center axis as function of z.
    #     min_cell_interdist : float
    #         Minimum cell to cell interdistance.
    #     **args : keyword arguments
    #         Additional inputs that is being ignored.


    #     Returns
    #     -------
    #     soma_pos : list
    #         List of dicts of len population size
    #         where dict have keys x, y, z specifying
    #         xyz-coordinates of cell at list entry `i`.


    #     See also
    #     --------
    #     PopulationSuper.calc_min_cell_interdist

    #     """
    #     x = (np.random.rand(self.POPULATION_SIZE) - 0.5) * radius * 2
    #     y = (np.random.rand(self.POPULATION_SIZE) - 0.5) * radius * 2
    #     z = np.random.rand(self.POPULATION_SIZE) * (z_max - z_min) + z_min
    #     min_r_z = {}
    #     min_r = np.array(min_r)
    #     if min_r.size > 0:
    #         if isinstance(min_r, type(np.array([]))):
    #             j = 0
    #             for j in range(min_r.shape[0]):
    #                 min_r_z[j] = np.interp(z, min_r[0, ], min_r[1, ])
    #                 if j > 0:
    #                     [w] = np.where(min_r_z[j] < min_r_z[j - 1])
    #                     min_r_z[j][w] = min_r_z[j - 1][w]
    #             minrz = min_r_z[j]
    #     else:
    #         minrz = np.interp(z, min_r[0], min_r[1])

    #     R_z = np.sqrt(x**2 + y**2)

    #     # want to make sure that no somas are in the same place.
    #     cell_interdist = self.calc_min_cell_interdist(x, y, z)

    #     [u] = np.where(np.logical_or((R_z < minrz) != (R_z > radius),
    #                                  cell_interdist < min_cell_interdist))

    #     while len(u) > 0:
    #         for i in range(len(u)):
    #             x[u[i]] = (np.random.rand() - 0.5) * radius * 2
    #             y[u[i]] = (np.random.rand() - 0.5) * radius * 2
    #             z[u[i]] = np.random.rand() * (z_max - z_min) + z_min
    #             if isinstance(min_r, type(())):
    #                 for j in range(np.shape(min_r)[0]):
    #                     min_r_z[j][u[i]] = \
    #                         np.interp(z[u[i]], min_r[0, ], min_r[1, ])
    #                     if j > 0:
    #                         [w] = np.where(min_r_z[j] < min_r_z[j - 1])
    #                         min_r_z[j][w] = min_r_z[j - 1][w]
    #                     minrz = min_r_z[j]
    #             else:
    #                 minrz[u[i]] = np.interp(z[u[i]], min_r[0, ], min_r[1, ])
    #         R_z = np.sqrt(x**2 + y**2)

    #         # want to make sure that no somas are in the same place.
    #         cell_interdist = self.calc_min_cell_interdist(x, y, z)

    #         [u] = np.where(np.logical_or((R_z < minrz) != (R_z > radius),
    #                                      cell_interdist < min_cell_interdist))

    #     soma_pos = []
    #     for i in range(self.POPULATION_SIZE):
    #         soma_pos.append({'x': x[i], 'y': y[i], 'z': z[i]})

    #     return soma_pos




    def calc_signal_sum(self, measure='LFP'):
        """
        Superimpose each cell's contribution to the compound population signal,
        i.e., the population CSD or LFP or some other lfpykit.<instance>

        Parameters
        ----------
        measure : str

        Returns
        -------
        numpy.ndarray
            The populations-specific compound signal.

        """
        # broadcast output shape from RANK 0 data which is guarantied to exist
        if RANK == 0:
            shape = self.output[measure].shape
        else:
            shape = None
        shape = COMM.bcast(shape, root=0)

        # compute the total LFP of cells on this RANK
        if self.RANK_CELLINDICES.size > 0:
            data = self.output[measure]
        else:
            data = np.zeros(shape, dtype=np.float64)

        # container for full LFP on RANK 0
        if RANK == 0:
            DATA = np.zeros_like(data, dtype=np.float64)
        else:
            DATA = None

        # sum to RANK 0 using automatic type discovery with MPI
        COMM.Reduce(data, DATA, op=MPI.SUM, root=0)

        return DATA

    def collectSingleContribs(self, measure='LFP'):
        """
        Collect single cell data and save them to HDF5 file.
        The function will also return signals generated by all cells


        Parameters
        ----------
        measure : str
            Either 'LFP', 'CSD' or 'current_dipole_moment'


        Returns
        -------
        numpy.ndarray
            output of all neurons in population, axis 0 correspond to neuron
            index

        """
        try:
            assert(self.recordSingleContribFrac <= 1 and
                   self.recordSingleContribFrac >= 0)
        except AssertionError:
            raise AssertionError(
                'recordSingleContribFrac {} not in [0, 1]'.format(
                    self.recordSingleContribFrac))

        if not self.recordSingleContribFrac:
            return
        else:
            # reconstruct RANK_CELLINDICES of all RANKs for controlling
            # communication
            if self.recordSingleContribFrac == 1.:
                SAMPLESIZE = self.POPULATION_SIZE
                RANK_CELLINDICES = []
                for i in range(SIZE):
                    RANK_CELLINDICES += [self.CELLINDICES[
                        self.CELLINDICES % SIZE == i]]
            else:
                SAMPLESIZE = int(self.recordSingleContribFrac *
                                 self.POPULATION_SIZE)
                RANK_CELLINDICES = []
                for i in range(SIZE):
                    ids = self.CELLINDICES[self.CELLINDICES % SIZE == i]
                    RANK_CELLINDICES += [ids[ids < SAMPLESIZE]]

            # gather data on this RANK
            if RANK_CELLINDICES[RANK].size > 0:
                for i, cellindex in enumerate(RANK_CELLINDICES[RANK]):
                    if i == 0:
                        data_temp = np.zeros((RANK_CELLINDICES[RANK].size, ) +
                                             self.output[cellindex
                                                         ][measure].shape,
                                             dtype=np.float32)
                    data_temp[i, ] = self.output[cellindex][measure]

            if RANK == 0:
                # container of all output
                data = np.zeros((SAMPLESIZE, ) +
                                self.output[cellindex][measure].shape,
                                dtype=np.float32)

                # fill in values from this RANK
                if RANK_CELLINDICES[0].size > 0:
                    for j, k in enumerate(RANK_CELLINDICES[0]):
                        data[k, ] = data_temp[j, ]

                # iterate over all other RANKs
                for i in range(1, len(RANK_CELLINDICES)):
                    if RANK_CELLINDICES[i].size > 0:
                        # receive on RANK 0 from all other RANK
                        data_temp = np.zeros((RANK_CELLINDICES[i].size, ) +
                                             self.output[cellindex
                                                         ][measure].shape,
                                             dtype=np.float32)
                        COMM.Recv([data_temp, MPI.FLOAT], source=i, tag=13)

                        # fill in values
                        for j, k in enumerate(RANK_CELLINDICES[i]):
                            data[k, ] = data_temp[j, ]
            else:
                data = None
                if RANK_CELLINDICES[RANK].size > 0:
                    # send to RANK 0
                    COMM.Send([data_temp, MPI.FLOAT], dest=0, tag=13)

            if RANK == 0:
                # save all single-cell data to file
                fname = os.path.join(self.populations_path,
                                     '%s_%ss.h5' % (self.Pop, measure))
                f = h5py.File(fname, 'w')
                f.create_dataset('data', data=data, compression=4)
                f['srate'] = 1E3 / self.dt_output
                f.close()
                assert(os.path.isfile(fname))

                print('file %s_%ss.h5 ok' % (self.Pop, measure))

            COMM.Barrier()

            return data

    def collect_savelist(self):
        '''collect cell attribute data to RANK 0 before dumping data to file'''
        if RANK == 0:
            f = h5py.File(os.path.join(self.populations_path,
                                       '{}_savelist.h5'.format(self.Pop)), 'w')
        for measure in self.savelist:
            if self.RANK_CELLINDICES.size > 0:
                shape = (self.POPULATION_SIZE,
                         ) + np.shape(
                    self.output[self.RANK_CELLINDICES[0]][measure])
                data = np.zeros(shape)
                for ind in self.RANK_CELLINDICES:
                    data[ind] = self.output[ind][measure]
            else:
                data = None

            # sum data arrays to RANK 0
            if RANK == 0:
                DATA = np.zeros_like(data)
            else:
                DATA = None
            COMM.Reduce(data, DATA, op=MPI.SUM, root=0)

            if RANK == 0:
                f[measure] = DATA

        if RANK == 0:
            f.close()

    def collect_data(self):
        """
        Collect LFPs, CSDs and soma traces from each simulated population,
        and save to file.


        Parameters
        ----------
        None


        Returns
        -------
        None

        """
        # collect single-cell attributes as defined in `savelist` and write
        # to files
        self.collect_savelist()

        # sum up single-cell predictions per probe and save
        for probe in self.probes:
            measure = probe.__class__.__name__
            data = self.calc_signal_sum(measure=measure)

            if RANK == 0:
                fname = os.path.join(self.populations_path,
                                     self.output_file.format(self.Pop,
                                                             measure) + '.h5')
                f = h5py.File(fname, 'w')
                f['srate'] = 1E3 / self.dt_output
                f.create_dataset('data', data=data, compression=4)
                f.close()
                print('save {} ok'.format(measure))

        if RANK == 0:
            # save the somatic placements:
            pop_soma_pos = np.zeros((self.POPULATION_SIZE, 3))
            keys = ['x', 'y', 'z']
            for i in range(self.POPULATION_SIZE):
                for j in range(3):
                    pop_soma_pos[i, j] = self.pop_soma_pos[i][keys[j]]
            fname = os.path.join(
                self.populations_path,
                self.output_file.format(
                    self.Pop,
                    'somapos.gdf'))
            np.savetxt(fname, pop_soma_pos)
            assert(os.path.isfile(fname))
            print('save somapos ok')


        # Save ImplementedMorphs
        
        # 1. Gather the local dictionaries from all ranks to Rank 0
        local_mapping = self.ImplementedMorph.get(self.Pop, {})
        gathered_mappings = COMM.gather(local_mapping, root=0)

        if RANK == 0:
            # 2. Merge all gathered dictionaries into one complete mapping
            full_mapping = {}
            for mapping in gathered_mappings:
                full_mapping.update(mapping)
                
            # 3. Create a NumPy array with shape (number of entries, 2)
            num_entries = len(full_mapping)
            pop_morph_data = np.zeros((num_entries, 2), dtype=int)
            
            # 4. Populate the array with SORTED cell indices
            for i, cell_idx in enumerate(sorted(full_mapping.keys())):
                pop_morph_data[i, 0] = cell_idx
                pop_morph_data[i, 1] = full_mapping[cell_idx]
                
            # 5. Construct the file path dynamically
            fname = os.path.join(
                self.populations_path,
                self.output_file.format(
                    self.Pop,
                    f'{self.Pop}_morphologies.gdf' 
                )
            )
            
            # 6. Save the array to text
            np.savetxt(fname, pop_morph_data, fmt='%d')
            assert(os.path.isfile(fname))
            print('save ImplementedMorph ok')






        if RANK == 0:
            # save rotations using hdf5
            fname = os.path.join(
                self.populations_path,
                self.output_file.format(
                    self.Pop,
                    'rotations.h5'))
            f = h5py.File(fname, 'w')
            f.create_dataset('x', (len(self.rotations),))
            f.create_dataset('y', (len(self.rotations),))
            f.create_dataset('z', (len(self.rotations),))

            for i, rot in enumerate(self.rotations):
                for key, value in list(rot.items()):
                    f[key][i] = value
            f.close()
            assert(os.path.isfile(fname))
            print('save rotations ok')

        # collect cell attributes in self.savelist
        for attr in self.savelist:
            self.collectSingleContribs(attr)

        # resync threads
        COMM.Barrier()


class Population(PopulationSuper):
    """
    Class `hybridLFPy.Population`, inherited from class `PopulationSuper`.

    This class rely on spiking times recorded in a network simulation,
    layer-resolved indegrees, synapse parameters, delay parameters, all per
    presynaptic population.

    Population inherits all the layout and data-saving abilities from PopulationSuper,
      but adds the complex machinery required to wire the cells up to the NEST point-neuron simulation.


    Parameters
    ----------
    X : list of str
        Each element denote name of presynaptic populations.
    networkSim : `hybridLFPy.cachednetworks.CachedNetwork` object
        Container of network spike events resolved per population
    k_yXL : numpy.ndarray
        Num layers x num presynapse populations array specifying the
        number of incoming connections per layer and per population type.
    synParams : dict of dicts
        Synapse parameters (cf. `LFPy.Synapse` class).
        Each toplevel key denote each presynaptic population,
        bottom-level dicts are parameters passed to `LFPy.Synapse`.
    synDelayLoc : list
        Average synapse delay for each presynapse connection.
    synDelayScale : list
        Synapse delay std for each presynapse connection.
    J_yX : list of floats
        Synapse weights for connections of each presynaptic population, see
        class `LFPy.Synapse`


    Returns
    -------
    `hybridLFPy.population.Population` object


    See also
    --------
    PopulationSuper, CachedNetwork, CachedFixedSpikesNetwork,
    CachedNoiseNetwork, LFPy.Cell, LFPy.RecExtElectrode
    """

    def __init__(self,
                 X=['EX', 'IN'],
                 networkSim='hybridLFPy.cachednetworks.CachedNetwork',
                 k_yXL=[[20, 0], [20, 10]],
                 synParams={
                     'EX': {
                         'section': ['apic', 'dend'],
                         'syntype': 'AlphaISyn',
                         # 'tau': [0.5, 0.5]
                     },
                     'IN': {
                         'section': ['dend'],
                         'syntype': 'AlphaISyn',
                         # 'tau': [0.5, 0.5],
                     },
                 },
                 synDelayLoc=[1.5, 1.5],
                 synDelayScale=[None, None],
                 J_yX=[0.20680155243678455, -1.2408093146207075],
                 tau_yX=[0.5, 0.5],
                 **kwargs):
        """
        Class `hybridLFPy.Population`, inherited from class `PopulationSuper`.

        This class rely on spiking times recorded in a network simulation,
        layer-resolved indegrees, synapse parameters, delay parameters, all per
        presynaptic population. It inherents also the run() function.


        Parameters
        ----------
        X : list of str
            Each element denote name of presynaptic populations.
        networkSim : `hybridLFPy.cachednetworks.CachedNetwork` object
            Container of network spike events resolved per population
        k_yXL : numpy.ndarray
            Num layers x num presynapse populations array specifying the
            number of incoming connections per layer and per population type.
        synParams : dict of dicts
            Synapse parameters (cf. `LFPy.Synapse` class).
            Each toplevel key denote each presynaptic population,
            bottom-level dicts are parameters passed to `LFPy.Synapse` however
            time constants `tau' takes one value per presynaptic population.
        synDelayLoc : list
            Average synapse delay for each presynapse connection.
        synDelayScale : list
            Synapse delay std for each presynapse connection.
        J_yX : list of floats
            Synapse weights for connections from each presynaptic population,
            see class `LFPy.Synapse`
        tau_yX : list of floats
            Synapse time constants for connections from each presynaptic
            population


        Returns
        -------
        `hybridLFPy.population.Population` object


        See also
        --------
        PopulationSuper, CachedNetwork, CachedFixedSpikesNetwork,
        CachedNoiseNetwork, LFPy.Cell, LFPy.RecExtElectrode
        """
        tic = time()

        PopulationSuper.__init__(self, **kwargs)
        # set some class attributes
        self.X = X
        self.networkSim = networkSim
        # local copy of synapse parameters
        self.synParams = synParams
        

        
        


        if RANK == 0:
            print("population initialized in %.2f seconds" % (time() - tic))

    @property
    def get_ImplementedMorph(self):

        ''' 
        Utility function used to output the choosen morphologies as well as their location
        '''


        return self.ImplementedMorph
    
    @property
    def get_PlacedSynapses(self):
        return self.PlacedSynapses






    def cellsim(self, cellindex):
            ###CHANGED
        """

        

        Do the actual simulations of LFP, using synaptic spike times from
        network simulation.


        Parameters
        ----------
        cellindex : int
            cell index between 0 and population size-1 Of the population iteratively generated in the main code and specific per layer
        return_just_cell : bool
            If True, return only the `LFPy.Cell` object
            if False, run full simulation, return None.


        Returns
        -------
        None or `LFPy.Cell` object


        See also
        --------
        hybridLFPy.csd, LFPy.Cell, LFPy.Synapse, LFPy.RecExtElectrode
        """
        tic = time()


        # 1. Grab the raw network ID for this specific cell
        raw_network_id = self.local_to_raw_map[cellindex]

        ## Morphology 
        Availale_morph = self.Subpop.copy()

        


        ## Load the variables of the main sub-function
        grid_extent = self.grid_extent 
        voxel_size = self.voxel_size 
        mtype_fast_lookup = self.mtype_fast_lookup
        cell_mtypes = self.cell_mtypes
        synapses_dir = self.synapse_base_path
        density_maps_dir = self.SpanTree_path
        layer_limits = self.input_dict['Layers']
        global_cell_coords = self.global_cell_coords
        pre_partners_matrix = self.Cell_afferences[cellindex]
        post_soma_pos  = self.post_soma_pos[cellindex]
        post_mtype_load = f"population_probability_{self.Pop}"





        Load_idx, nid, mapped_synapses_dict = evaluate_select_place(
                                                post_cell_index=cellindex,
                                                post_mtype = post_mtype_load,
                                                post_soma_pos = post_soma_pos,
                                                pre_partners_matrix = pre_partners_matrix,
                                                cell_coords = global_cell_coords,
                                                layer_bounds = layer_limits,
                                                density_maps_dir = density_maps_dir,
                                                morph_paths = Availale_morph,
                                                synapses_dir=synapses_dir,
                                                cell_mtypes=cell_mtypes,
                                                mtype_fast_lookup=mtype_fast_lookup,
                                                voxel_size=voxel_size,
                                                grid_extent=grid_extent
                                            )
        
        if Load_idx is None:
            print(f"⚠️ Cell {cellindex} aborted: No valid morphology found.")
            return None
        # Update the dictionary, best to handle parallelization via MPI
        # Update dictionary to store the Raw Network ID alongside the Morphology ID
        self.ImplementedMorph[self.Pop][cellindex] = [raw_network_id, Load_idx, nid]
        self.PlacedSynapses[cellindex] = mapped_synapses_dict



        # Change the cellParams' field, after copy.
        current_params = self.cellParams.copy()
        current_params['morphology'] = Availale_morph[Load_idx]






        ### Here we create the cell object: initiates the morphology, set its position and rotation, and insert synapses.
        ### Maybe it would be possible to change the cellParams. for example add to the 'morphology' field the specific path to the cell's morphology file.
        ### We need to load the cell-specific morphology file.
        cell = LFPy.Cell(**current_params)
        cell.set_pos(**self.pop_soma_pos[cellindex])
        cell.set_rotation(**self.rotations[cellindex])

        

        ### TODO THE SYNAPSE ARE ALREADY PLACED AT THE FIRST INSTANCE.
        self.insert_all_mapped_synapses(cellindex, cell)

        # set LFPykit.models instance cell attribute
        for probe in self.probes:
            probe.cell = cell

        if 'rec_imem' in self.simulationParams.keys():
            try:
                assert self.simulationParams['rec_imem']
                cell.simulate(**self.simulationParams)
                for probe in self.probes:
                    M = probe.get_transformation_matrix()
                    probe.data = M @ cell.imem
                del cell.imem
            except AssertionError:
                cell.simulate(probes=self.probes, **self.simulationParams)
        else:
            cell.simulate(probes=self.probes, **self.simulationParams)

        # downsample probe.data attribute and unset cell
        for probe in self.probes:
            probe.data = ss.decimate(probe.data,
                                        q=self.decimatefrac)
            probe.cell = None

        # put all necessary cell output in output dict
        for attrbt in self.savelist:
            attr = getattr(cell, attrbt)
            if isinstance(attr, np.ndarray):
                self.output[cellindex][attrbt] = attr.astype('float32')
            else:
                try:
                    self.output[cellindex][attrbt] = attr
                except BaseException:
                    self.output[cellindex][attrbt] = str(attr)
            self.output[cellindex]['srate'] = 1E3 / self.dt_output

        # collect probe output
        for probe in self.probes:
            if cellindex == self.RANK_CELLINDICES[0]:
                self.output[probe.__class__.__name__] = \
                    probe.data.copy()
            else:
                self.output[probe.__class__.__name__] += \
                    probe.data.copy()
            probe.data = None

        # clean up hoc namespace
        cell.__del__()

        print('cell %s population %s in %.2f s' % (cellindex, self.Pop,
                                                    time() - tic))

    def insert_all_mapped_synapses(self, cellindex, cell):
        """
        Insert all synaptic events from all presynaptic layers on
        cell object with index `cellindex`.


        Parameters
        ----------
        cellindex : int
            cell index in the population.
        cell : `LFPy.Cell` instance
            Postsynaptic target cell.


        Returns
        -------
        None


        See also
        --------
        Population.insert_synapse

        """
        ## Load the cellindex-specific pre-synaptic partners dict
        mapped_synapses_dict = self.PlacedSynapses.get(cellindex, {}) # .get() prevents key errors
        if not mapped_synapses_dict:
            return

        for Pre_neuron_idx, lfpy_segments in mapped_synapses_dict.items():

            ### Extract the syaptic weights

            # 1. Deduce the pre-synaptic population database name (X)
            # This allows us to find the spikes in self.networkSim.dbs
            specific_mtype = self.cell_mtypes[Pre_neuron_idx]
            layer, bio_type = self.mtype_fast_lookup[specific_mtype]
            
            if "SS" in specific_mtype.upper():
                type_suffix = "ss"
            else:
                type_suffix = "exc" if bio_type == "Excitatory" else "inh"
                
            pre_pop_name = f"{layer}_{type_suffix}" # Ensure this matches your NEST db keys
            post_pop_name = self.Pop
            
            ## Retrieve the specific weight
            SynW = self.SynWeigths[pre_pop_name][post_pop_name]
            SynT = self.SynTau[pre_pop_name][post_pop_name]


            synParams = self.synParams.copy()
            synParams.update({
                'weight': SynW,
                'tau': SynT,
            })

            if self.synDelays is not None:
                    synDelays = self.CalculateDelay() ## Function to add

            else:
                synDelays = None


            # 3. Delegate to the worker to actually attach the spikes
            self.insert_specific_connection(
                cell=cell,
                pre_idx=Pre_neuron_idx,
                lfpy_indices=lfpy_segments,
                pre_pop_name=pre_pop_name,
                synParams=synParams,
                synDelay=synDelays
            )



                


    def insert_specific_connection(self, cell, pre_idx, lfpy_indices, pre_pop_name=None, synParams=None, synDelay=None):
        """
        Inserts all synapses from a SINGLE pre-synaptic neuron onto the post-synaptic cell.
        Uses weight-scaling to efficiently model multiple synapses on the same segment
        without artificially depleting Short-Term Plasticity vesicle pools.
        """
        if not lfpy_indices:
            return # No synapses to place for this connection

        # 1. Fetch the spike train for this SINGLE pre-synaptic neuron
        try:
            spikes = self.networkSim.dbs[pre_pop_name].select([pre_idx])[0]
        except AttributeError:
            raise AssertionError(f"Could not open CachedNetwork database for {pre_pop_name}")

        # If this pre-synaptic neuron never fired, skip building the synapse
        if len(spikes) == 0:
            return

        # 2. Apply axonal delay
        if synDelay is not None:
            spikes = spikes + synDelay

        # 3. Group by unique dendritic segment
        uidx = np.unique(lfpy_indices)
        
        for i in uidx:
            # Count how many synapses from this pre-neuron land on this exact segment
            synapse_count = list(lfpy_indices).count(i)
            
            # --- STP-SAFE GROUPING TRICK ---
            # We copy the parameters so we don't mutate the global dictionary
            local_synParams = synParams.copy()
            
            # We scale the base NEURON NetCon weight by the number of synapses.
            # This perfectly mimics N synapses without triggering artificial STP depletion.
            base_weight = local_synParams.get('weight', 0.001)
            local_synParams['weight'] = base_weight * synapse_count
            
            # Update the segment index and force the new MOD file mechanism
            local_synParams.update({
                'idx': i,
                'syntype': 'ProbAMPANMDA'
            })
            
            # 4. Instantiate the Synapse
            synapse = LFPy.Synapse(cell, **local_synParams)
            
            # 5. Feed the ORIGINAL (non-tiled) spike train
            st = np.sort(spikes) + cell.tstart
            synapse.set_spike_times(st)
            
            # 6. Reproducible Random Streams (CRITICAL FOR MPI)
            # ProbAMPANMDA.mod requires a pointer to a NEURON random generator for `erand()`
            import neuron
            rng = neuron.h.Random()
            # Generate a unique, reproducible seed based on Population, Cell, and Segment
            # (Assuming you have self.POPULATIONSEED available)
            rng.Random123(self.POPULATIONSEED, cellindex, i) 
            rng.negexp(1)
            
            # Access the underlying NEURON hoc object created by LFPy to set the RNG
            synapse.synapse.setRNG(rng)





class TopoPopulation(Population):
    def __init__(self,
                 topology_connections={
                     'EX': {
                         'edge_wrap': True,
                         'extent': [4000., 4000.],
                         'allow_autapses': True,
                         'kernel': {'exponential': {
                             'a': 1., 'c': 0.0, 'tau': 300.}},
                         'mask': {'circular': {'radius': 2000.}},
                         'delays': {
                             'linear': {
                                 'c': 1.,
                                 'a': 2.
                             }
                         },
                     },
                     'IN': {
                         'edge_wrap': True,
                         'extent': [4000., 4000.],
                         'allow_autapses': True,
                         'kernel': {'exponential': {
                             'a': 1., 'c': 0.0, 'tau': 300.}},
                         'mask': {'circular': {'radius': 2000.}},
                         'delays': {
                             'linear': {
                                 'c': 1.,
                                 'a': 2.
                             }
                         },
                     },
                 },
                 **kwargs):
        '''
        Initialization of class TopoPopulation, for dealing with networks
        created using the NEST topology library (distance dependent
        connectivity).

        Inherited of class hybridLFPy.Population

        Arguments
        ---------
        topology_connections : dict
            nested dictionary with topology-connection parameters for each
            presynaptic population


        Returns
        -------
        object : populations.TopoPopulation
            population object with connections, delays, positions, simulation
            methods

        See also
        --------
        hybridLFPy.Population

        '''
        # set networkSim attribute so that monkey-patched methods can work
        self.networkSim = kwargs['networkSim']
        self.topology_connections = topology_connections

        # initialize parent class
        Population.__init__(self, **kwargs)

        # set class attributes

   

import numpy as np
import plotly.graph_objects as go
from collections import Counter



if __name__ == '__main__':
    import doctest
    doctest.testmod()
