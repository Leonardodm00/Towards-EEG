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
                 y='EX',
                 layerBoundaries=[[0.0, -300], [-300, -500]],

                SubPopulations_list = ['xxxxx.hoc','xxxxx.hoc'],
                layer_names =  ['L1', 'L2/3', 'L4', 'L5', 'L6'],
                Pop_to_Syntype = {'L4':'exc'},
                synapse_base_path = '',

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
            list of the current subpopulation's (self.y) morpholgy paths

        layer_names: list 
            Layer names

        synapse_base_path: string
            path to the folder where the per-neuron synapses csv files are held

    
        Pop_to_Syntype : dict
            The keys are the sub-populations while the values are the synapse types 'exc' or 'inh'


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
        self.POPULATION_SIZE = populationParams['number']
        self.y = y
        self.layerBoundaries = np.array(layerBoundaries)
        self.Subpop = SubPopulations_list
        self.ImplementedMorph = {self.y : {}}
        self.layer_names = layer_names
        self.Pop_to_Syntype = Pop_to_Syntype
        self.synapse_base_path = synapse_base_path
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

        self.pop_soma_pos = self.set_pop_soma_pos()
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

            print('cell %s population %s in %.2f s' % (cellindex, self.y,
                                                       time() - tic))

    def set_pop_soma_pos(self):
        """
        Set `pop_soma_pos` using draw_rand_pos().

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
            pop_soma_pos = self.draw_rand_pos(
                # min_r = self.electrodeParams['r_z'],
                **self.populationParams)
        else:
            pop_soma_pos = None

        if RANK == 0:
            print('found cell positions in %.2f s' % (time() - tic))

        return COMM.bcast(pop_soma_pos, root=0)

    def set_rotations(self):
        """

        Append random z-axis rotations for each cell in population.

        This method takes no keyword arguments


        Parameters
        ----------
        None


        Returns
        -------
        numpyp.ndarray
            Rotation angle around axis `Population.rand_rot_axis` of each
            neuron in the population


        """
        tic = time()
        if RANK == 0:
            rotations = []
            for i in range(self.POPULATION_SIZE):
                defaultrot = {}
                for axis in self.rand_rot_axis:
                    defaultrot.update({axis: np.random.rand() * 2 * np.pi})
                rotations.append(defaultrot)
        else:
            rotations = None

        if RANK == 0:
            print('found cell rotations in %.2f s' % (time() - tic))

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

    def draw_rand_pos(self, radius, z_min, z_max,
                      min_r=np.array([0]), min_cell_interdist=10.,
                      **args):
        """
        Draw some random location within radius, z_min, z_max,
        and constrained by min_r and the minimum cell interdistance.
        Returned argument is a list of dicts with keys ['x', 'y', 'z'].


        Parameters
        ----------
        radius : float
            Radius of population.
        z_min : float
            Lower z-boundary of population.
        z_max : float
            Upper z-boundary of population.
        min_r : numpy.ndarray
            Minimum distance to center axis as function of z.
        min_cell_interdist : float
            Minimum cell to cell interdistance.
        **args : keyword arguments
            Additional inputs that is being ignored.


        Returns
        -------
        soma_pos : list
            List of dicts of len population size
            where dict have keys x, y, z specifying
            xyz-coordinates of cell at list entry `i`.


        See also
        --------
        PopulationSuper.calc_min_cell_interdist

        """
        x = (np.random.rand(self.POPULATION_SIZE) - 0.5) * radius * 2
        y = (np.random.rand(self.POPULATION_SIZE) - 0.5) * radius * 2
        z = np.random.rand(self.POPULATION_SIZE) * (z_max - z_min) + z_min
        min_r_z = {}
        min_r = np.array(min_r)
        if min_r.size > 0:
            if isinstance(min_r, type(np.array([]))):
                j = 0
                for j in range(min_r.shape[0]):
                    min_r_z[j] = np.interp(z, min_r[0, ], min_r[1, ])
                    if j > 0:
                        [w] = np.where(min_r_z[j] < min_r_z[j - 1])
                        min_r_z[j][w] = min_r_z[j - 1][w]
                minrz = min_r_z[j]
        else:
            minrz = np.interp(z, min_r[0], min_r[1])

        R_z = np.sqrt(x**2 + y**2)

        # want to make sure that no somas are in the same place.
        cell_interdist = self.calc_min_cell_interdist(x, y, z)

        [u] = np.where(np.logical_or((R_z < minrz) != (R_z > radius),
                                     cell_interdist < min_cell_interdist))

        while len(u) > 0:
            for i in range(len(u)):
                x[u[i]] = (np.random.rand() - 0.5) * radius * 2
                y[u[i]] = (np.random.rand() - 0.5) * radius * 2
                z[u[i]] = np.random.rand() * (z_max - z_min) + z_min
                if isinstance(min_r, type(())):
                    for j in range(np.shape(min_r)[0]):
                        min_r_z[j][u[i]] = \
                            np.interp(z[u[i]], min_r[0, ], min_r[1, ])
                        if j > 0:
                            [w] = np.where(min_r_z[j] < min_r_z[j - 1])
                            min_r_z[j][w] = min_r_z[j - 1][w]
                        minrz = min_r_z[j]
                else:
                    minrz[u[i]] = np.interp(z[u[i]], min_r[0, ], min_r[1, ])
            R_z = np.sqrt(x**2 + y**2)

            # want to make sure that no somas are in the same place.
            cell_interdist = self.calc_min_cell_interdist(x, y, z)

            [u] = np.where(np.logical_or((R_z < minrz) != (R_z > radius),
                                         cell_interdist < min_cell_interdist))

        soma_pos = []
        for i in range(self.POPULATION_SIZE):
            soma_pos.append({'x': x[i], 'y': y[i], 'z': z[i]})

        return soma_pos

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
                                     '%s_%ss.h5' % (self.y, measure))
                f = h5py.File(fname, 'w')
                f.create_dataset('data', data=data, compression=4)
                f['srate'] = 1E3 / self.dt_output
                f.close()
                assert(os.path.isfile(fname))

                print('file %s_%ss.h5 ok' % (self.y, measure))

            COMM.Barrier()

            return data

    def collect_savelist(self):
        '''collect cell attribute data to RANK 0 before dumping data to file'''
        if RANK == 0:
            f = h5py.File(os.path.join(self.populations_path,
                                       '{}_savelist.h5'.format(self.y)), 'w')
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
                                     self.output_file.format(self.y,
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
                    self.y,
                    'somapos.gdf'))
            np.savetxt(fname, pop_soma_pos)
            assert(os.path.isfile(fname))
            print('save somapos ok')


        # Save ImplementedMorphs
        
        # 1. Gather the local dictionaries from all ranks to Rank 0
        local_mapping = self.ImplementedMorph.get(self.y, {})
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
                    self.y,
                    f'{self.y}_morphologies.gdf' 
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
                    self.y,
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
        self.k_yXL = np.array(k_yXL)

        # local copy of synapse parameters
        self.synParams = synParams
        self.synDelayLoc = synDelayLoc
        self.synDelayScale = synDelayScale
        self.J_yX = J_yX
        self.tau_yX = tau_yX

        # Now loop over all cells in the population and assess
        # - number of synapses in each z-interval (from layerbounds)
        # - placement of synapses in each z-interval

        # get in this order, the
        # - postsynaptic compartment indices
        # - presynaptic cell indices
        # - synapse delays per connection
        self.synIdx = self.get_all_synIdx()
        self.SpCells = self.get_all_SpCells()
        self.synDelays = self.get_all_synDelays()

        if RANK == 0:
            print("population initialized in %.2f seconds" % (time() - tic))


    def get_ImplementedMorph(self):

        ''' 
        Utility function used to output the choosen morphologies as well as their location
        '''


        return self.ImplementedMorph



    def get_all_synIdx(self):
        """
        Auxilliary function to set up class attributes containing
        synapse locations given as LFPy.Cell compartment indices

        This function takes no inputs.

        


        Parameters
        ----------
        None


        Returns
        -------
        synIdx : dict
            `output[cellindex][populationindex][layerindex]` numpy.ndarray of
            compartment indices.


        See also
        --------
        Population.get_synidx, Population.fetchSynIdxCell
        """
        tic = time()

        # containers for synapse idxs existing on this rank
        # The final form is: self.synIdx[cellindex][presynaptic_population][layer_index]
        synIdx = {}

        # ok then, we will draw random numbers across ranks, which have to
        # be unique per cell. Now, we simply record the random state,
        # change the seed per cell, and put the original state back below.
        randomstate = np.random.get_state()

        for cellindex in self.RANK_CELLINDICES:
            # set the random seed on for each cellindex
            np.random.seed(self.POPULATIONSEED + cellindex)

            # find synapse locations for cell in parallel
            synIdx[cellindex] = self.get_synidx(cellindex)

        # reset the random number generator
        np.random.set_state(randomstate)

        if RANK == 0:
            print('found synapse locations in %.2f seconds' % (time() - tic))

        # print the number of synapses per layer from which presynapse
        # population
        if self.verbose:
            for cellindex in self.RANK_CELLINDICES:
                for i, synidx in enumerate(synIdx[cellindex]):
                    print(
                        'to:\t%s\tcell:\t%i\tfrom:\t%s:' %
                        (self.y, cellindex, self.X[i]),)
                    idxcount = 0
                    for idx in synidx:
                        idxcount += idx.size
                        print('\t%i' % idx.size,)
                    print('\ttotal %i' % idxcount)

        return synIdx

    def get_all_SpCells(self):
        """
        For each postsynaptic cell existing on this RANK, load or compute
        the presynaptic cell index for each synaptic connection

        This function takes no kwargs.


        Parameters
        ----------
        None


        Returns
        -------
        SpCells : dict
            `output[cellindex][populationname][layerindex]`, np.array of
            presynaptic cell indices.


        See also
        --------
        Population.fetchSpCells

        """
        tic = time()

        # container
        SpCells = {}

        # ok then, we will draw random numbers across ranks, which have to
        # be unique per cell. Now, we simply record the random state,
        # change the seed per cell, and put the original state back below.
        randomstate = np.random.get_state()

        for cellindex in self.RANK_CELLINDICES:
            # set the random seed on for each cellindex
            np.random.seed(
                self.POPULATIONSEED +
                cellindex +
                self.POPULATION_SIZE)

            SpCells[cellindex] = {}
            for i, X in enumerate(self.X):
                SpCells[cellindex][X] = self.fetchSpCells(
                    self.networkSim.nodes[X], self.k_yXL[:, i])

        # reset the random number generator
        np.random.set_state(randomstate)

        if RANK == 0:
            print('found presynaptic cells in %.2f seconds' % (time() - tic))

        return SpCells

    def get_all_synDelays(self):
        """
        Create and load arrays of connection delays per connection on this rank

        Get random normally distributed synaptic delays,
        returns dict of nested list of same shape as SpCells.

        Delays are rounded to dt.

        This function takes no kwargs.


        Parameters
        ----------
        None


        Returns
        -------
        dict
            output[cellindex][populationname][layerindex]`, np.array of
            delays per connection.


        See also
        --------
        numpy.random.normal

        """
        tic = time()

        # ok then, we will draw random numbers across ranks, which have to
        # be unique per cell. Now, we simply record the random state,
        # change the seed per cell, and put the original state back below.
        randomstate = np.random.get_state()

        # container
        delays = {}

        for cellindex in self.RANK_CELLINDICES:
            # set the random seed on for each cellindex
            np.random.seed(
                self.POPULATIONSEED +
                cellindex +
                2 *
                self.POPULATION_SIZE)

            delays[cellindex] = {}
            for j, X in enumerate(self.X):
                delays[cellindex][X] = []
                for i in self.k_yXL[:, j]:
                    loc = self.synDelayLoc[j]
                    loc /= self.dt
                    scale = self.synDelayScale[j]
                    if scale is not None:
                        scale /= self.dt
                        delay = np.random.normal(loc, scale, i).astype(int)
                        while np.any(delay < 1):
                            inds = delay < 1
                            delay[inds] = np.random.normal(
                                loc, scale, inds.sum()).astype(int)
                        delay = delay.astype(float)
                        delay *= self.dt
                    else:
                        delay = np.zeros(i) + self.synDelayLoc[j]
                    delays[cellindex][X].append(delay)

        # reset the random number generator
        np.random.set_state(randomstate)

        if RANK == 0:
            print('found delays in %.2f seconds' % (time() - tic))

        return delays

    def get_synidx(self, cellindex):
        """
        Local function, draw and return synapse locations corresponding
        to a single cell, using a random seed set as
        `POPULATIONSEED` + `cellindex`.

        The get_synidx method in the population.py script serves as the
        dedicated function to determine exactly where synapses should be physically placed
        on the 3D morphology of a single, specific cell.

        ###TOSEE: understand how the synapses are set according to the position of the arbour inthe specific layer.



        Parameters
        ----------
        cellindex : int
            Index of cell object.


        Returns
        -------
        synidx : dict
            `LFPy.Cell` compartment indices


        See also
        --------
        Population.get_all_synIdx, Population.fetchSynIdxCell

        """
        # create a cell instance


        ## PROVIDE THE  nidx=self.k_yXL[:, i] TO CHECK THE MORPHOLOGY
        ## Emulate the for i, X in enumerate(self.X): to check for arbour compliance.

      
        ###CHANGED
        cell,neuron_id = self.cellsim(cellindex, return_just_cell=True) # Load the cell



        # Download the csv file with the mapped incoming synapses
        base_SynPath = self.synapse_base_path
        csv_file_path = os.path.join(base_SynPath, f"neuron_{neuron_id}_mapped_synapses.csv")

        if not os.path.exists(csv_file_path):
            print(f"⚠️ Failed to find synapse CSV: {csv_file_path}")
            return None
        syn_df = pd.read_csv(csv_file_path)
        # Retrieve the absolute Z placement of the cell
        soma_z = self.pop_soma_pos[cellindex]['z']
        # Translate the entire column ONCE
        syn_df['z'] = syn_df['z'] + soma_z






        # local containers
        synidx = {}

        # get synaptic placements and cells from the network,
        # then set spike times,
        for i, X in enumerate(self.X):  # Loops through all the sub-populations.
            synidx[X] = self.fetchSynIdxCell(syn_df=syn_df,
                                             nidx=self.k_yXL[:, i],
                                             Prepop = X)
        # clean up hoc namespace
        cell.__del__()

        return synidx
    




    def fetch_mapped_synapse_indices(self,syn_df, syn_type, z_min=-np.inf, z_max=np.inf):
        if syn_type not in ['exc', 'inh']:
            raise ValueError(f"Invalid syn_type '{syn_type}'. Must be 'exc' or 'inh'.")

        type_mask = syn_df['synapse_type'] == syn_type
        depth_mask = (syn_df['z'] >= z_min) & (syn_df['z'] <= z_max)
        
        filtered_df = syn_df[type_mask & depth_mask]
        syn_indices = filtered_df['lfpy_idx'].values.astype(int)
        
        return syn_indices


    ############################## FINISH HERE TO CHANGEEEEEEEEE

    def fetchSynIdxCell(self, syn_df, nidx,Prepop):
        """
        Find possible synaptic placements for each cell
        As synapses are placed within layers with bounds determined by
        self.layerBoundaries, it will check this matrix accordingly, and
        use the probabilities from `self.connProbLayer to distribute.

        For each layer, the synapses are placed with probability normalized
        by membrane area of each compartment


        Parameters
        ----------
        cell : `LFPy.Cell` instance
        nidx : numpy.ndarray
            Numbers of synapses per presynaptic population X.
        synParams : which `LFPy.Synapse` parameters to use.
        Prepop : The current pre-synaptic population. This is needed to set the correct type of synapses 
                that is established


        Returns
        -------
        syn_idx : list
            List of arrays of synapse placements per connection.


        See also
        --------
        Population.get_all_synIdx, Population.get_synIdx, LFPy.Synapse

        """
        # Retrieve the type of presynaptic neuron to establish the synapse type: whether exc or inh
        Synapse_type = self.Pop_to_Syntype[Prepop]
        
        


        # segment indices in each layer is stored here, list of np.array
        syn_idx = []
        # loop over layer bounds, find synapse locations
        for i, zz in enumerate(self.layerBoundaries):
            if nidx[i] == 0:
                syn_idx.append(np.array([], dtype=int))
            else:

                syn_indicies = self.fetch_mapped_synapse_indices(syn_df, 
                                                                Synapse_type, 
                                                                z_min=zz.min(),
                                                                z_max=zz.max()).astype(np.int16)
                # Randomly sample nidx synapses
                # Initialize the generator
                rng = np.random.default_rng()
                # Sample nidx elements without replacement
                syn_idx.append(rng.choice(syn_indicies, size=nidx[i], replace=True))
                

        return syn_idx
    





    def _plot_capacity_at_z(self,dend_segments, syn_df, layer_boundaries, layer_names, valid_layers, target_z_pos, morph_idx):
        """Helper function to visualize the selected morphology and its synapse capacity."""
        fig, ax = plt.subplots(figsize=(10, 12))
        
        color_palette = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4']
        layer_colors = {layer_idx: color_palette[i % len(color_palette)] for i, layer_idx in enumerate(valid_layers)}
        
        # 1. Plot layer boundaries
        for i, bounds in enumerate(layer_boundaries):
            upper_bound, lower_bound = bounds
            ax.axhline(upper_bound, color='black', linestyle='--', alpha=0.5)
            
            bg_color = layer_colors.get(i, 'lightgrey')
            alpha_val = 0.15 if i in valid_layers else 0.05
            ax.axhspan(lower_bound, upper_bound, facecolor=bg_color, alpha=alpha_val)
            
            layer_label = layer_names[i] if i < len(layer_names) else f"Layer {i}"
            ax.text(-250, (upper_bound + lower_bound) / 2, layer_label, 
                    va='center', ha='right', fontsize=12, fontweight='bold')
        
        ax.axhline(layer_boundaries[-1][1], color='black', linestyle='--', alpha=0.5)

        # 2. Plot dendritic segments (Light Grey)
        lines = []
        for p1, p2 in dend_segments:
            shifted_z1 = p1[2] + target_z_pos
            shifted_z2 = p2[2] + target_z_pos
            lines.append([(p1[0], shifted_z1), (p2[0], shifted_z2)])
            
        lc = LineCollection(lines, colors='darkgrey', linewidths=1.2, alpha=0.7)
        ax.add_collection(lc)
        
        # 3. Plot Mapped Synapses
        shifted_syn_z = syn_df['z'] + target_z_pos
        
        # Plot unused synapses in light grey
        ax.scatter(syn_df['x'], shifted_syn_z, color='lightgrey', s=8, alpha=0.5, label='Unused Synapses', zorder=3)
        
        # Plot valid synapses in their respective layer colors
        for layer_idx in valid_layers:
            upper_bound, lower_bound = layer_boundaries[layer_idx]
            mask = (shifted_syn_z >= lower_bound) & (shifted_syn_z <= upper_bound)
            
            if mask.sum() > 0:
                ax.scatter(syn_df.loc[mask, 'x'], shifted_syn_z[mask], 
                        color=layer_colors[layer_idx], s=25, 
                        label=f'Capacity in {layer_names[layer_idx]} ({mask.sum()})', zorder=4)

        # 4. Plot the soma
        ax.scatter([0], [target_z_pos], color='black', s=150, zorder=5, label='Soma')
        
        # 5. Final plot formatting
        ax.set_aspect('equal')
        ax.set_xlim(-350, 350) 
        ax.set_ylim(layer_boundaries[-1][1] - 50, layer_boundaries[0][0] + 50)
        ax.set_xlabel('X distance (um)', fontsize=12)
        ax.set_ylabel('Depth / Z distance (um)', fontsize=12)
        ax.set_title(f'Selected Morphology (ID: {morph_idx})\nCapacity Evaluated at Z = {target_z_pos:.1f} um', fontsize=16, fontweight='bold')
        ax.legend(loc='lower left', fontsize=10)
        
        plt.tight_layout()
        plt.show()

    def cell_MorphSelect(self,
        morph_paths,
        layer_boundaries,
        layer_names,          # Added for plotting
        synapses_per_layer,
        target_z_pos,
        synapse_base_path,
        plot_result=False     # Added for debugging hook
        ):
            """
            Selects a valid neuronal morphology by ensuring its pre-mapped connectome 
            has enough physical synapse locations to satisfy the network's requirements 
            at a specific absolute cortical depth.

            Unlike geometric approaches that estimate viability based on dendritic length, 
            this function performs a direct empirical check. It virtually places the 
            neuron's soma at the target depth, shifts its pre-calculated synapse coordinates 
            accordingly, and counts the exact number of available docking sites falling 
            within the required cortical layers.

            Args:
                morph_paths (list of str): 
                    A list of file paths to the candidate `.hoc` morphology files.
                    Example: ['/path/neuron_123_aligned.hoc', '/path/neuron_456_aligned.hoc']
                layer_boundaries (numpy.ndarray): 
                    A 2D array of shape (N, 2) defining the [upper_Z, lower_Z] boundaries 
                    for each cortical layer in micrometers (μm).
                    Example: [[0.0, -81.6], [-81.6, -587.1], ...]
                layer_names (list of str): 
                    Display names for each layer, used strictly for annotating the debug plot.
                    Example: ['L1', 'L2/3', 'L4', 'L5', 'L6']
                synapses_per_layer (list or numpy.ndarray of int): 
                    The number of synapses the network model demands in each layer. 
                    The length must exactly match the number of rows in `layer_boundaries`.
                    Example: [0, 1500, 3200, 500, 0]
                target_z_pos (float): 
                    The absolute Z-coordinate (in μm) where this specific cell's soma 
                    will be placed in the simulated cortical column. 
                synapse_base_path (str): 
                    The directory path containing the mapped synapse CSV files.
                plot_result (bool, optional): 
                    If True, boots up the NEURON simulator to extract the 3D skeleton of 
                    the winning morphology and generates a detailed visual report of the 
                    synapse capacity across layers. Defaults to False.

            Returns:
                tuple (int, int): 
                    - original_idx: The integer index of the winning morphology in the 
                                    original `morph_paths` list.
                    - nid: The extracted integer Neuron ID of the winning morphology.

            Raises:
                RuntimeError: 
                    If the function evaluates every candidate morphology in the list and 
                    none possess the required synapse capacity at the specified `target_z_pos`.
            """
            valid_layers = [i for i, syn in enumerate(synapses_per_layer) if syn > 0]
            shuffled_paths = list(enumerate(morph_paths))
            random.shuffle(shuffled_paths)
            
            for original_idx, path in shuffled_paths:
                print(f"Evaluating Original Index: {original_idx}")
                
                match = re.search(r'neuron_(\d+)', os.path.basename(path))
                nid = int(match.group(1)) if match else -1
                
                csv_path = os.path.join(synapse_base_path, f"neuron_{nid}_mapped_synapses.csv")
                if not os.path.exists(csv_path):
                    print(f"⚠️ Warning: Synapse CSV not found for {nid} at {csv_path}")
                    continue
                    
                syn_df = pd.read_csv(csv_path)
                
                shifted_z = syn_df['z'] + target_z_pos
                is_viable = True
                
                for layer_idx in valid_layers:
                    upper_bound, lower_bound = layer_boundaries[layer_idx]
                    required_synapses = synapses_per_layer[layer_idx]
                    
                    available_synapses = ((shifted_z >= lower_bound) & (shifted_z <= upper_bound)).sum()
                    
                    if available_synapses < required_synapses:
                        is_viable = False
                        break 
                        
                if is_viable:
                    # ---> DEBUG PLOTTING HOOK <---
                    if plot_result:
                        print(f"✅ Winner found ({nid}). Extracting 3D geometry for plot...")
                        h('forall delete_section()')
                        success = h.load_file(str(path))
                        
                        if success:
                            dend_segments = []
                            for sec in h.allsec():
                                if 'dend' in sec.name().lower() or 'apic' in sec.name().lower():
                                    n3d = int(h.n3d(sec=sec))
                                    if n3d > 0:
                                        pts = np.array([[h.x3d(i, sec=sec), h.y3d(i, sec=sec), h.z3d(i, sec=sec)] for i in range(n3d)])
                                        for i in range(n3d - 1):
                                            dend_segments.append((pts[i], pts[i+1]))
                            
                            self._plot_capacity_at_z(
                                dend_segments, syn_df, layer_boundaries, layer_names, 
                                valid_layers, target_z_pos, nid
                            )
                    # ------------------------------
                    return original_idx, nid
                    
            raise RuntimeError(
                f"No morphology found with enough synapse capacity to satisfy "
                f"the requirements: {synapses_per_layer} at Z={target_z_pos:.1f}"
            )





    def cellsim(self, cellindex, return_just_cell=False):
            ###CHANGED
        """
        My notes:
        This function also performs a morphological check based on the synapse distribution across layers. It must
        discard the cells that doesn't have dendritic branches in the layers targeted by the synapses. The total number and 
        poistion of the synapses is retrieved by scanning the k_yXL (already constructed in a populations specific manner) across
        all possible presynaptic macro-populations

        Sum_per_Layer = np.sum(self.k_yXL, axis=1)

        Sum_per_Layer stores the total number of synapses divided per layer. The funciton will evaluate whether the arbour
        has a certain length of dendrites (basal or apical is not important) within the non-zero layers in Sum_per_Layer.

        The variable 'self.Subpop' will have stored all the available morphologies for that specific sub-population which will
        be randomly scanned and in case their arbour comply with the requirements choosen and the self.ImplementedMorph updated
        accordingly. The variable is a list containing the file paths to the candidate .hoc morphology files.
                Example: ['/path/to/neuron_1.hoc', '/path/to/neuron_2.hoc']

        The morphologies MUST be already rotated and the soma must be fixed at the origin. The calculation of the arbour length will be done 
        accounting for the extreme cases, lower and higher depth bound for the placemnt of the soma wihtin the layer of belonging of the neuron.
        Moreover is needed to put a threshold for the minimum branch length wihtin each layer that accomodates for synapses.

  

        The variable self.ImplementedMorph has a similar structure. The top key in the hierarchy refers to the subpopulation, while the value
        stores matrix with self.RANK_CELLINDICES x 1 where the index is the current cell index 'cellindex' while is stored the index of the
        pathway for the loading of the morphology.

        
        
        ---------  return_just_cell == FALSE ---------
        In this case we need to upload the correct morphology based on the sifting done in the previouse call of the same function but with
        return_just_cell == TRUE used only to load the morphology and place the synapses onto it.
        Thus for a specific sub-population we resort to the variable self.ImplementedMorph[self.y] which encompasses the list of indicies
        that locate in the self.Subpop variable. The correct morphology is located by the current 'cellindex' which inherently 
        links the latter with the synapses distributed in the previous call of the function
        
        




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

        ## Morphology 
        Availale_morph = self.Subpop.copy()

        if return_just_cell:

            soma_z = self.pop_soma_pos[cellindex]['z']
            Sum_per_Layer = np.sum(self.k_yXL, axis=1)
            Load_idx,nid = self.cell_MorphSelect(Availale_morph,
                                        self.layerBoundaries,
                                        self.layer_names,
                                        Sum_per_Layer,
                                        target_z_pos=soma_z,
                                        synapse_base_path=self.synapse_base_path,
                                        plot_result=False)
            # Update the dictionary, best to handle parallelization via MPI
            self.ImplementedMorph[self.y][cellindex] = [Load_idx,nid]


        else:
             # Retrieve the morphology choosen in the previous call of the function
             Load_idx,nid = self.ImplementedMorph[self.y][cellindex]





        # Change the cellParams' field, after copy.
        current_params = self.cellParams.copy()
        current_params['morphology'] = Availale_morph[Load_idx]






        ### Here we create the cell object: initiates the morphology, set its position and rotation, and insert synapses.
        ### Maybe it would be possible to change the cellParams. for example add to the 'morphology' field the specific path to the cell's morphology file.
        ### We need to load the cell-specific morphology file.
        cell = LFPy.Cell(**current_params)
        cell.set_pos(**self.pop_soma_pos[cellindex])
        cell.set_rotation(**self.rotations[cellindex])

        if return_just_cell:
            return cell,nid
        else:

            ### TOSEE AFTERWARDS
            self.insert_all_synapses(cellindex, cell)

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

            print('cell %s population %s in %.2f s' % (cellindex, self.y,
                                                       time() - tic))

    def insert_all_synapses(self, cellindex, cell):
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
        for i, X in enumerate(self.X):  # Pre-synaptic population
            synParams = self.synParams
            synParams.update({
                'weight': self.J_yX[i],
                'tau': self.tau_yX[i],
            })
            for j in range(len(self.synIdx[cellindex][X])):  # Loop over the layers j is the layer
                if self.synDelays is not None:
                    synDelays = self.synDelays[cellindex][X][j]
                else:
                    synDelays = None
                self.insert_synapses(cell=cell,
                                     cellindex=cellindex,
                                     synParams=synParams,
                                     idx=self.synIdx[cellindex][X][j],
                                     X=X,
                                     SpCell=self.SpCells[cellindex][X][j],
                                     synDelays=synDelays)

    def insert_synapses(self, cell, cellindex, synParams, idx=np.array([]),
                        X='EX', SpCell=np.array([]),
                        synDelays=None):
        """
        Insert synapse with `parameters`=`synparams` on cell=cell, with
        segment indexes given by `idx`. `SpCell` and `SpTimes` picked from
        Brunel network simulation


        Parameters
        ----------
        cell : `LFPy.Cell` instance
            Postsynaptic target cell.
        cellindex : int
            Index of cell in population.
        synParams : dict
            Parameters passed to `LFPy.Synapse`.
        idx : numpy.ndarray
            Postsynaptic compartment indices.
        X : str
            presynaptic population name
        SpCell : numpy.ndarray
            Presynaptic spiking cells.
        synDelays : numpy.ndarray
            Per connection specific delays.


        Returns
        -------
        None


        See also
        --------
        Population.insert_all_synapses

        """
        # Insert synapses in an iterative fashion
        try:
            spikes = self.networkSim.dbs[X].select(SpCell[:idx.size])
        except AttributeError:
            raise AssertionError(
                'could not open CachedNetwork database objects')

        # convert to object array for slicing
        spikes = np.array(spikes, dtype=object)

        # apply synaptic delays
        if synDelays is not None and idx.size > 0:
            for i, delay in enumerate(synDelays):
                if spikes[i].size > 0:
                    spikes[i] += delay

        # unique postsynaptic compartments
        uidx = np.unique(idx)
        for i in uidx:
            st = np.sort(np.concatenate(spikes[idx == i]))
            st += cell.tstart  # needed?
            if st.size > 0:
                synParams.update({'idx': i})
                # Create synapse(s) and setting times using class LFPy.Synapse
                synapse = LFPy.Synapse(cell, **synParams)
                synapse.set_spike_times(st)
            else:
                pass

    def fetchSpCells(self, nodes, numSyn):
        """
        For N (nodes count) nestSim-cells draw
        POPULATION_SIZE x NTIMES random cell indexes in
        the population in nodes and broadcast these as `SpCell`.

        The returned argument is a list with len = numSyn.size of np.arrays,
        assumes `numSyn` is a list


        Parameters
        ----------
        nodes : numpy.ndarray, dtype=int
            Node # of valid presynaptic neurons.
        numSyn : numpy.ndarray, dtype=int
            # of synapses per connection.


        Returns
        -------
        SpCells : list
            presynaptic network-neuron indices


        See also
        --------
        Population.fetch_all_SpCells
        """
        SpCell = []
        for size in numSyn:
            SpCell.append(np.random.randint(nodes.min(), nodes.max(),
                                            size=size).astype('int32'))
        return SpCell


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

    def get_all_synDelays(self):
        """
        Create and load arrays of connection delays per connection on this rank

        Get random normally distributed synaptic delays,
        returns dict of nested list of same shape as SpCells.

        Delays are rounded to dt.

        This function takes no kwargs.


        Parameters
        ----------
        None


        Returns
        -------
        dict
            output[cellindex][populationname][layerindex]`, np.array of
            delays per connection.


        See also
        --------
        numpy.random.normal

        """
        tic = time()  # timing

        # ok then, we will draw random numbers across ranks, which have to
        # be unique per cell. Now, we simply record the random state,
        # change the seed per cell, and put the original state back below.
        randomstate = np.random.get_state()

        # container
        delays = {}

        for cellindex in self.RANK_CELLINDICES:
            # set the random seed on for each cellindex
            np.random.seed(
                self.POPULATIONSEED +
                cellindex +
                2 *
                self.POPULATION_SIZE)

            delays[cellindex] = {}
            for j, X in enumerate(self.X):
                delays[cellindex][X] = []
                if 'delays' not in list(
                        self.topology_connections[X][self.y].keys()):
                    # old behaviour, draw delays from normal distribution
                    for i, k in enumerate(self.k_yXL[:, j]):
                        loc = self.synDelayLoc[j]
                        loc /= self.dt
                        scale = self.synDelayScale[j]
                        if scale is not None:
                            scale /= self.dt
                            delay = np.random.normal(loc, scale, k).astype(int)
                            inds = delay < 1
                            while np.any(inds):
                                delay[inds] = np.random.normal(
                                    loc, scale, inds.sum()).astype(int)
                                inds = delay < 1
                            delay = delay.astype(float)
                            delay *= self.dt
                        else:
                            delay = np.zeros(k) + self.synDelayLoc[j]
                        delays[cellindex][X] += [delay]
                else:
                    topo_conn = self.topology_connections[X][self.y]['delays']
                    if 'linear' in list(topo_conn.keys()):
                        # ok, we're using linear delays,
                        # delay(r) = a * r + c
                        a = topo_conn['linear']['a']
                        c = topo_conn['linear']['c']

                        # radial distance to all cells
                        r = _calc_radial_dist_to_cell(
                            self.pop_soma_pos[cellindex]['x'],
                            self.pop_soma_pos[cellindex]['y'],
                            self.networkSim.positions[X],
                            self.topology_connections[X][self.y]['extent'][0],
                            self.topology_connections[X][self.y]['extent'][1],
                            self.topology_connections[X][self.y]['edge_wrap'])

                        # get presynaptic unit GIDs for connections
                        i0 = self.networkSim.nodes[X][0]
                        for i, k in enumerate(self.k_yXL[:, j]):
                            x = self.SpCells[cellindex][X][i]
                            if self.synDelayScale[j] is not None:
                                scale = self.synDelayScale[j]
                                delay = np.random.normal(0, scale, k)
                                # avoid zero or lower delays
                                inds = delay < self.dt - c
                                while np.any(inds):
                                    # print inds.size
                                    delay[inds] = np.random.normal(0, scale,
                                                                   inds.sum())
                                    inds = delay < self.dt - c
                            else:
                                delay = np.zeros(x.size)

                            # add linear dependency
                            delay += r[x - i0] * a + c

                            # round delays to nearest dt
                            delay /= self.dt
                            delay = np.round(delay) * self.dt

                            # fill in values
                            delays[cellindex][X] += [delay]
                    else:
                        raise NotImplementedError(
                            '{0} delay not implemented'.format(
                                list(
                                    topo_conn.keys())[0]))

        # reset the random number generator
        np.random.set_state(randomstate)

        if RANK == 0:
            print(('found delays in %.2f seconds' % (time() - tic)))

        return delays

    def get_all_SpCells(self):
        """
        For each postsynaptic cell existing on this RANK, load or compute
        the presynaptic cell index for each synaptic connection to the
        postsynaptic cell, using distance-dependent connectivity

        This function takes no kwargs.


        Parameters
        ----------
        None


        Returns
        -------
        SpCells : dict
            `output[cellindex][populationname][layerindex]`, np.array of
            presynaptic cell indices.


        See also
        --------
        Population.fetchSpCells, TopoPopulation.fetchSpCells

        """
        tic = time()  # timing

        # ok then, we will draw random numbers across ranks, which have to
        # be unique per cell. Now, we simply record the random state,
        # change the seed per cell, and put the original state back below.
        randomstate = np.random.get_state()

        # set the random seed on for first cellindex on RANK
        if self.RANK_CELLINDICES.size > 0:
            np.random.seed(
                self.POPULATIONSEED +
                self.RANK_CELLINDICES[0] +
                self.POPULATION_SIZE)
        else:
            pass  # no cells should be drawn here anyway.

        SpCells = _get_all_SpCells(self.RANK_CELLINDICES,
                                   self.X,
                                   self.y,
                                   self.pop_soma_pos,
                                   self.networkSim.positions,
                                   self.topology_connections,
                                   self.networkSim.nodes,
                                   self.k_yXL)

        # reset the random number generator
        np.random.set_state(randomstate)

        if RANK == 0:
            print(('found presynaptic cells in %.2f seconds' % (time() - tic)))

        # resync
        COMM.Barrier()

        return SpCells

    def set_pop_soma_pos(self):
        """
        Set `pop_soma_pos` using draw_rand_pos().

        This method takes no keyword arguments.


        Parameters
        ----------
        None


        Returns
        -------
        np.ndarray
            (x,y,z) coordinates of each neuron in the population


        See also
        --------
        TopoPopulation.draw_rand_pos

        """
        if RANK == 0:
            pop_soma_pos = self.draw_rand_pos(**self.populationParams)
        else:
            pop_soma_pos = None
        return COMM.bcast(pop_soma_pos, root=0)

    def draw_rand_pos(self, z_min, z_max, position_index_in_Y, **args):
        """
        Draw cell positions from the CachedNetwork object (x,y), but with
        randomized z-position between z_min and z_max.
        Returned argument is a list of dicts
        [{'x' : float, 'y' : float, 'z' : float}, {...}].


        Parameters
        ----------
        z_min : float
            Lower z-boundary of population.
        z_max : float
            Upper z-boundary of population.
        position_index_in_Y : list
            parent population Y of cell type y in Y, first index for
            slicing pos
        **args : keyword arguments
            Additional inputs that is being ignored.


        Returns
        -------
        soma_pos : list
            List of dicts of len population size
            where dict have keys x, y, z specifying
            xyz-coordinates of cell at list entry `i`.


        See also
        --------
        TopoPopulation.set_pop_soma_pos

        """

        print("assess somatic locations: ")
        Y, ind = position_index_in_Y
        z = np.random.rand(self.POPULATION_SIZE) * (z_max - z_min) + z_min
        soma_pos = [{'x': x, 'y': y, 'z': z[k]} for k, (x, y) in enumerate(
            self.networkSim.positions[Y][ind:ind + z.size])]

        print('done!')

        return soma_pos


if __name__ == '__main__':
    import doctest
    doctest.testmod()