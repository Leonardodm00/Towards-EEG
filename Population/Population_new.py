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

        # cell = self.cellsim(cellindex, return_just_cell=True)
        ###CHANGED
        cell = self.cellsim(cellindex, return_just_cell=True)

        # local containers
        synidx = {}

        # get synaptic placements and cells from the network,
        # then set spike times,
        for i, X in enumerate(self.X):
            synidx[X] = self.fetchSynIdxCell(cell=cell,
                                             nidx=self.k_yXL[:, i],
                                             synParams=self.synParams.copy())
        # clean up hoc namespace
        cell.__del__()

        return synidx

    def fetchSynIdxCell(self, cell, nidx, synParams):
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


        Returns
        -------
        syn_idx : list
            List of arrays of synapse placements per connection.


        See also
        --------
        Population.get_all_synIdx, Population.get_synIdx, LFPy.Synapse

        """

        # segment indices in each layer is stored here, list of np.array
        syn_idx = []
        # loop over layer bounds, find synapse locations
        for i, zz in enumerate(self.layerBoundaries):
            if nidx[i] == 0:
                syn_idx.append(np.array([], dtype=int))
            else:
                syn_idx.append(cell.get_rand_idx_area_norm(
                    section=synParams['section'],
                    nidx=nidx[i],
                    z_min=zz.min(),
                    z_max=zz.max()).astype('int16'))

        return syn_idx
    



    def _plot_both_morphologies(dend_segments, layer_boundaries, layer_names, valid_layers, target_upper_z, target_lower_z, morph_idx):
        """Helper function to visualize the selected morphology at both boundaries side-by-side."""
        fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharey=True)

        color_palette = ['#e6194B', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#42d4f4']
        layer_colors = {layer_idx: color_palette[i % len(color_palette)] for i, layer_idx in enumerate(valid_layers)}

        placements = [
            (axes[0], target_upper_z, "Upper Boundary Placement"),
            (axes[1], target_lower_z, "Lower Boundary Placement")
        ]

        for ax, soma_z, title in placements:
            # 1. Plot layer boundaries
            for i, bounds in enumerate(layer_boundaries):
                upper_bound, lower_bound = bounds
                ax.axhline(upper_bound, color='black', linestyle='--', alpha=0.5)

                bg_color = layer_colors.get(i, 'lightgrey')
                alpha_val = 0.1 if i in valid_layers else 0.05
                ax.axhspan(lower_bound, upper_bound, facecolor=bg_color, alpha=alpha_val)

                # Only draw the text labels on the left plot to avoid clutter
                if ax == axes[0]:
                    layer_label = layer_names[i] if i < len(layer_names) else f"Layer {i}"
                    ax.text(-200, (upper_bound + lower_bound) / 2, layer_label,
                            va='center', ha='right', fontsize=10, fontweight='bold')

            ax.axhline(layer_boundaries[-1][1], color='black', linestyle='--', alpha=0.5)

            # 2. Plot dendritic segments
            lines = []
            line_colors = []

            for mid_z, _, p1, p2 in dend_segments:
                shifted_z1 = p1[2] + soma_z
                shifted_z2 = p2[2] + soma_z
                mid_shifted_z = mid_z + soma_z

                segment = [(p1[0], shifted_z1), (p2[0], shifted_z2)]
                lines.append(segment)

                seg_color = 'darkgrey'
                for layer_idx in valid_layers:
                    upper_bound, lower_bound = layer_boundaries[layer_idx]
                    if lower_bound <= mid_shifted_z <= upper_bound:
                        seg_color = layer_colors[layer_idx]
                        break
                line_colors.append(seg_color)

            lc = LineCollection(lines, colors=line_colors, linewidths=1.5, alpha=0.8)
            ax.add_collection(lc)

            # 3. Plot the soma
            ax.scatter([0], [soma_z], color='black', s=100, zorder=5, label='Soma')

            # 4. Final plot formatting
            ax.set_aspect('equal')
            ax.set_xlim(-250, 250)
            ax.set_ylim(layer_boundaries[-1][1] - 50, layer_boundaries[0][0] + 50)
            ax.set_xlabel('X distance (um)')
            if ax == axes[0]:
                ax.set_ylabel('Depth / Z distance (um)')
            ax.set_title(title)
            ax.legend()

        fig.suptitle(f'Selected Morphology (Index: {morph_idx}) Validated For Both Placements', fontsize=16)
        plt.tight_layout()
        plt.show()

    def cell_MorphSelect(
        morph_paths,
        layer_boundaries,
        layer_names,
        synapses_per_layer,
        length_threshold,
        target_layer_idx,
        plot_result=False
    ):
        """

        Selects a random morphology meeting a length threshold within synapse-containing
        layers using the NEURON simulator library. It evaluates the morphology by placing
        its soma at both the upper and lower boundaries of its target layer.

        Args:
            morph_paths (list of str):
                A list containing the file paths to the candidate .hoc morphology files.
                Example: ['/path/to/neuron_1.hoc', '/path/to/neuron_2.hoc']

            layer_boundaries (numpy.ndarray):
                A 2D array of shape (N, 2) where N is the number of cortical layers.
                Each row must contain exactly two float values representing the
                [upper_Z, lower_Z] boundaries in micrometers (um). Values are typically
                negative, moving deeper from the pia (0.0).
                Example: np.array([[0.0, -81.6], [-81.6, -587.1], ...])

            layer_names (list of str):
                A list of strings containing the display names for each layer. The length
                must exactly match the number of rows in `layer_boundaries`. Used purely
                for labeling the output plot.
                Example: ['L1', 'L2/3', 'L4', 'L5', 'L6']

            synapses_per_layer (list of int or float):
                A 1D list storing the total number of synapses per layer. The length
                must match the rows in `layer_boundaries`. The function checks if an
                entry is > 0 to determine if a layer is "valid" for dendritic length counting.
                Example: [0, 1500, 3200, 500, 0]

            length_threshold (float):
                The minimum cumulative length of the dendritic arbor (in um) that *must* fall within the synapse-containing layers for the morphology to be accepted.
                Example: 1500.0

            target_layer_idx (int):
                A single integer representing the 0-based index of the layer to which
                the chosen neuron belongs. Used to look up the upper and lower Z-boundaries
                in `layer_boundaries` to position the soma during the stress test.
                Example: 3 (Targets the 4th row in layer_boundaries)

            plot_result (bool, optional):
                If True, generates a side-by-side matplotlib figure showing the successful
                neuron placed at both the upper and lower boundaries. Defaults to False.

        Returns:
            int: The index of the successfully selected morphology from the original
                `morph_paths` list.

        Raises:
            RuntimeError: If the function evaluates every file in `morph_paths` and none
                        meet the length threshold for both extreme soma placements.






        Neuron Morphology Selection Script

        This script acts as a strict filter for neuronal morphologies. Its goal is to
        find a single random neuron from a dataset whose dendritic tree is large enough
        to reach the right synaptic connections, regardless of where its soma sits inside
        its home layer.

        How it works:
        1. It shuffles the provided list of candidate .hoc morphologies for random selection.
        2. It loads a candidate into the NEURON simulator, isolates the dendritic tree
        (basal and apical), extracts its 3D coordinates, and scales them from nm to um.
        3. It performs a two-part stress test by virtually placing the neuron's soma at
        the absolute top of its assigned home layer, and then at the absolute bottom.
        4. For both extreme positions, it calculates exactly how much of the dendritic
        tree falls inside the specific layers designated as having synapses.
        5. If the total dendritic length in those synaptic layers drops below the minimum
        threshold during *either* the top or bottom placement, the neuron is rejected.
        6. The first neuron to successfully meet the threshold for *both* placements is
        chosen. The script then generates a side-by-side plot of both placements and
        returns the winning neuron's index. If no neurons pass, it raises an error.



        Selects a random morphology meeting a length threshold within synapse-containing
        layers using the NEURON simulator library.
        """
        valid_layers = [i for i, syn in enumerate(synapses_per_layer) if syn > 0]
        target_upper_z = layer_boundaries[target_layer_idx][0]
        target_lower_z = layer_boundaries[target_layer_idx][1]

        shuffled_paths = list(enumerate(morph_paths))
        random.shuffle(shuffled_paths)

        for original_idx, path in shuffled_paths:
            # Clear previous morphology from NEURON memory
            h('forall delete_section()')

            # Load the new .hoc file
            success = h.load_file(str(path))
            if not success:
                print(f"Warning: Could not load {path}")
                continue

            dend_segments = []

            # Iterate through all sections loaded into NEURON
            for sec in h.allsec():
                if 'dend' in sec.name().lower() or 'apic' in sec.name().lower():
                    n3d = int(h.n3d(sec=sec))
                    if n3d > 0:
                        pts = np.array([[h.x3d(i, sec=sec), h.y3d(i, sec=sec), h.z3d(i, sec=sec)] for i in range(n3d)])

                        for i in range(n3d - 1):
                            p1 = pts[i] / 1000.0
                            p2 = pts[i+1] / 1000.0

                            length = np.linalg.norm(p1 - p2)
                            mid_z = (p1[2] + p2[2]) / 2.0

                            dend_segments.append((mid_z, length, p1, p2))

            valid_for_both_placements = True

            # Test BOTH upper and lower boundaries
            for soma_z_placement in [target_upper_z, target_lower_z]:
                valid_dend_length = 0.0

                for mid_z, length, _, _ in dend_segments:
                    shifted_z = mid_z + soma_z_placement

                    in_synapse_layer = False
                    for layer_idx in valid_layers:
                        upper_bound, lower_bound = layer_boundaries[layer_idx]
                        if lower_bound <= shifted_z <= upper_bound:
                            in_synapse_layer = True
                            break

                    if in_synapse_layer:
                        valid_dend_length += length

                # If the length drops below the threshold for EITHER placement, it fails
                if valid_dend_length < length_threshold:
                    valid_for_both_placements = False
                    break

            # If it passed BOTH loops, we generate the side-by-side plot and return it
            if valid_for_both_placements:
                if plot_result:
                    _plot_both_morphologies(
                        dend_segments, layer_boundaries, layer_names, valid_layers,
                        target_upper_z, target_lower_z, original_idx
                    )
                return original_idx

        raise RuntimeError(
            f"No morphology met the length threshold of {length_threshold}um "
            f"within the valid synaptic layers."
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

        The variable self.Subpop is structured as a dataframe in whcih the keys are the sub-populations reference while the vlaues are the lists
        used to import the morphologies. As explained above the morphologies will be drawn randomly and checked before passing them to the 
        LFPy.Cell() function. This will be done by passing an updated version of the cellParams dictionary where the field 'morphology'
        is updaed with the choosen morphology.

        The variable self.ImplementedMorph has a similar structure. The top key in the hierarchy refers to the subpopulation, while the value
        stores matrix with self.RANK_CELLINDICES x 1 where the index is the current cell index 'cellindex' while is stored the index of the
        pathway for the loading of the morphology.

        
        
        ---------  return_just_cell == FALSE ---------
        In this case we need to upload the correct morphology based on the sifting done in the previouse call of the same function but with
        return_just_cell == TRUE used only to load the morphology and place the synapses onto it.
        Thus for a specific sub-population we resort to the variable self.ImplementedMorph[self.y] which encompasses the list of indicies
        that locate in the self.Subpop[self.y] variable. The correct morphology is located by the current 'cellindex' which inherently 
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

        ## Morphology choice
        Availale_morph = self.Subpop[self.y].copy() 
        Layer_index = self.layer_map[self.y]
        Sum_per_Layer = np.sum(self.k_yXL, axis=1)
        Load_idx = cell_MorphSelect(Availale_morph,
                                    self.layerBoundaries,
                                    self.layer_names,
                                    Sum_per_Layer,
                                    self.lenTh,
                                    Layer_index,
                                    plot_result=False)
        # Update the dictionary
        self.ImplementedMorph[self.y].append(Load_idx)
        # Change the cellParams' field, this will change iteratively.
        self.cellParams['morphology'] = Availale_morph[Load_idx]






        ### Here we create the cell object: initiates the morphology, set its position and rotation, and insert synapses.
        ### Maybe it would be possible to change the cellParams. for example add to the 'morphology' field the specific path to the cell's morphology file.
        ### We need to load the cell-specific morphology file.
        cell = LFPy.Cell(**self.cellParams)
        cell.set_pos(**self.pop_soma_pos[cellindex])
        cell.set_rotation(**self.rotations[cellindex])

        if return_just_cell:
            return cell
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
        for i, X in enumerate(self.X):  # range(self.k_yXL.shape[1]):
            synParams = self.synParams
            synParams.update({
                'weight': self.J_yX[i],
                'tau': self.tau_yX[i],
            })
            for j in range(len(self.synIdx[cellindex][X])):
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
