#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Hybrid LFP scheme example script, applying the methodology with the model of:

Potjans, T. and Diesmann, M. "The Cell-Type Specific Cortical Microcircuit:
Relating Structure and Activity in a Full-Scale Spiking Network Model".
Cereb. Cortex (2014) 24 (3): 785-806.
doi: 10.1093/cercor/bhs358

Synopsis of the main simulation procedure:
1. Loading of parameterset
    a. network parameters
    b. parameters for hybrid scheme
2. Set up file destinations for different simulation output
3. network simulation
    a. execute network simulation using NEST (www.nest-initiative.org)
    b. merge network output (spikes, currents, voltages)
4. Create a object-representation that uses sqlite3 of all the spiking output 
5. Iterate over post-synaptic populations:
    a. Create Population object with appropriate parameters for
       each specific population
    b. Run all computations for populations
    c. Postprocess simulation output of all cells in population
6. Postprocess all cell- and population-specific output data
7. Create a tarball for all non-redundant simulation output

The full simulation can be evoked by issuing a mpirun call, such as
mpirun -np 64 python cellsim16pops.py

Given the size of the network and demands for the multi-compartment LFP-
predictions using the present scheme, running the model on nothing but a large-
scale compute facility is strongly discouraged.

'''
import os
import numpy as np
from time import time
import neuron # NEURON compiled with MPI must be imported before NEST and mpi4py
              # to avoid NEURON being aware of MPI.
import nest   # Import not used, but done in order to ensure correct execution
import nest_simulation
from hybridLFPy import PostProcess, Population, CachedNetwork, setup_file_dest
import nest_output_processing


#set some seed values
SEED = 12345678
SIMULATIONSEED = 12345678
np.random.seed(SEED)



################################################################################
## PARAMETERS
################################################################################

# from params_evoked_with_EEG import multicompartment_params, \
#                                point_neuron_network_params, \
#                                MorphoPath, Connectomics

import params_evoked_with_EEG 


#Full set of parameters including network parameters
params = multicompartment_params()

#set up the file destination
tic = time()

setup_file_dest(params, clearDestination=True)


###############################################################################
# MAIN simulation procedure
###############################################################################

#tic toc


######## Perform network simulation ############################################

# NEED TO CHANGE IN BRIAN2


##initiate nest simulation with only the point neuron network parameter class
networkParams = point_neuron_network_params()
nest_simulation.sli_run(parameters=networkParams,
                       fname='microcircuit.sli',
                       verbosity='M_WARNING')

#preprocess the gdf files containing spiking output, voltages, weighted and
#spatial input spikes and currents:
nest_output_processing.merge_gdf(networkParams,
                            raw_label=networkParams.spike_detector_label,
                            file_type='gdf',
                            fileprefix=params.networkSimParams['label'])
nest_output_processing.merge_gdf(networkParams,
                            raw_label=networkParams.voltmeter_label,
                            file_type='dat',
                            fileprefix='voltages')
nest_output_processing.merge_gdf(networkParams,
                            raw_label=networkParams.weighted_input_spikes_label,
                            file_type='dat',
                            fileprefix='population_input_spikes')
##spatial input currents
#nest_output_processing.create_spatial_input_spikes_hdf5(networkParams,
#                                        fileprefix='depth_res_input_spikes-')


#Create an object representation of the simulation output that uses sqlite3
networkSim = CachedNetwork(**params.networkSimParams)


toc = time() - tic
print('NEST simulation and gdf file processing done in  %.3f seconds' % toc)



# ----------------------------------------

# MORPHOLOGICAL PATHS

# ----------------------------------------

# Initialize the morphology loading pathways
Morph_pathways = MorphoPath()

# Construct the object
name_list = [
    "L23_exc", "L23_inh", 
    "L4_exc", "L4_inh", "L4_ss", 
    "L5_exc", "L5_inh", 
    "L6_exc", "L6_inh"
]
name_path = r''
morph_path = r''
Morph_pathways.construct_dict(name_list, name_path, morph_path)
Paths = Morph_pathways.get_paths



# ----------------------------------------

# CONNECTOMICS 

# ----------------------------------------

connectomics_path = r''
SpanTree_path = r''
N_synapse_path = r''
connectomics_output = r''



conn =  Connectomics(connectomics_path,connectomics_output,N_synapse_path,SpanTree_path,name_list)
conn_dict = conn.get_ConnectomicInfo


# Extract Infos
extracted_pops = extract_macro_populations(conn_dict, name_list)  # cell_coords, local_to_raw_map, pre_to_post, post_to_pre

# Map between the specific NMC mtype and its layer and synaptic type.
mtype_fast_lookup = conn_dict['mtype_fast_lookup']

# 1xN list of all cells coupled by the index to all the keys' value in conn_dict. 
cell_mtypes = conn_dict['cell_mtypes']

# Dictionary with the spanning trees distribution
TreeDensity_load = conn_dict['TreeDensity_load']


Pop_to_Syntype = { 'L23_exc' : 'exc',
      'L23_inh' : 'inh',
      
      'L4_exc' : 'exc', 
      'L4_ss' : 'exc', 
      'L4_inh' : 'inh', 

      'L5_exc' : 'exc', 
      'L5_inh' : 'inh', 

      'L6_exc' : 'exc', 
      'L6_inh' : 'inh'

          }


syn_path = ''


####### Set up populations #####################################################

#iterate over each cell type, and create populationulation object
for i, Pop in enumerate(name_list):
    #create population:
    Macro_Pop = Population(
            #parent class
            cellParams = params.yCellParams[y],
            rand_rot_axis = params.rand_rot_axis[y],
            simulationParams = params.simulationParams,
            populationParams = params.populationParams[y],
            Pop = Pop,
            layerBoundaries = params.layerBoundaries,
            electrodeParams = params.electrodeParams,
            savelist = params.savelist,
            savefolder = params.savefolder,
            calculateCSD = params.calculateCSD,
            dt_output = params.dt_output, 
            POPULATIONSEED = SIMULATIONSEED + i,


            # New
            SubPopulations_list = Paths[Pop],
            Pop_to_Syntype = Pop_to_Syntype,
            synapse_base_path = syn_path,
            TreeDensity_load = TreeDensity_load,


            cell_mtypes = cell_mtypes,
            mtype_fast_lookup = mtype_fast_lookup,


            local_to_raw_map= extracted_pops[Pop]['local_to_raw_map'],
            Cell_afferences = extracted_pops[Pop]['synapse_dict'],
            Cell_coords = extracted_pops[Pop]['cell_coords'],

            #daughter class kwargs
            X = params.X,
            networkSim = networkSim,
            k_yXL = params.k_yXL[y],
            synParams = params.synParams[y],
            synDelayLoc = params.synDelayLoc[y],
            synDelayScale = params.synDelayScale[y],
            J_yX = params.J_yX[y],
            tau_yX = params.tau_yX[y],
            recordSingleContribFrac = params.recordSingleContribFrac,
        )
    #run population simulation and collect the data
    Macro_Pop.run()
    Macro_Pop.collect_data()
    

    #object no longer needed
    del Macro_Pop


####### Postprocess the simulation output ######################################


#reset seed, but output should be deterministic from now on
np.random.seed(SIMULATIONSEED)

#do some postprocessing on the collected data, i.e., superposition
#of population LFPs, CSDs etc
postproc = PostProcess(y = params.y,
                       dt_output = params.dt_output,
                       savefolder = params.savefolder,
                       mapping_Yy = params.mapping_Yy,
                       )

#run through the procedure
postproc.run()

#create tar-archive with output for plotting
postproc.create_tar_archive()
postproc.create_cdm_tar_archive()

#tic toc
print('Execution time: %.3f seconds' % (time() - tic))
