#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Modified parameters file for the Hybrid LFP scheme, applying the methodology with
the model of:

Potjans, T. and Diesmann, M. "The Cell-Type Specific Cortical Microcircuit:
Relating Structure and Activity in a Full-Scale Spiking Network Model".
Cereb. Cortex (2014) 24 (3): 785-806.
doi: 10.1093/cercor/bhs358


'''
import numpy as np
import os
import json
from mpi4py import MPI #this is needed to initialize other classes correctly


###################################
# Initialization of MPI stuff     #
###################################
COMM = MPI.COMM_WORLD
SIZE = COMM.Get_size()
RANK = COMM.Get_rank()

'''
TODO: rename to simulation_and_model_params.py
'''


####################################
# HELPER FUNCTIONS                 #
####################################

flattenlist = lambda lst: sum(sum(lst, []),[])



####################################
# SPATIAL CONNECTIVITY EXTRACTION  #
####################################

'''
Include functions that extract information from binzegger.json here
'''



def get_F_y(fname='binzegger_connectivity_table.json', y=['p23']): 
    '''
    Extract frequency of occurrences of those cell types that are modeled.
    The data set contains cell types that are not modeled (TCs etc.)
    The returned percentages are renormalized onto modeled cell-types, i.e. they sum up to 1 
    '''
    # Load data from json dictionary
    f = open(fname,'r')
    data = json.load(f)
    f.close()
    
    occurr = []
    for cell_type in y:
        occurr += [data['data'][cell_type]['occurrence']]
    return list(np.array(occurr)/np.sum(occurr)) 



def get_L_yXL(fname, y, x_in_X, L):
    '''
    compute the layer specificity, defined as:
    ::
    
        L_yXL = k_yXL / k_yX
    '''
    def _get_L_yXL_per_yXL(fname, x_in_X, X_index,
                                  y, layer):
        # Load data from json dictionary
        f = open(fname, 'r')
        data = json.load(f)
        f.close()
    
        
        # Get number of synapses
        if layer in [str(key) for key in list(data['data'][y]['syn_dict'].keys())]:
            #init variables
            k_yXL = 0
            k_yX = 0
            
            for x in x_in_X[X_index]:
                p_yxL = data['data'][y]['syn_dict'][layer][x] / 100.
                k_yL = data['data'][y]['syn_dict'][layer]['number of synapses per neuron']
                k_yXL += p_yxL * k_yL
                
            for l in [str(key) for key in list(data['data'][y]['syn_dict'].keys())]:
                for x in x_in_X[X_index]:
                    p_yxL = data['data'][y]['syn_dict'][l][x] / 100.
                    k_yL = data['data'][y]['syn_dict'][l]['number of synapses per neuron']
                    k_yX +=  p_yxL * k_yL
            
            if k_yXL != 0.:
                return k_yXL / k_yX
            else:
                return 0.
        else:
            return 0.


    #init dict
    L_yXL = {}

    #iterate over postsynaptic cell types
    for y_value in y:
        #container
        data = np.zeros((len(L), len(x_in_X)))
        #iterate over lamina
        for i, Li in enumerate(L):
            #iterate over presynapse population inds
            for j in range(len(x_in_X)):
                data[i][j]= _get_L_yXL_per_yXL(fname, x_in_X,
                                                          X_index=j,
                                                          y=y_value,
                                                          layer=Li)
        L_yXL[y_value] = data

    return L_yXL



def get_T_yX(fname, y, y_in_Y, x_in_X, F_y):
    '''
    compute the cell type specificity, defined as:
    ::
    
        T_yX = K_yX / K_YX
            = F_y * k_yX / sum_y(F_y*k_yX) 
    
    
    '''
    def _get_k_yX_mul_F_y(y, y_index, X_index):
        # Load data from json dictionary
        f = open(fname, 'r')
        data = json.load(f)
        f.close()
    
        #init variables
        k_yX = 0.
        
        for l in [str(key) for key in list(data['data'][y]['syn_dict'].keys())]:
            for x in x_in_X[X_index]:
                p_yxL = data['data'][y]['syn_dict'][l][x] / 100.
                k_yL = data['data'][y]['syn_dict'][l]['number of synapses per neuron']
                k_yX +=  p_yxL * k_yL
        
        return k_yX * F_y[y_index]


    #container
    T_yX = np.zeros((len(y), len(x_in_X)))
    
    #iterate over postsynaptic cell types
    for i, y_value in enumerate(y):
        #iterate over presynapse population inds
        for j in range(len(x_in_X)):
            k_yX_mul_F_y = 0
            for k, yy in enumerate(sum(y_in_Y, [])):                
                if y_value in yy:
                    for yy_value in yy:
                        ii = np.where(np.array(y) == yy_value)[0][0]
                        k_yX_mul_F_y += _get_k_yX_mul_F_y(yy_value, ii, j)
            
            
            if k_yX_mul_F_y != 0:
                T_yX[i, j] = _get_k_yX_mul_F_y(y_value, i, j) / k_yX_mul_F_y
            
    return T_yX


class general_params(object):
    def __init__(self):
        '''class collecting general model parameters'''

        ####################################
        # REASON FOR THIS SIMULATION       #
        ####################################

        self.reason = 'EEG calculation from evoked activity'

    ####################################
    #                                  #
    #                                  #
    #     SIMULATION PARAMETERS        #
    #                                  #
    #                                  #
    ####################################  


        ####################################
        # MAIN SIMULATION CONTROL          #
        ####################################

        # simulation step size
        self.dt = 2**-3

        # simulation start
        self.tstart = 0

        # simulation stop
        self.tstop = 4200


        ####################################
        # OUTPUT LOCATIONS                 #
        ####################################
        
        # TODO: try except does not work with hambach

        # folder for all simulation output and scripts
        # HAMBACH and STALLO have scratch areas for saving
        if os.path.isdir(os.path.join('/', 'scratch', os.environ['USER'])):
            self.savefolder = os.path.join('/', 'scratch', os.environ['USER'],
                                           'hybrid_model',
                                           'evoked_cdm')
        # # LOCALLY
        else:
            self.savefolder = 'evoked_cdm'

        # folder for simulation scripts
        self.sim_scripts_path = os.path.join(self.savefolder, 'sim_scripts')

        # folder for each individual cell's output
        self.cells_path = os.path.join(self.savefolder, 'cells')

        # folder for figures
        self.figures_path = os.path.join(self.savefolder, 'figures')

        # folder for population resolved output signals
        self.populations_path = os.path.join(self.savefolder, 'populations')

        # folder for raw nest output files
        self.raw_nest_output_path = os.path.join(self.savefolder,
                                                 'raw_nest_output')

        # folder for raw nest output files
        self.cdm_path = os.path.join(self.savefolder, 'cdm')


        # folder for processed nest output files
        self.spike_output_path = os.path.join(self.savefolder,
                                                       'processed_nest_output')

    
    ####################################
    #                                  #
    #                                  #
    #        MODEL PARAMETERS          #
    #                                  #
    #                                  #
    ####################################
    


        ####################################
        # POPULATIONS                      #
        ####################################

        # Number of populations
        self.Npops = 9


        # Microcolumn properties
        conn = Connectomics(Calculate=False)
        self.Col_prop = conn.get_ColumnProp


        # Number of thalamic neurons/ point processes
        self.n_thal = 902

        # population names TODO: rename
        self.Pre = ['TC',
                "L23_exc", "L23_inh", 
                "L4_exc", "L4_inh", "L4_ss", 
                "L5_exc", "L5_inh", 
                "L6_exc", "L6_inh"]
        self.Post = self.Pre[1:]

        # TC and cortical population sizes in one list TODO: rename
        self.N_X = np.array([self.n_thal]+flattenlist([self.full_scale_num_neurons]))

    
        import numpy as np

        # 1. Base values extracted from literature (L2/3 and L5)
        # Explicitly defining every Pre -> Post combination.
        base_params = {
            'L23': {'PY': {}, 'PV': {}, 'SST': {}, 'VIP': {}},
            'L5':  {'PY': {}, 'PV': {}, 'SST': {}, 'VIP': {}}
        }

        # ==========================================
        # LAYER 2/3 DEFINITIONS
        # ==========================================

        # --- PRE: PY (Excitatory) ---
        base_params['L23']['PY']['PY']  = {'Use': 0.46, 'Dep': 670.0, 'Fac': 17.0,  'tau_r_AMPA': 0.2, 'tau_d_AMPA': 1.7, 'tau_r_NMDA': 0.29, 'tau_d_NMDA': 43.0, 'gmax': 0.0002482}
        base_params['L23']['PY']['SST'] = {'Use': 0.09, 'Dep': 140.0, 'Fac': 670.0, 'tau_r_AMPA': 0.2, 'tau_d_AMPA': 1.7, 'tau_r_NMDA': 0.29, 'tau_d_NMDA': 43.0, 'gmax': 0.00038}
        base_params['L23']['PY']['PV']  = {'Use': 0.88, 'Dep': 510.0, 'Fac': 180.0, 'tau_r_AMPA': 0.2, 'tau_d_AMPA': 1.7, 'tau_r_NMDA': 0.29, 'tau_d_NMDA': 43.0, 'gmax': 0.000337}
        base_params['L23']['PY']['VIP'] = {'Use': 0.50, 'Dep': 670.0, 'Fac': 17.0,  'tau_r_AMPA': 0.2, 'tau_d_AMPA': 1.7, 'tau_r_NMDA': 0.29, 'tau_d_NMDA': 43.0, 'gmax': 0.00031}

        # --- PRE: SST (Inhibitory - Martinotti/LTS) ---
        base_params['L23']['SST']['PY']  = {'Use': 0.30, 'Dep': 1300.0, 'Fac': 2.0,   'tau_r': 0.2, 'tau_d': 5.0, 'gmax': 0.00124}
        base_params['L23']['SST']['SST'] = {'Use': 0.25, 'Dep': 720.0,  'Fac': 21.0,  'tau_r': 0.2, 'tau_d': 5.0, 'gmax': 0.00034}
        base_params['L23']['SST']['PV']  = {'Use': 0.25, 'Dep': 710.0,  'Fac': 21.0,  'tau_r': 0.2, 'tau_d': 5.0, 'gmax': 0.00033}
        base_params['L23']['SST']['VIP'] = {'Use': 0.31, 'Dep': 890.0,  'Fac': 25.0,  'tau_r': 0.2, 'tau_d': 5.0, 'gmax': 0.00046}

        # --- PRE: PV (Inhibitory - Fast Spiking) ---
        base_params['L23']['PV']['PY']  = {'Use': 0.08, 'Dep': 710.0, 'Fac': 23.0, 'tau_r': 0.2, 'tau_d': 2.5, 'gmax': 0.00291}
        base_params['L23']['PV']['SST'] = {'Use': 0.25, 'Dep': 700.0, 'Fac': 21.0, 'tau_r': 0.2, 'tau_d': 2.5, 'gmax': 0.00033}
        base_params['L23']['PV']['PV']  = {'Use': 0.25, 'Dep': 710.0, 'Fac': 21.0, 'tau_r': 0.2, 'tau_d': 2.5, 'gmax': 0.00033}
        base_params['L23']['PV']['VIP'] = {'Use': 0.26, 'Dep': 720.0, 'Fac': 21.0, 'tau_r': 0.2, 'tau_d': 2.5, 'gmax': 0.00034}

        # --- PRE: VIP (Inhibitory) ---
        base_params['L23']['VIP']['PY']  = {'Use': 0.23, 'Dep': 300.0, 'Fac': 160.0, 'tau_r': 0.2, 'tau_d': 3.5, 'gmax': 0.0}
        base_params['L23']['VIP']['SST'] = {'Use': 0.27, 'Dep': 760.0, 'Fac': 22.0,  'tau_r': 0.2, 'tau_d': 3.5, 'gmax': 0.00036}
        base_params['L23']['VIP']['PV']  = {'Use': 0.25, 'Dep': 720.0, 'Fac': 21.0,  'tau_r': 0.2, 'tau_d': 3.5, 'gmax': 0.00034}
        base_params['L23']['VIP']['VIP'] = {'Use': 0.26, 'Dep': 720.0, 'Fac': 21.0,  'tau_r': 0.2, 'tau_d': 3.5, 'gmax': 0.00034}


        # ==========================================
        # LAYER 5 DEFINITIONS
        # ==========================================

        # --- PRE: PY (Excitatory) ---
        base_params['L5']['PY']['PY']  = {'Use': 0.19,  'Dep': 670.0, 'Fac': 17.0,  'tau_r_AMPA': 0.2, 'tau_d_AMPA': 1.7, 'tau_r_NMDA': 0.29, 'tau_d_NMDA': 43.0, 'gmax': 0.00037}
        base_params['L5']['PY']['PV']  = {'Use': 0.40,  'Dep': 580.0, 'Fac': 120.0, 'tau_r_AMPA': 0.2, 'tau_d_AMPA': 1.7, 'tau_r_NMDA': 0.29, 'tau_d_NMDA': 43.0, 'gmax': 0.00030}
        base_params['L5']['PY']['SST'] = {'Use': 0.094, 'Dep': 150.0, 'Fac': 690.0, 'tau_r_AMPA': 0.2, 'tau_d_AMPA': 1.7, 'tau_r_NMDA': 0.29, 'tau_d_NMDA': 43.0, 'gmax': 0.00030}
        base_params['L5']['PY']['VIP'] = {'Use': 0.50,  'Dep': 670.0, 'Fac': 17.0,  'tau_r_AMPA': 0.2, 'tau_d_AMPA': 1.7, 'tau_r_NMDA': 0.29, 'tau_d_NMDA': 43.0, 'gmax': 0.00030}

        # --- PRE: PV (Inhibitory) ---
        base_params['L5']['PV']['PY']  = {'Use': 0.24, 'Dep': 660.0, 'Fac': 27.0, 'tau_r': 0.2, 'tau_d': 1.8, 'gmax': 0.00092}
        base_params['L5']['PV']['PV']  = {'Use': 0.25, 'Dep': 720.0, 'Fac': 21.0, 'tau_r': 0.2, 'tau_d': 1.8, 'gmax': 0.00034}
        base_params['L5']['PV']['SST'] = {'Use': 0.25, 'Dep': 700.0, 'Fac': 21.0, 'tau_r': 0.2, 'tau_d': 1.8, 'gmax': 0.00033}
        base_params['L5']['PV']['VIP'] = {'Use': 0.24, 'Dep': 680.0, 'Fac': 20.0, 'tau_r': 0.2, 'tau_d': 1.8, 'gmax': 0.00031}

        # --- PRE: SST (Inhibitory) ---
        base_params['L5']['SST']['PY']  = {'Use': 0.30, 'Dep': 1200.0, 'Fac': 2.2,  'tau_r': 0.2, 'tau_d': 5.2, 'gmax': 0.00134}
        base_params['L5']['SST']['PV']  = {'Use': 0.25, 'Dep': 710.0,  'Fac': 21.0, 'tau_r': 0.2, 'tau_d': 5.2, 'gmax': 0.00034}
        base_params['L5']['SST']['SST'] = {'Use': 0.25, 'Dep': 710.0,  'Fac': 21.0, 'tau_r': 0.2, 'tau_d': 5.2, 'gmax': 0.00033}
        base_params['L5']['SST']['VIP'] = {'Use': 0.24, 'Dep': 670.0,  'Fac': 20.0, 'tau_r': 0.2, 'tau_d': 5.2, 'gmax': 0.00031}

        # --- PRE: VIP (Inhibitory) ---
        base_params['L5']['VIP']['PY']  = {'Use': 0.26, 'Dep': 360.0, 'Fac': 100.0, 'tau_r': 0.2, 'tau_d': 3.0, 'gmax': 0.00}
        base_params['L5']['VIP']['PV']  = {'Use': 0.24, 'Dep': 680.0, 'Fac': 20.0,  'tau_r': 0.2, 'tau_d': 3.0, 'gmax': 0.00031}
        base_params['L5']['VIP']['SST'] = {'Use': 0.25, 'Dep': 700.0, 'Fac': 21.0,  'tau_r': 0.2, 'tau_d': 3.0, 'gmax': 0.00033}
        base_params['L5']['VIP']['VIP'] = {'Use': 0.28, 'Dep': 810.0, 'Fac': 23.0,  'tau_r': 0.2, 'tau_d': 3.0, 'gmax': 0.00040}


        # ==========================================
        # AUTOMATED AVERAGING FOR L4 AND L6
        # ==========================================
        genetic_types = ['PY', 'PV', 'SST', 'VIP']

        for target_layer in ['L4', 'L6']:
            base_params[target_layer] = {}
            for pre_gtype in genetic_types:
                base_params[target_layer][pre_gtype] = {}
                for post_gtype in genetic_types:
                    base_params[target_layer][pre_gtype][post_gtype] = {}
                    for key in base_params['L23'][pre_gtype][post_gtype].keys():
                        val23 = base_params['L23'][pre_gtype][post_gtype][key]
                        val5  = base_params['L5'][pre_gtype][post_gtype][key]
                        # Average the values and round to 3 decimal places
                        base_params[target_layer][pre_gtype][post_gtype][key] = round((val23 + val5) / 2.0, 3)

        # ==========================================
        # CONSTRUCT FINAL SYNPARAMS DICTIONARY
        # ==========================================
        synParams = {}
        layers = ['L23', 'L4', 'L5', 'L6']
        thalamic_gtypes = ['VPM', 'VPL', 'POm']

        # Pre-define parameter templates for cleanliness
        tc_params = {
            'tau_r': 0.2,
            'tau_d': 1.74,
            'Use': 0.72,
            'Dep': 227.0,
            'Fac': 14.0,
            'e': 0.0,          # Excitatory projection
            'gmax': 0.00072,   # 0.72 nS
            'syntype': 'ProbUDFsyn'
        }
        
        bg_inh_params = {
            'tau_r': 0.2, 
            'tau_d': 5.0, 
            'Use': 0.25, 
            'Dep': 700.0, 
            'Fac': 20.0,
            'e': -80.0, 
            'gmax': 0.00033, 
            'syntype': 'ProbUDFsyn'
        }

        # 1. Base Cortical Parameters
        for layer in layers:
            synParams[layer] = {}
            synParams[layer]['BG_exc'] = {}
            synParams[layer]['BG_inh'] = {}
            
            for pre_gtype in genetic_types:
                synParams[layer][pre_gtype] = {}
                
                for post_gtype in genetic_types:
                    bp = base_params[layer][pre_gtype][post_gtype]
                    
                    if pre_gtype == 'PY':
                        params = {
                            'tau_r_AMPA': bp['tau_r_AMPA'],
                            'tau_d_AMPA': bp['tau_d_AMPA'],
                            'tau_r_NMDA': bp['tau_r_NMDA'],
                            'tau_d_NMDA': bp['tau_d_NMDA'],
                            'Use': bp['Use'],
                            'Dep': bp['Dep'],
                            'Fac': bp['Fac'],
                            'e': 0.0,
                            'gmax': bp['gmax'],
                            'weight_factor_NMDA': 1.0,
                            'syntype': 'ProbAMPANMDA'
                        }
                    else:
                        params = {
                            'tau_r': bp['tau_r'],
                            'tau_d': bp['tau_d'],
                            'Use': bp['Use'],
                            'Dep': bp['Dep'],
                            'Fac': bp['Fac'],
                            'e': -80.0,
                            'gmax': bp['gmax'],
                            'syntype': 'ProbUDFsyn'
                        }
                    
                    # Assign to the Layer -> Pre -> Post hierarchy
                    synParams[layer][pre_gtype][post_gtype] = params

            # ==========================================
            # 2. Add Thalamocortical Parameters (Model #25)
            # ==========================================
            for th_pre in thalamic_gtypes:
                synParams[layer][th_pre] = {}
                for post_gtype in genetic_types:
                    # Apply identical TC parameters to all post-synaptic targets
                    synParams[layer][th_pre][post_gtype] = tc_params

            # ==========================================
            # 3. Add Background Parameters
            # ==========================================
            for post_gtype in genetic_types:
                # Excitatory BG (Sharing TC mechanics)
                synParams[layer]['BG_exc'][post_gtype] = tc_params 
                
                # Inhibitory BG
                synParams[layer]['BG_inh'][post_gtype] = bg_inh_params

        self.synParams = synParams






        ####################################
        # EEG PROPERTIES            #
        ####################################

        # Radii for Brain, CSF, Skull, Scalp (in um)
        self.radii = [79000., 80000., 85000., 90000.] 

        # Conductivities for Brain, CSF, Skull, Scalp (in S/m)
        self.sigmas = [0.3, 1.5, 0.015, 0.3] 

        # The electrode MUST sit exactly on the scalp surface (radius = 90000 um)
        # Here, we place one electrode directly above the column at the top of the head (Cz)
        self.r_electrodes = np.array([[0.0, 0.0, 90000.0]])


  

        
        
        ####################################
        # CONNECTION PROPERTIES            #
        ####################################
                

        # mean dendritic delays for excitatory and inhibitory transmission (ms)
        self.delays = [1.5, 0.75] 

        # standard deviation relative to mean delays; former delay_rel
        self.delay_rel_sd = 0.5
        
      
        ####################################
        # CELL-TYPE PARAMETERS             #
        ####################################
        
















################################################################################################################


class point_neuron_network_params(general_params):
    def __init__(self):
        '''class point-neuron network parameters'''

        # inherit general params
        general_params.__init__(self)
        

    ####################################
    #                                  #
    #                                  #
    #     SIMULATION PARAMETERS        #
    #                                  #
    #                                  #
    #################################### 

        # use same number of threads as MPI COMM.size()
        self.total_num_virtual_procs = SIZE

        ####################################
        # RNG PROPERTIES                   #
        #################################### 

        # offset for RNGs
        self.seed_offset = 45


        ####################################
        # RECORDING PARAMETERS             #        
        ####################################

        self.to_memory = False

        self.overwrite_existing_files = True 

        # recording can either be done from a fraction of neurons in each population or from a fixed number

        # whether to record spikes from a fixed fraction of neurons in each population. 
        self.record_fraction_neurons_spikes = True 

        if self.record_fraction_neurons_spikes:
            self.frac_rec_spikes = 1.
        else:
            self.n_rec_spikes = 100 

        # whether to record membrane potentials from a fixed fraction of neurons in each population
        self.record_fraction_neurons_voltage = False 

        if self.record_fraction_neurons_voltage:
            self.frac_rec_voltage = 0.1 
        else:
            self.n_rec_voltage = 100 

        # whether to record weighted input spikes from a fixed fraction of neurons in each population
        self.record_fraction_neurons_input_spikes = False

        if self.record_fraction_neurons_input_spikes:
            self.frac_rec_input_spikes = 0.1 
        else:
            self.n_rec_input_spikes = 100 

        # number of recorded neurons for depth resolved input currents
        self.n_rec_depth_resolved_input = 0 
 
        # whether to write any recorded cortical spikes to file
        self.save_cortical_spikes = True 

        # whether to write any recorded membrane potentials to file
        self.save_voltages = True 

        # whether to record thalamic spikes
        self.record_thalamic_spikes = True 

        # whether to write any recorded thalamic spikes to file
        self.save_thalamic_spikes = True 

        # global ID file name
        self.GID_filename = 'population_GIDs.dat'

        # readout global ID file name
        self.readout_GID_filename = 'readout_GIDs.dat' 

        # stem for spike detector file labels
        self.spike_detector_label = 'spikes_'

        # stem for voltmeter file labels
        self.voltmeter_label = 'voltages_'

        # stem for thalamic spike detector file labels
        self.th_spike_detector_label = 'spikes_0'

        # stem for in-degree file labels
        self.in_degree_label = 'in_degrees_'

        # stem for file labels for in-degree from thalamus 
        self.th_in_degree_label = 'in_degrees_th_'

        # stem for weighted input spikes labels
        self.weighted_input_spikes_label = 'weighted_input_spikes_'


    ####################################
    #                                  #
    #                                  #
    #        MODEL PARAMETERS          #
    #                                  #
    #                                  #
    ####################################


        ####################################
        # SCALING                          #
        ####################################  
       
        # scaling parameter for population sizes
        self.area = 1.
        
        # preserve indegrees when downscaling
        self.preserve_K = False         
        

        ####################################
        # SINGLE NEURON PARAMS             #
        ####################################

        # neuron model
        self.neuron_model = '/iaf_psc_exp'      

        # mean of initial membrane potential (mV)
        self.Vm0_mean = -58.0

        # std of initial membrane potential (mV)
        self.Vm0_std = 10.0

        # mean of threshold potential (mV)
        self.V_th_mean = -50.

        # std of threshold potential (mV)
        self.V_th_std = 0.
        
        self.model_params = { 'tau_m': 10.,        # membrane time constant (ms)
                              'tau_syn_ex': 0.5,   # excitatory synaptic time constant (ms)
                              'tau_syn_in': 0.5,   # inhibitory synaptic time constant (ms)
                              't_ref': 2.,         # absolute refractory period (ms)
                              'E_L': -65.,         # resting membrane potential (mV)
                              'V_th': self.V_th_mean, # spike threshold (mV)
                              'C_m': 250.,         # membrane capacitance (pF)
                              'V_reset': -65.      # reset potential (mV)
                              } 
        


        ####################################
        # EXTERNAL INPUTS                  #
        ####################################

        #number of external inputs (Potjans-Diesmann model 2012)
        self.K_bg = [[1600,    # layer 23 e
                      1500],   # layer 23 i
                     [2100,    # layer 4 e
                      1900],   # layer 4 i
                     [2000,    # layer 5 e
                      1900],   # layer 5 i
                     [2900,    # layer 6 e
                      2100]]   # layer 6 i
        
        # rate of Poisson input at each external input synapse (spikess)
        self.bg_rate = 0.       

        # rate of equivalent input used for DC amplitude calculation,
        # set to zero if self.bg_rate > 0.
        self.bg_rate_dc = 8.

        # DC amplitude at each external input synapse (pA)  
        # to each neuron via 'dc_amplitude  = tau_syn_ex/1000*bg_rate*PSC_ext'
        self.dc_amplitude = self.model_params["tau_syn_ex"] * self.bg_rate_dc *\
                            self._compute_J()

        # mean EPSP amplitude (mV) for thalamic and non-thalamic external input spikes
        self.PSP_ext = 0.15 	       

        # mean delay of thalamic input (ms)
        self.delay_th = 1.5  	

        # standard deviation relative to mean delay of thalamic input
        self.delay_th_rel_sd = 0.5   


        ####################################
        # THALAMIC INPUT VERSIONS  	   #
        ####################################
  
        # off-option for start of thalamic input versions
        self.off = 100.*self.tstop

   
        ## poisson_generator (pure Poisson input)
        self.th_poisson_start = self.off  	# onset (ms)
        self.th_poisson_duration = 10.	        # duration (ms)
        self.th_poisson_rate = 120. 	        # rate (spikess)


        ## spike_generator 
        # Note: This can be used with a large Gaussian delay distribution in order to mimic a 
        #       Gaussian pulse packet which is different for each thalamic neuron
        self.th_spike_times = [499. + i*500 for i in range(10)]	# time of the thalamic pulses (ms), assume 1 ms delay


        ## sinusoidal_poisson_generator (oscillatory Poisson input)
        self.th_sin_start = self.off      	# onset (ms)
        self.th_sin_duration = 5000.  	        # duration (ms)
        self.th_sin_mean_rate = 30. 	        # mean rate (spikess)
        self.th_sin_fluc_rate = 30.  	        # rate modulation amplitude (spikess)
        self.th_sin_freq = 15. 	                # frequency of the rate modulation (Hz)
        self.th_sin_phase = 0.                  # phase of rate modulation (deg)


        ## Gaussian_pulse_packages
        self.th_gauss_times = [self.off]                # package center times
        self.th_gauss_num_spikes_per_packet = 1 	# number of spikes per packet
        self.th_gauss_sd = 5. 				# std of Gaussian pulse packet (ms^2)


        ####################################
        # SPATIAL ORGANIZATION             #
        ####################################
        ####################################

        # needed for spatially resolved input currents
        
        # number of layers TODO: find a better solution for that
        self.num_input_layers = 5


    def _compute_J(self):
        '''
        Compute the current amplitude corresponding to the exponential
        synapse model PSP amplitude
        
        Derivation using sympy:
        ::
            from sympy import *
            #define symbols
            t, tm, Cm, ts, Is, Vmax = symbols('t tm Cm ts Is Vmax')
            
            #assume zero delay, t >= 0
            #using eq. 8.10 in Sterrat et al
            V = tm*ts*Is*(exp(-t/tm) - exp(-t/ts)) / (tm-ts) / Cm
            print 'V = %s' % V
            
            #find time of V == Vmax
            dVdt = diff(V, t)
            print 'dVdt = %s' % dVdt
            
            [t] = solve(dVdt, t)
            print 't(t@dVdT==Vmax) = %s' % t
            
            #solve for Is at time of maxima
            V = tm*ts*Is*(exp(-t/tm) - exp(-t/ts)) / (tm-ts) / Cm
            print 'V(%s) = %s' % (t, V)
            
            [Is] = solve(V-Vmax, Is)
            print 'Is = %s' % Is
        
        resulting in:
        ::
            Cm*Vmax*(-tm + ts)/(tm*ts*(exp(tm*log(ts/tm)/(tm - ts))
                                     - exp(ts*log(ts/tm)/(tm - ts))))

        
        Latex source:
        ::
            J&=-\frac{C_\text{m} V_\text{PSP} (\tau_\text{m} - \tau_\text{syn})}{\tau_\text{m} \tau_\text{syn}(
            \exp\frac{\tau_\text{m} \ln(\tau_\text{syn}/\tau_\text{m})}{\tau_\text{m} - \tau_\text{syn}}
            -\exp\frac{\tau_\text{syn} \ln(\tau_\text{syn}/\tau_\text{m})}{\tau_\text{m} - \tau_\text{syn}})} \\
            I^\text{ext} &= \tau_\text{syn} \nu^\text{ext} J \\
            &=-\frac{\nu^\text{ext}C_\text{m} V_\text{PSP} (\tau_\text{m} - \tau_\text{syn})}{\tau_\text{m}(
            \exp\frac{\tau_\text{m} \ln(\tau_\text{syn}/\tau_\text{m})}{\tau_\text{m} - \tau_\text{syn}}
            -\exp\frac{\tau_\text{syn} \ln(\tau_\text{syn}/\tau_\text{m})}{\tau_\text{m} - \tau_\text{syn}})}
        
        '''
        #LIF params
        tm = self.model_params['tau_m']
        Cm = self.model_params['C_m']
        
        #synapse
        ts = self.model_params['tau_syn_ex']
        Vmax = self.PSP_e
        
        #max current amplitude
        J = Cm*Vmax*(-tm + ts)/(tm*ts*(np.exp(tm*np.log(ts/tm)/(tm - ts))
                                     - np.exp(ts*np.log(ts/tm)/(tm - ts))))
        
        #unit conversion pF*mV -> nA
        J *= 1E-3
        
        return J
 

class multicompartment_params(point_neuron_network_params):
    '''
    Inherited class defining additional attributes needed by e.g., the
    classes population.Population and
    population.DummyNetwork

    This class do not take any kwargs    

    '''
    def __init__(self):
        '''
        Inherited class defining additional attributes needed by e.g., the
        classes population.Population and
        population.DummyNetwork
        
        This class do not take any kwargs    
        
        '''
        
        # initialize parent classes
        point_neuron_network_params.__init__(self)
 

    ####################################
    #                                  #
    #                                  #
    #     SIMULATION PARAMETERS        #
    #                                  #
    #                                  #
    ####################################
       

        #######################################
        # PARAMETERS FOR LOADING NEST RESULTS #
        #######################################


        # parameters for class population.DummyNetwork class
        self.networkSimParams = {
            'simtime' :     self.tstop - self.tstart,
            'dt' :          self.dt,
            'spike_output_path' : self.spike_output_path,
            'label' :       'population_spikes',
            'ext' :         'gdf',
            'GIDs' : self.get_GIDs(),
            'Prepop' : self.Pre,
        }


        # Switch for current source density computations
        self.calculateCSD = True


    ####################################
    #                                  #
    #                                  #
    #        MODEL PARAMETERS          #
    #                                  #
    #                                  #
    ####################################

       
        ####################################
        # SCALING (VOLUME not density)     #
        ####################################  
           
        self.SCALING = 1.0
        
  
        ####################################
        # MORPHOLOGIES                     #
        ####################################

        # list of morphology files with default location, testing = True
        # will point to simplified morphologies
        testing = False
        if testing:
            self.PATH_m_y = os.path.join('morphologies', 'ballnsticks')
            self.m_y = [Y + '_' + y + '.hoc' for Y, y in self.mapping_Yy]
        else:
            self.PATH_m_y = os.path.join('morphologies', 'stretched')
            self.m_y = [
                'L23E_oi24rpy1.hoc',
                'L23I_oi38lbc1.hoc',
                'L23I_oi38lbc1.hoc',
            
                'L4E_53rpy1.hoc',
                'L4E_j7_L4stellate.hoc',
                'L4E_j7_L4stellate.hoc',
                'L4I_oi26rbc1.hoc',
                'L4I_oi26rbc1.hoc',
            
                'L5E_oi15rpy4.hoc',
                'L5E_j4a.hoc',
                'L5I_oi15rbc1.hoc',
                'L5I_oi15rbc1.hoc',
            
                'L6E_51-2a.CNG.hoc',
                'L6E_oi15rpy4.hoc',
                'L6I_oi15rbc1.hoc',
                'L6I_oi15rbc1.hoc',
                ]


        ####################################
        # CONNECTION WEIGHTS               #
        ####################################
        
        # compute the synapse weight from fundamentals of exp synapse LIF neuron
        self.J = self._compute_J()
        
        # set up matrix containing the synapse weights between any population X
        # and population Y, including exceptions for certain connections
        J_YX = np.zeros(self.C_YX.shape)
        J_YX += self.J
        J_YX[:, 2::2] *= self.g
        if hasattr(self, 'PSP_23e_4e'):
            J_YX[0, 3] *= self.PSP_23e_4e / self.PSP_e
        if hasattr(self, 'g_4e_4i'):
            J_YX[2, 4] *= self.g_4e_4i / self.g
        

        # extrapolate weights between populations X and
        # cell type y in population Y
        self.J_yX = {}
        for Y, y in self.mapping_Yy:
            [i] = np.where(np.array(self.Post) == Y)[0]
            self.J_yX.update({y : J_YX[i, ]})
        
    
        ####################################
        # GEOMETRY OF CORTICAL COLUMN      #
        ####################################
        
        # set the boundaries of each layer, L1->L6,
        # and mean depth of soma layers
        self.layerBoundaries = np.array([[     0.0,   -81.6],
                                          [  -81.6,  -587.1],
                                          [ -587.1,  -922.2],
                                          [ -922.2, -1170.0],
                                          [-1170.0, -1491.7]])
        
        # assess depth of each 16 subpopulation
        self.depths = self._calcDepths()
        
        # make a nice structure with data for each subpopulation
        self.Post_zip_list = list(zip(self.Post, self.m_y,
                            self.depths, self.N_y))



        ##############################################################
        # POPULATION PARAMS (cells, population, synapses, electrode) #
        ##############################################################


        # Global LFPy.Cell-parameters, by default shared between populations
        # Some passive parameters will not be fully consistent with LIF params
        self.cellParams = {
            'v_init' : self.model_params['E_L'],
            'cm' : 1.0,
            'Ra' : 150,
            'passive' : True,
            'passive_parameters' : dict(g_pas=1./(self.model_params['tau_m'] * 1E3), #assume cm=1
                                        e_pas=self.model_params['E_L']),
            'nsegs_method' : 'lambda_f',
            'lambda_f' : 100,
            'dt' : self.dt,
            'tstart' : self.tstart,
            'tstop' : self.tstop,
            'verbose' : False,
        }
        

        # layer specific LFPy.Cell-parameters as nested dictionary
        self.PostCellParams = self._yCellParams()
        
        
        # set the axis of which each cell type y is randomly rotated,
        # SS types and INs are rotated around both x- and z-axis
        # in the population class, while P-types are
        # only rotated around the z-axis
        self.rand_rot_axis = {}
        for y, _, _, _ in self.Post_zip_list:
            #identify pyramidal cell populations:
            if y.rfind('p') >= 0:
                self.rand_rot_axis.update({y : ['z']})
            else:
                self.rand_rot_axis.update({y : ['x', 'z']})
        
        
        # additional simulation kwargs, see LFPy.Cell.simulate() docstring
        self.simulationParams = {'rec_current_dipole_moment': True}
        
                
        # a dict setting the number of cells N_y and geometry
        # of cell type population y
        self.populationParams = {}
        for y, _, depth, N_y in self.Post_zip_list:
            self.populationParams.update({
                y : {
                    'number' : int(N_y*self.SCALING),
                    'radius' : np.sqrt(1000**2 / np.pi),
                    'z_max' : depth[0], # Upper boundary (e.g., -81.6)
                    'z_min' : depth[1], # Lower boundary (e.g., -587.1)
                    'min_cell_interdist' : 1.,            
                }
            })

        # Set up cell type specific synapse parameters in terms of synapse model
        # and synapse locations

        for y in self.Pre:


            if y.split('_')[-1] == 'exc' or y.split('_')[-1] == 'ss':

                section = ['dend']


                self.synParams[y].update({
                    'section' : section,
                })

            else:
                section = ['axon','dend','soma']


                self.synParams[y].update({
                    'section' : section,
                })


        # set up dictionary of synapse time constants specific to each
        # postsynaptic cell type and presynaptic population
        #self.tau_yX = {}
        #for y in self.Post:
        #    self.tau_yX.update({
        #        y : [self.model_params["tau_syn_in"] if 'I' in X else
        #             self.model_params["tau_syn_ex"] for X in self.Pre]
        #    })

        #synaptic delay parameters, loc and scale is mean and std for every
        #network population, negative values will be removed
        self.synDelayLoc, self.synDelayScale = self._synDelayParams()

         
        # Define electrode geometry corresponding to a laminar electrode,
        # where contact points have a radius r, surface normal vectors N,
        # and LFP calculated as the average LFP in n random points on
        # each contact. Recording electrode emulate NeuroNexus array,
        # contact 0 is superficial
        self.electrodeParams = {
            #contact locations:
            'x' : np.zeros(16),
            'y' : np.zeros(16),
            'z' : -np.mgrid[0:16] * 100,
            #extracellular conductivity:
            'sigma' : 0.3,
            #contact surface normals, radius, n-point averaging
            'N' : np.array([[1, 0, 0]]*16),
            'r' : 7.5,
            'n' : 50,
            'seedvalue' : None,
            #dendrite line sources, soma sphere source (Linden2014)
            'method' : 'soma_as_point',
            #no somas within the constraints of the "electrode shank":
            'r_z': np.array([[-1E199, -1600, -1550, 1E99],[0, 0, 10, 10]]),
        }
        
        
        #these variables will be saved to file for each cell and electrdoe object
        self.savelist = [
            'somav',
            'dt',
            'somapos',
            'x',
            'y',
            'z',
            'LFP',
            'CSD',
            'morphology',
            'current_dipole_moment',
            # 'EEG',
            'default_rotation',
            'electrodecoeff',
        ]
        
        
        #########################################
        # MISC                                  #
        #########################################
        
        #time resolution of downsampled data in ms
        self.dt_output = 2**-2

        #set fraction of neurons from population which LFP output is stored
        self.recordSingleContribFrac = 0.


    def get_GIDs(self):
        GIDs = {}
        ind = 1
        for i, (X, N_X) in enumerate(zip(self.Pre, self.N_X)):
            GIDs[X] = [ind, N_X]
            ind += N_X
        return GIDs


    def _synDelayParams(self):
        '''
        set up the detailed synaptic delay parameters,
        loc is mean delay,
        scale is std with low bound cutoff,
        assumes numpy.random.normal is used later
        '''
        delays = {}
        #mean delays
        loc = np.zeros((len(self.Post), len(self.Pre)))
        loc[:, 0] = self.delays[0]
        loc[:, 1::2] = self.delays[0]
        loc[:, 2::2] = self.delays[1]
        #standard deviations
        scale = loc * self.delay_rel_sd
        
        #prepare output
        delay_loc = {}
        for i, y in enumerate(self.Post):
            delay_loc.update({y : loc[i]})
        
        delay_scale = {}
        for i, y in enumerate(self.Post):
            delay_scale.update({y : scale[i]})
                
        return delay_loc, delay_scale


    def _calcDepths(self):
        '''
        Return the cortical depth boundaries [z_max, z_min] of each subpopulation's layer.
        '''
        # Extract the actual boundaries, skipping Layer 1 (index 0) 
        # since somas are not placed there in your model.
        layer_bounds = self.layerBoundaries[1:]

        depth_y = []
        for y in self.Post:
            if y in ['p23', 'b23', 'nb23']:
                depth_y.append(layer_bounds[0])  # Layer 2/3 boundaries
            elif y in ['p4', 'ss4(L23)', 'ss4(L4)', 'b4', 'nb4']:
                depth_y.append(layer_bounds[1])  # Layer 4 boundaries
            elif y in ['p5(L23)', 'p5(L56)', 'b5', 'nb5']:
                depth_y.append(layer_bounds[2])  # Layer 5 boundaries
            elif y in ['p6(L4)', 'p6(L56)', 'b6', 'nb6']:
                depth_y.append(layer_bounds[3])  # Layer 6 boundaries
            else:
                raise Exception('Error, revise parameters')
                
        return depth_y

    




    def _yCellParams(self):
        '''
        Return dict with parameters for each population.
        The main operation is filling in cell type specific morphology
        '''
        #cell type specific parameters going into LFPy.Cell        
        yCellParams = {}
        for layer, morpho, _, _ in self.Post_zip_list:
            yCellParams.update({layer : self.cellParams.copy()})
            yCellParams[layer].update({
                'morphology' : os.path.join(self.PATH_m_y, morpho),
            })
        return yCellParams
 


import os
import pickle
import numpy as np
from scipy import sparse
from scipy.spatial.distance import cdist # This fixes the NameError
import numpy as np
import math
class Connectomics:


    def __init__(self,connectomics_path='',connectomics_output='',NSyn_path='',SpanTree_path ='',thalamus_path= '',name_list = [],Calculate=True):
        # NSyn_path path to the synapse number distribution per morphological path.
        # SpanTree_path = path to the saved spanning tree distributon per macro-population
        # name_list list of macro populations' loading names

        self.connectomics_path = connectomics_path
        self.connectomics_output = connectomics_output
        self.dat_file_path = NSyn_path
        self.SpanTree_path = SpanTree_path
        self.name_list = name_list
        self.thalamus_path = thalamus_path


        # --------------------------------------
        # Construct the input dictionary

        input_dict = {
            'Layers': {
                'L1':  [-250.0, 0.0],
                'L23': [-1200.0, -250.0],
                'L4':  [-1580.0, -1200.0],
                'L5':  [-2175.0, -1580.0],
                'L6':  [-2770.0, -2175.0]
            },
            'Cells': {
                'L1':  {'inh': 5145.9},                        # L1 only has inhibitory cells
                'L23': {'exc': 21466.6+8927.3, 'inh': 11655.3 + 5118.9},
                'L4':  {'exc': 23201.8, 'inh': 5502.3},           # Your example format
                'L5':  {'Excitatory': 10297.6, 'Inhibitory': 3249.9}, # Explicit keys work too
                'L6':  {'exc': 9823.0, 'inh': 1427.2}
            },
            'Geometry': {
                'radius': 300.0 # micrometers
            }
        }


        




        if Calculate:


            
            
            # State variables
            self.bbp_results = None
            self.bbp_totals = None
            self.mtype_fast_lookup = None
            self.cell_mtypes = None
            self.cell_coords = None
            self.adj_matrix = None
            self.post_to_pre = None
            self.pre_to_post = None
            self.synapse_dict = None

            # Open the file in 'read-binary' mode and load the data
            full_path_conn = os.path.join(self.connectomics_path, 'conn.pkl')
            with open(full_path_conn, 'rb') as f:
                self.conn_data = pickle.load(f)

            # Open the file in 'read-binary' mode and load the data
            full_thalamus_path = os.path.join(self.thalamus_path, 'convergence_Th_S1.txt')
            
            self.convergence_Th = {'VPM_sTC': {}, 'VPL_sTC': {}, 'POm_sTC_s1': {}}
            # 1. Build the convergence dictionary from the text file
            convergence_Th = {
                'VPM_sTC': {},
                'VPL_sTC': {},
                'POm_sTC_s1': {}
            }

            with open(full_thalamus_path, 'r') as f:
                for line in f:
                    if not line.strip():
                        continue
                    # The file columns: Target_mtype, VPM_count, POm_count, Group
                    parts = line.split()
                    mtype = parts[0]
                    preFO = int(parts[1]) # First-Order (VPM/VPL)
                    preHO = int(parts[2]) # Higher-Order (POm)

                    # Populate the dictionary
                    convergence_Th['VPM_sTC'][mtype] = preFO
                    convergence_Th['VPL_sTC'][mtype] = preFO
                    convergence_Th['POm_sTC_s1'][mtype] = preHO


            self.convergence_Th = convergence_Th
            

            # Execute Pipeline
            self.get_lookup_table()
            self.calculate_bbp_relative_presences()
            self.get_ADJ()
            self.extract_connectivity_dicts()
            self.add_independent_thalamic_sources()
            self.add_independent_background_sources()
            self.validate_all_virtual_sources()
            self.extract_multapses()



    @property
    def get_ColumnProp(self):
        return self.input_dict
    



    @property
    def get_ConnectomicInfo(self):
        """
        Returns a dictionary containing the full state of the microcircuit.

        Returns:
            dict: A collection of data structures defining the circuit:
                - 'bbp_results': Hierarchical dict of m-type percentages and counts per layer.
                - 'bbp_totals': Total cell counts grouped by layer and biological type (Exc/Inh).
                - 'mtype_fast_lookup': Map of m-type strings to (Layer, BioType) tuples.
                - 'cell_mtypes': 1D array of morphological types for every generated neuron.
                - 'cell_gtypes' : 1D array of genetical types for every generated neuron.
                - 'cell_coords': 2D array of shape $(N, 3)$ containing XYZ coordinates in $\mu m$.
                - 'adj_matrix': Sparse CSR matrix representing the synaptic adjacency graph.
                - 'post_to_pre': Afferent dict mapping Post-synaptic IDs to lists of Pre-synaptic IDs.
                - 'pre_to_post': Efferent dict mapping Pre-synaptic IDs to lists of Post-synaptic IDs.
                - 'synapse_dict' : Afferent dict mapping Post-synaptic IDs to lists of Pre-synaptic IDs and number of synapses per connections
                - 'self.TreeDensity_load': Maps the macro-population to the loading path for the spanning tree densities.
        """
        return {
            'bbp_results': self.bbp_results,
            'bbp_totals': self.bbp_totals,
            'mtype_fast_lookup': self.mtype_fast_lookup,
            'cell_mtypes': self.cell_mtypes,
            'cell_coords': self.cell_coords,
            'cell_gtypes': self.cell_gtypes,
            'synapse_dict': self.synapse_dict,
            'adj_matrix': self.adj_matrix,
            'post_to_pre': self.post_to_pre,
            'pre_to_post': self.pre_to_post,
            'TreeDensity_load' : self.TreeDensity_load

        }
    



    import math

    def add_independent_background_sources(self):
        """
        Appends independent virtual background (BG) sources to the cortical network.
        Adds exactly 10 independent sources (5 excitatory, 5 inhibitory) to every 
        original cortical cell. Handles mtypes and gtypes as NumPy arrays.
        """
        mtypes = self.cell_mtypes
        gtypes = self.cell_gtypes 
        post_to_pre = self.post_to_pre 

        # Protect against adding background noise to Thalamic cells if this is 
        # called AFTER the thalamic function. We only want to target the original cortex.
        if hasattr(self, 'original_cortical_ids'):
            target_cells = self.original_cortical_ids
        else:
            target_cells = list(range(len(mtypes)))
            self.original_cortical_ids = target_cells

        # 1. Convert NumPy arrays to Python lists for efficient appending
        mtypes_list = mtypes.tolist() if isinstance(mtypes, np.ndarray) else list(mtypes)
        gtypes_list = gtypes.tolist() if isinstance(gtypes, np.ndarray) else list(gtypes)

        # 2. Find the starting ID for virtual cells
        current_virtual_id = len(mtypes_list)

        # Iterate only over the original cortical cells
        for post_id in target_cells:

            # Ensure the cell exists in the post_to_pre structure
            if post_id not in post_to_pre:
                post_to_pre[post_id] = []

            # --- Add 5 Excitatory Background Sources ---
            for _ in range(5):
                mtypes_list.append('BG_exc')
                gtypes_list.append('BG_exc')
                post_to_pre[post_id].append(current_virtual_id)
                current_virtual_id += 1
                
            # --- Add 5 Inhibitory Background Sources ---
            for _ in range(5):
                mtypes_list.append('BG_inh')
                gtypes_list.append('BG_inh')
                post_to_pre[post_id].append(current_virtual_id)
                current_virtual_id += 1

        # 3. Convert back to NumPy arrays
        self.cell_mtypes = np.array(mtypes_list)
        self.cell_gtypes = np.array(gtypes_list)
        self.post_to_pre = post_to_pre
        
        print(f"Added background sources. Network grew from {len(mtypes)} to {len(mtypes_list)} total units.")

    def add_independent_thalamic_sources(self):
        """
        Appends independent virtual thalamic sources to the cortical network.
        Handles mtypes and gtypes as NumPy arrays.

        
        """

        mtypes = self.cell_mtypes
        gtypes = self.cell_gtypes 
        post_to_pre = self.post_to_pre 
        convergence_data = self.convergence_Th


        self.original_cortical_ids = list(range(len(mtypes)))



        # 1. Convert NumPy arrays to Python lists for efficient appending
        mtypes_list = mtypes.tolist() if isinstance(mtypes, np.ndarray) else list(mtypes)
        gtypes_list = gtypes.tolist() if isinstance(gtypes, np.ndarray) else list(gtypes)

        # 2. Find the starting ID for virtual cells
        # In an array, the cell IDs are exactly the indices (0 to N-1)
        # Therefore, the next available ID is simply the current length of the array
        current_virtual_id = len(mtypes_list)
        original_num_cells = len(mtypes_list)

        synapses_per_source = 9.0

        # Iterate only over the original cortical cells (from index 0 to original_num_cells - 1)
        for post_id in range(original_num_cells):
            cortical_mtype = mtypes_list[post_id]

            # Ensure the cell exists in the post_to_pre structure
            if post_id not in post_to_pre:
                post_to_pre[post_id] = []

            # Add VPM, VPL, and POm sources
            for th_pop in ['VPM', 'VPL', 'POm']:

                # Map the clean label to the dictionary keys
                th_key = th_pop + '_sTC' if th_pop in ['VPM', 'VPL'] else 'POm_sTC_s1'

                if th_key not in convergence_data or cortical_mtype not in convergence_data[th_key]:
                    continue

                raw_target = convergence_data[th_key][cortical_mtype]
                if raw_target <= 0:
                    continue

                # Calculate how many independent virtual sources this cell needs
                required_sources = int(np.ceil(raw_target / synapses_per_source))

                # Generate the specific unique IDs for these sources
                for _ in range(required_sources):
                    # Append the generic 'TC' label
                    mtypes_list.append('TC')

                    # Append the specific thalamic nucleus
                    gtypes_list.append(th_pop)

                    # Attach the new pre-synaptic ID to the cortical neuron
                    post_to_pre[post_id].append(current_virtual_id)

                    # Increment the global counter to guarantee independence
                    current_virtual_id += 1

        # 3. Convert back to NumPy arrays before returning
        self.cell_mtypes = np.array(mtypes_list)
        self.cell_gtypes = np.array(gtypes_list)
        self.post_to_pre = post_to_pre

         
        
    def calculate_bbp_relative_presences(self):
        """
        Parses the BBP S1 distribution file and calculates the relative percentage
        of each m-type among all neurons of the same biological type in its layer.
        """

        # Expanded acronym map to include L1-specific inhibitory cells from the BBP data
        acronym_map = {
            # Excitatory Types
            'PC': 'Excitatory', 'SS': 'Excitatory', 'SP': 'Excitatory',
            'TTPC1': 'Excitatory', 'TTPC2': 'Excitatory', 'STPC': 'Excitatory',
            'UTPC': 'Excitatory', 'BPC': 'Excitatory', 'IPC': 'Excitatory',
            'TPC_L1': 'Excitatory', 'TPC_L4': 'Excitatory',

            # Inhibitory Types
            'LBC': 'Inhibitory', 'NBC': 'Inhibitory', 'SBC': 'Inhibitory',
            'ChC': 'Inhibitory', 'MC': 'Inhibitory', 'BTC': 'Inhibitory',
            'DBC': 'Inhibitory', 'BP': 'Inhibitory', 'NGC': 'Inhibitory',
            'HAC': 'Inhibitory', 'DAC': 'Inhibitory', 'SAC': 'Inhibitory',

            # L1 Specific Inhibitory Types
            'NGC-DA': 'Inhibitory', 'NGC-SA': 'Inhibitory',
            'DLAC': 'Inhibitory', 'SLAC': 'Inhibitory'
        }


        file_path='S1-cells-distributions-Rat.txt'
        full_path_conn = os.path.join(self.connectomics_path, file_path)



        if not os.path.exists(full_path_conn):
            print(f"Error: Could not find {full_path_conn}")
            return None
        
    

        mtype_raw_counts = {}

        # 1. Parse the BBP text file
        with open(full_path_conn, 'r') as f:
            for line in f:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 5:
                        mtype = parts[1]
                        m_count = int(parts[4]) # Column 5 'm' is the total m-type count

                        # Since the file lists multiple e-types per m-type,
                        # we just overwrite with the same 'm_count' total for that m-type
                        mtype_raw_counts[mtype] = m_count

        # 2. Group into layers and sum up the totals for Excitatory/Inhibitory
        layer_totals = {}
        composition = {}

        for mtype, count in mtype_raw_counts.items():
            mtype_parts = mtype.split('_', 1)
            layer = mtype_parts[0]
            acronym = mtype_parts[1] if len(mtype_parts) > 1 else mtype

            # Handle the TPC exceptions (e.g. 'TPC_L1', 'TPC_L4')
            if acronym.startswith('TPC'):
                acronym = mtype.replace(f"{layer}_", "")

            bio_type = acronym_map.get(acronym, 'Unknown')

            if layer not in composition:
                composition[layer] = {'Excitatory': {}, 'Inhibitory': {}}
                layer_totals[layer] = {'Excitatory': 0, 'Inhibitory': 0}

            if bio_type in ['Excitatory', 'Inhibitory']:
                composition[layer][bio_type][mtype] = count
                layer_totals[layer][bio_type] += count

        # 3. Calculate the relative percentages
        results = {}
        for layer in composition:
            results[layer] = {'Excitatory': {}, 'Inhibitory': {}}

            for bio_type in ['Excitatory', 'Inhibitory']:
                total_cells = layer_totals[layer][bio_type]

                for mtype, count in composition[layer][bio_type].items():
                    perc = (count / total_cells * 100) if total_cells > 0 else 0.0
                    results[layer][bio_type][mtype] = {
                        'count': count,
                        'percentage': perc
                    }


        self.bbp_results = results
        self.bbp_totals = layer_totals

        if results:
            print("BBP ORIGINAL MICROCIRCUIT COMPOSITION")
            print("="*45)

            # Sort layers (L1, L23, L4, L5, L6)
            for layer in sorted(results.keys()):
                print(f"\n--- {layer} ---")

                for bio_type in ['Excitatory', 'Inhibitory']:
                    total = layer_totals[layer][bio_type]
                    if total > 0:
                        print(f"  {bio_type} (Total: {total}):")

                        # Sort m-types by percentage (highest to lowest)
                        mtypes_data = results[layer][bio_type]
                        sorted_mtypes = sorted(mtypes_data.items(), key=lambda item: item[1]['percentage'], reverse=True)

                        for mtype, data in sorted_mtypes:
                            perc = data['percentage']
                            count = data['count']
                            print(f"    • {mtype:<12}: {perc:>6.2f}%  ({count} cells)")

 
        
        
        
        
        
        
    def get_ADJ(self):

        self.generate_microcolumn_cells()
        self.build_adjacency_matrix()

       



    def extract_multapses(self):
        """
        Constructs a dictionary mapping post-synaptic IDs to an Nx2 matrix of 
        pre-synaptic IDs and their calculated multapse counts, with macro-population
        averaging for missing structural pathways.
        """
        dat_file_path = self.dat_file_path
        post_to_pre   = self.post_to_pre
        cell_mtypes   = self.cell_mtypes
        mtype_fast_lookup = self.mtype_fast_lookup  # Used to group cells into macro-populations
        
         # 1. Parse the synNumberperconex.dat file as a text file
        file_path = 'synNumberperconex.dat'
        full_path_conn = os.path.join(dat_file_path, file_path)
        
        from collections import defaultdict
        
        # Nested dictionary to store specific mean and std
        syn_stats = defaultdict(dict)
        
        with open(full_path_conn, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 6:
                    mean_val = float(parts[2])
                    std_val = float(parts[3])
                    proj = parts[5]
                    
                    if ':' in proj:
                        pre_mtype_str, post_mtype_str = proj.split(':')
                        syn_stats[pre_mtype_str][post_mtype_str] = {
                            'mean': mean_val, 
                            'std': std_val
                        }

        # ---------------------------------------------------------
        # NEW: Calculate Macro-Population Averages for Imputation
        # ---------------------------------------------------------
        # Accumulator: Key = ((PreLayer, PreBio), (PostLayer, PostBio))
        macro_stats_accumulator = defaultdict(lambda: {'sum_mean': 0.0, 'sum_std': 0.0, 'count': 0})
        
        for pre_m, post_dict in syn_stats.items():
            pre_macro = mtype_fast_lookup.get(pre_m)
            if not pre_macro: continue # Skip if mtype isn't in our circuit
            
            for post_m, stats in post_dict.items():
                post_macro = mtype_fast_lookup.get(post_m)
                if not post_macro: continue
                
                macro_key = (pre_macro, post_macro)
                macro_stats_accumulator[macro_key]['sum_mean'] += stats['mean']
                macro_stats_accumulator[macro_key]['sum_std'] += stats['std']
                macro_stats_accumulator[macro_key]['count'] += 1
                
        # Calculate final averages
        macro_averages = {}
        for macro_key, acc in macro_stats_accumulator.items():
            if acc['count'] > 0:
                macro_averages[macro_key] = {
                    'mean': acc['sum_mean'] / acc['count'],
                    'std': acc['sum_std'] / acc['count']
                }
        # ---------------------------------------------------------

        multapse_dict = {}

        for post_idx, pre_indices in post_to_pre.items():
            if not pre_indices:
                continue
                
            post_mtype = cell_mtypes[post_idx]
            post_macro = mtype_fast_lookup.get(post_mtype)
            
            pre_array = np.array(pre_indices, dtype=int)
            pre_mtypes = cell_mtypes[pre_array]
            
            n_pre = len(pre_array)
            means = np.zeros(n_pre)
            stds = np.zeros(n_pre)
            
            # Populate means and stds based on the pathway
            for i, pre_mtype in enumerate(pre_mtypes):
                
                # ---> NEW CHANGE: Intercept Thalamic Sources <---
                if pre_mtype == 'TC':
                    means[i] = 9.0
                    stds[i] = 3.0
                    continue
                # ------------------------------------------------

                # ---> NEW CHANGE: Intercept Background Sources <---
                if pre_mtype == 'BG':
                    means[i] = 1.0
                    stds[i] = 0.0
                    continue
                # ------------------------------------------------
                
                try:
                    # Strategy A: Exact structural pathway match
                    means[i] = syn_stats[pre_mtype][post_mtype]['mean']
                    stds[i] = syn_stats[pre_mtype][post_mtype]['std']
                    
                except KeyError:
                    # Strategy B: Impute using macro-population average
                    pre_macro = mtype_fast_lookup.get(pre_mtype)
                    macro_key = (pre_macro, post_macro)
                    
                    if macro_key in macro_averages:
                        means[i] = macro_averages[macro_key]['mean']
                        stds[i] = macro_averages[macro_key]['std']
                    else:
                        # Strategy C: Absolute last resort to prevent crashes
                        means[i] = 1.0
                        stds[i] = 0.0
                    
            # 2. Draw from the Gaussian distribution
            sampled_synapses = np.random.normal(loc=means, scale=stds)
            
            # 3. Clean the data (ensure at least 1 synapse if connected)
            n_synapses = np.maximum(1, np.round(sampled_synapses)).astype(int)
            
            # 4. Construct the N_pre x 2 matrix
            result_matrix = np.column_stack((pre_array, n_synapses))
            
            multapse_dict[post_idx] = result_matrix

        self.synapse_dict = multapse_dict

    def validate_all_virtual_sources(self):
        """
        Generalized validation function that checks the integrity of the network 
        AFTER both Thalamic (TC) and Background (BG) virtual sources have been added.
        Handles mtypes and gtypes as NumPy arrays.
        """


        original_cortical_ids = self.original_cortical_ids
        mtypes = self.cell_mtypes
        gtypes = self.cell_gtypes 
        post_to_pre = self.post_to_pre 
        convergence_data = self.convergence_Th




        errors = []
    
        # Expected Constants based on your pipeline
        synapses_per_source = 9.0
        expected_bg_exc = 5
        expected_bg_inh = 5
        
        # 1. Check Strict Independence (Across EVERYTHING)
        all_assigned_virtual_ids = []
        for post_id in original_cortical_ids:
            pre_list = post_to_pre.get(post_id, [])
            
            # A virtual ID is any ID generated AFTER the original cortical network
            virtual_pres = [
                pre_id for pre_id in pre_list 
                if pre_id >= len(original_cortical_ids)
            ]
            all_assigned_virtual_ids.extend(virtual_pres)
                
        # If there's a mismatch, it means two cortical cells share a virtual source
        if len(all_assigned_virtual_ids) != len(set(all_assigned_virtual_ids)):
            errors.append("INDEPENDENCE BUG: Some cortical cells are sharing the same virtual source ID (TC or BG collision).")
            
        # 2. Check Dictionary/Array Consistency
        for v_id in all_assigned_virtual_ids:
            if v_id >= len(mtypes) or v_id >= len(gtypes):
                errors.append(f"CONSISTENCY BUG: Virtual ID {v_id} is out of bounds for the arrays.")
                continue
                
            mtype_val = str(mtypes[v_id])
            gtype_val = str(gtypes[v_id])
            
            # Route checking for Thalamic Cells
            if gtype_val in ['VPM', 'VPL', 'POm']:
                if mtype_val != 'TC':
                    errors.append(f"CONSISTENCY BUG: ID {v_id} is {gtype_val} but has mtype '{mtype_val}' instead of 'TC'.")
                    
            # Route checking for Background Cells
            elif gtype_val in ['BG_exc', 'BG_inh']:
                if mtype_val != 'BG':
                    errors.append(f"CONSISTENCY BUG: ID {v_id} is {gtype_val} but has mtype '{mtype_val}' instead of 'BG'.")
                    
            else:
                errors.append(f"CONSISTENCY BUG: ID {v_id} has an unrecognized/invalid gtype '{gtype_val}'.")

        # 3. Check Mathematical Accuracy (Convergence)
        error_limit = 15  
        math_errors = 0
        
        for post_id in original_cortical_ids:
            cortical_mtype = str(mtypes[post_id])
            
            # Count actual assigned sources for this specific cortical cell
            actual_counts = {'VPM': 0, 'VPL': 0, 'POm': 0, 'BG_exc': 0, 'BG_inh': 0}
            
            for pre_id in post_to_pre.get(post_id, []):
                if pre_id >= len(original_cortical_ids) and pre_id < len(gtypes):
                    nuc = str(gtypes[pre_id])
                    if nuc in actual_counts:
                        actual_counts[nuc] += 1
                    
            # Calculate EXPECTED Thalamic sources
            for th_pop in ['VPM', 'VPL', 'POm']:
                th_key = th_pop + '_sTC' if th_pop in ['VPM', 'VPL'] else 'POm_sTC_s1'
                
                if th_key in convergence_data and cortical_mtype in convergence_data[th_key]:
                    raw_target = convergence_data[th_key][cortical_mtype]
                    expected_count = int(np.ceil(max(0, raw_target) / synapses_per_source))
                else:
                    expected_count = 0
                    
                if actual_counts[th_pop] != expected_count:
                    math_errors += 1
                    if math_errors <= error_limit:
                        errors.append(f"MATH BUG for Cell {post_id} ({cortical_mtype}): "
                                    f"Expected {expected_count} {th_pop} sources, but found {actual_counts[th_pop]}.")
            
            # Calculate EXPECTED Background sources
            if actual_counts['BG_exc'] != expected_bg_exc:
                math_errors += 1
                if math_errors <= error_limit:
                    errors.append(f"MATH BUG for Cell {post_id} ({cortical_mtype}): "
                                f"Expected {expected_bg_exc} BG_exc sources, found {actual_counts['BG_exc']}.")
                                
            if actual_counts['BG_inh'] != expected_bg_inh:
                math_errors += 1
                if math_errors <= error_limit:
                    errors.append(f"MATH BUG for Cell {post_id} ({cortical_mtype}): "
                                f"Expected {expected_bg_inh} BG_inh sources, found {actual_counts['BG_inh']}.")
        
        if math_errors > error_limit:
            errors.append(f"... and {math_errors - error_limit} more math errors suppressed.")

        # --- Final Report ---
        if not errors:
            print("✅ FULL VALIDATION PASSED: All Thalamic and Background sources are mathematically accurate, perfectly independent, and properly typed.")

        else:
            print("❌ VALIDATION FAILED with the following hidden bugs:")
            for error in errors:
                print(f"  - {error}")


    def extract_connectivity_dicts(self):
        """
        Extracts pre-to-post and post-to-pre connectivity dictionaries from a sparse adjacency matrix.
        Optionally saves them as high-efficiency pickle files if an output folder is provided.

        Parameters:
        - adj_matrix: scipy.sparse.csr_matrix (Shape: N x N)
        - output_folder: str, path to the directory where files should be saved (optional).

        Returns:
        - post_to_pre: Dict where Key = Post-synaptic ID, Value = List of Pre-synaptic IDs
        - pre_to_post: Dict where Key = Pre-synaptic ID, Value = List of Post-synaptic IDs
        """

        
        adj_matrix = self.adj_matrix
        output_folder = self.connectomics_output
        

        N = adj_matrix.shape[0]

        # 1. Pre-synaptic focus (Outputs / Efferent connections)
        pre_to_post = {}
        for i in range(N):
            start_idx = adj_matrix.indptr[i]
            end_idx = adj_matrix.indptr[i+1]

            if start_idx != end_idx:
                pre_to_post[i] = adj_matrix.indices[start_idx:end_idx].tolist()

        # 2. Post-synaptic focus (Inputs / Afferent connections)
        csc_matrix = adj_matrix.tocsc()
        post_to_pre = {}
        for j in range(N):
            start_idx = csc_matrix.indptr[j]
            end_idx = csc_matrix.indptr[j+1]

            if start_idx != end_idx:
                post_to_pre[j] = csc_matrix.indices[start_idx:end_idx].tolist()



        # 3. Save to disk if requested
        if output_folder:
            # Ensure the target directory exists
            os.makedirs(output_folder, exist_ok=True)

            pre_to_post_path = os.path.join(output_folder, 'pre_to_post.pkl')
            post_to_pre_path = os.path.join(output_folder, 'post_to_pre.pkl')

            # Save using the highest protocol for maximum compression and speed
            with open(pre_to_post_path, 'wb') as f:
                pickle.dump(pre_to_post, f, protocol=pickle.HIGHEST_PROTOCOL)

            with open(post_to_pre_path, 'wb') as f:
                pickle.dump(post_to_pre, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"Dictionaries successfully saved to:")
            print(f" - {pre_to_post_path}")
            print(f" - {post_to_pre_path}")


        self.post_to_pre = post_to_pre
        self.pre_to_post = pre_to_post

    

    def build_adjacency_matrix(self):

        print("="*40)
        print('BUILDING ADJ MATRIX')

        """
        Optimized builder for massive 3D microcircuits using sparse matrices.
        """

        cell_mtypes = self.cell_mtypes
        cell_coords = self.cell_coords 
        conn_data = self.conn_data




        N = len(cell_mtypes)
        unique_mtypes = np.unique(cell_mtypes)

        all_rows = []
        all_cols = []

        mtype_indices = {m: np.where(cell_mtypes == m)[0] for m in unique_mtypes}

        for pre_mtype in unique_mtypes:
            pre_idx = mtype_indices[pre_mtype]
            coords_pre = cell_coords[pre_idx]

            for post_mtype in unique_mtypes:
                if post_mtype not in conn_data.get('best_fit', {}).get(pre_mtype, {}):
                    continue

                post_idx = mtype_indices[post_mtype]
                coords_post = cell_coords[post_idx]
                d_sub = cdist(coords_pre, coords_post, metric='euclidean')
                fit_type = conn_data['best_fit'][pre_mtype][post_mtype]

                if fit_type == 'exp':
                    a0 = float(conn_data['a0mat_exp'][pre_mtype][post_mtype])
                    l = float(conn_data['lmat_exp'][pre_mtype][post_mtype])
                    d0 = float(conn_data['d0_exp'][pre_mtype][post_mtype])
                    p_sub = a0 * np.exp(-np.maximum(d_sub - d0, 0) / l)

                elif fit_type == 'gauss':
                    a0 = float(conn_data['a0mat_gauss'][pre_mtype][post_mtype])
                    l = float(conn_data['lmat_gauss'][pre_mtype][post_mtype])
                    x0 = float(conn_data['x0_gauss'][pre_mtype][post_mtype])
                    p_sub = a0 * np.exp(-((d_sub - x0)**2) / (l**2))

                else:
                    p_sub = np.zeros_like(d_sub)
                    dist_bins = [
                        (12.5, 'pmat12um'), (25.0, 'pmat25um'), (50.0, 'pmat50um'),
                        (75.0, 'pmat75um'), (100.0, 'pmat100um'), (125.0, 'pmat125um'),
                        (150.0, 'pmat150um'), (175.0, 'pmat175um'), (200.0, 'pmat200um'),
                        (225.0, 'pmat225um'), (250.0, 'pmat250um'), (275.0, 'pmat275um'),
                        (300.0, 'pmat300um'), (325.0, 'pmat325um'), (350.0, 'pmat350um'),
                        (375.0, 'pmat375um')
                    ]
                    prev_d = -1.0
                    for max_d, bin_key in dist_bins:
                        try:
                            bin_prob = float(conn_data[bin_key][pre_mtype][post_mtype])
                        except (KeyError, ValueError, TypeError):
                            bin_prob = 0.0
                        mask = (d_sub > prev_d) & (d_sub <= max_d)
                        p_sub[mask] = bin_prob
                        prev_d = max_d

                trial = np.random.rand(*p_sub.shape) < p_sub
                local_rows, local_cols = np.where(trial)

                if len(local_rows) > 0:
                    global_rows = pre_idx[local_rows]
                    global_cols = post_idx[local_cols]

                    mask_autapse = global_rows != global_cols
                    all_rows.extend(global_rows[mask_autapse])
                    all_cols.extend(global_cols[mask_autapse])

                del d_sub, p_sub, trial

        data = np.ones(len(all_rows), dtype=np.uint8)
        adj_matrix = sparse.csr_matrix((data, (all_rows, all_cols)), shape=(N, N))

        # ==========================================
        # ASSIGN GENETIC MARKERS BASED ON CONNECTIVITY
        # ==========================================
        cell_genetics = np.empty(N, dtype=object)

        # 1. Create a boolean mask of which cells are Excitatory
        is_exc = np.array([self.mtype_fast_lookup[m][1] == 'Excitatory' for m in cell_mtypes])
        is_inh = ~is_exc

        # Excitatory cells are 'PY'
        cell_genetics[is_exc] = 'PY'

        # 2. Count how many Excitatory targets each cell connects to
        # adj_matrix dot product with the boolean 'is_exc' array calculates 
        # the sum of excitatory connections along each row efficiently.
        targets_exc_count = adj_matrix.dot(is_exc.astype(int))

        # 3. Inhibitory cells targeting NO excitatory cells -> VIP
        is_vip = is_inh & (targets_exc_count == 0)
        cell_genetics[is_vip] = 'VIP'

        # 4. Inhibitory cells targeting at least 1 excitatory cell -> SST (3/5) or PV (2/5)
        is_sst_pv = is_inh & (targets_exc_count > 0)
        num_sst_pv = is_sst_pv.sum()
        
        if num_sst_pv > 0:
            cell_genetics[is_sst_pv] = np.random.choice(
                ['SST', 'PV'], 
                size=num_sst_pv, 
                p=[3/5, 2/5]
            )

        self.cell_gtypes = cell_genetics
        self.adj_matrix = adj_matrix



    def generate_microcolumn_cells(self, verbose=True):

        print("="*40)
        print('PLACEMENT OF CELLS')


        """
        Generates cell m-types and 3D coordinates based on top-level densities
        and biological percentage breakdowns.

        Parameters:
        - input_dict: Dictionary containing 'Layers' (Z bounds),
                    'Cells' (densities grouped by Layer -> BioType),
                    and 'Geometry' (radius).
        - bbp_results: Nested dictionary mapping Layer -> BioType -> MType -> %
        - verbose: Boolean. If True, prints a detailed breakdown of generated cells.
        - output_folder: String (optional). Path to save the generated arrays.

        Returns:
        - cell_mtypes: 2D numpy array of shape (N, 2) containing (m-type, genetic_type).
        - cell_coords: 2D numpy array of shape (N, 3) containing (x, y, z).
        """
        input_dict = self.input_dict
        bbp_results = self.bbp_results
        output_folder = self.connectomics_output
        radius = input_dict['Geometry']['radius']

        base_mtypes_list = []
        coords_list = []

        area = np.pi * (radius ** 2)

        if verbose:
            print("BUILDING MICROCIRCUIT...")
            print("="*40)

        for layer, bio_types in input_dict['Cells'].items():
            if layer not in input_dict['Layers']:
                if verbose: print(f"Warning: Z-boundaries for '{layer}' not defined. Skipping.")
                continue

            bounds = input_dict['Layers'][layer]
            z_min, z_max = min(bounds), max(bounds)
            layer_height = z_max - z_min
            volume_um3 = area * layer_height

            if verbose:
                print(f"\n--- {layer} (Height: {layer_height} um) ---")

            for raw_bio_type, density_mm3 in bio_types.items():
                if density_mm3 <= 0:
                    continue

                bio_type = 'Inhibitory' if raw_bio_type.lower().startswith('inh') else 'Excitatory'
                density_um3 = density_mm3 / 1e9
                total_group_count = int(np.round(density_um3 * volume_um3))

                if total_group_count <= 0:
                    continue

                if layer not in bbp_results or bio_type not in bbp_results[layer]:
                    if verbose: print(f"  Warning: BBP results for {layer} {bio_type} not found. Skipping.")
                    continue

                if verbose:
                    print(f"  {bio_type} (Target Total: ~{total_group_count} cells):")

                mtype_data = bbp_results[layer][bio_type]

                for mtype, data in mtype_data.items():
                    percentage = data['percentage']
                    count = int(np.round(total_group_count * (percentage / 100.0)))

                    if count <= 0:
                        continue

                    if verbose:
                        print(f"    -> {mtype:<12}: {count:>5} cells ({percentage:>5.2f}%)")

                    r_random = radius * np.sqrt(np.random.rand(count))
                    theta_random = np.random.rand(count) * 2 * np.pi

                    x = r_random * np.cos(theta_random)
                    y = r_random * np.sin(theta_random)
                    z = np.random.uniform(z_min, z_max, count)

                    coords = np.column_stack((x, y, z))
                    coords_list.append(coords)

                    base_mtypes = np.full(count, mtype, dtype=object)
                    base_mtypes_list.append(base_mtypes)

        if base_mtypes_list:
            cell_mtypes = np.concatenate(base_mtypes_list)
            cell_coords = np.vstack(coords_list)
            if verbose:
                print("="*40)
                print(f"SUCCESS: Generated {len(cell_mtypes)} total neurons.")
        else:
            cell_mtypes = np.array([], dtype=object)
            cell_coords = np.empty((0, 3))
            if verbose:
                print("WARNING: No cells were generated. Check your input dictionary.")

        if output_folder:
            os.makedirs(output_folder, exist_ok=True)
            mtypes_path = os.path.join(output_folder, 'cell_mtypes.npy')
            coords_path = os.path.join(output_folder, 'cell_coords.npy')

            np.save(mtypes_path, cell_mtypes)
            np.save(coords_path, cell_coords)

            if verbose:
                print(f"Data saved to: {output_folder}")
        # ----------------------------------------------------------------
        self.cell_mtypes = cell_mtypes
        self.cell_coords = cell_coords







    def get_lookup_table(self):

        print("="*40)
        print('STARTING THE GENERATION OF THE LOOKUP TABLE')
        # Master Dictionary mapping acronyms to (Full Name, Type)

        acronym_map = {
            # Excitatory Types
            'PC': ('Pyramidal Cell', 'Excitatory'),
            'SS': ('Spiny Stellate', 'Excitatory'),
            'SP': ('Star Pyramidal', 'Excitatory'),
            'TTPC1': ('Thick Tufted Pyramidal Cell 1', 'Excitatory'),
            'TTPC2': ('Thick Tufted Pyramidal Cell 2', 'Excitatory'),
            'STPC': ('Slender Tufted Pyramidal Cell', 'Excitatory'),
            'UTPC': ('Untufted Pyramidal Cell', 'Excitatory'),
            'BPC': ('Bipolar Pyramidal Cell', 'Excitatory'),
            'IPC': ('Inverted Pyramidal Cell', 'Excitatory'),
            'TPC_L1': ('Tufted Pyramidal Cell (L1)', 'Excitatory'),
            'TPC_L4': ('Tufted Pyramidal Cell (L4)', 'Excitatory'),

            # Standard Inhibitory Types
            'LBC': ('Large Basket Cell', 'Inhibitory'),
            'NBC': ('Nest Basket Cell', 'Inhibitory'),
            'SBC': ('Small Basket Cell', 'Inhibitory'),
            'ChC': ('Chandelier Cell', 'Inhibitory'),
            'MC': ('Martinotti Cell', 'Inhibitory'),
            'BTC': ('Bitufted Cell', 'Inhibitory'),
            'DBC': ('Double Bouquet Cell', 'Inhibitory'),
            'BP': ('Bipolar Cell', 'Inhibitory'),
            'NGC': ('Neurogliaform Cell', 'Inhibitory'),
            'HAC': ('Horizontal Axon Cell', 'Inhibitory'),
            'DAC': ('Descending Axon Cell', 'Inhibitory'),
            'SAC': ('Small Axon Cell', 'Inhibitory'),
            
            # --- ADDED: L1 Specific Inhibitory Types ---
            'NGC-DA': ('Neurogliaform Cell with Dense Axonal Arborization', 'Inhibitory'),
            'NGC-SA': ('Neurogliaform Cell with Sparse Axonal Arborization', 'Inhibitory'),
            'DLAC': ('Deiters-Like Axon Cell', 'Inhibitory'),
            'SLAC': ('Single-Layer Axon Cell', 'Inhibitory')
        }

        file_path = 'S1-cells-distributions-Rat.txt'
        valid_layers = {"L1", "L23", "L4", "L5", "L6"}

        # Initialize the fast-readable variable
        mtype_fast_lookup = {}

        if not os.path.exists(file_path):
            print(f"Error: '{file_path}' not found. Please upload it to the 'anatomy' folder.")
        else:
            with open(file_path, 'r') as f:
                for line in f.read().split('\n'):
                    if line.strip():
                        parts = line.split()
                        if len(parts) >= 5:
                            metype, mtype, etype, n, m = parts[:5]

                            # Split the mtype to determine layer and acronym
                            mtype_parts = mtype.split('_', 1)
                            target_layer = mtype_parts[0]
                            acronym = mtype_parts[1] if len(mtype_parts) > 1 else mtype

                            # Handle acronym exceptions
                            if acronym == 'TPC' and len(mtype_parts) > 2:
                                acronym = f"TPC_{mtype_parts[2]}"

                            # Look up the biological type
                            _, bio_type = acronym_map.get(acronym, (acronym, 'Unknown'))

                            # If valid, populate the fast lookup dictionary directly
                            if target_layer in valid_layers and bio_type in ["Excitatory", "Inhibitory"]:
                                mtype_fast_lookup[mtype] = (target_layer, bio_type)


            self.mtype_fast_lookup = mtype_fast_lookup


        







    


    


# 1. The Updated Extraction Function (Now includes 'cell_mtypes' and 'cell_gtypes')
def extract_macro_populations(connectomics_data, name_list):
    """
    Extracts specific macro-populations using the fast lookup table,
    returning only spatial coordinates, mapping indices, the integrated synapse dictionary,
    and the genetic types.
    """
    # Unpack only the necessary data
    mtype_fast_lookup = connectomics_data['mtype_fast_lookup']
    cell_mtypes = connectomics_data['cell_mtypes']
    cell_coords = connectomics_data['cell_coords']
    
    # NEW: Only grab the integrated synapse_dict
    synapse_dict = connectomics_data['synapse_dict']
    
    # NEW: Grab the genetic types array
    cell_gtypes = connectomics_data['cell_gtypes']

    extracted_data = {}

    for name in name_list:
        # 1. Parse the macro-population string (e.g., "L23_exc" -> "L23", "exc")
        parts = name.split('_')
        target_layer = parts[0].upper()
        target_group = parts[1].lower()

        global_indices = []

        # 2. Iterate through the array using the FAST LOOKUP
        for raw_idx, mtype in enumerate(cell_mtypes):
            if mtype not in mtype_fast_lookup:
                continue

            layer, bio_type = mtype_fast_lookup[mtype]

            # 3. Matching Logic
            is_match = False
            if layer == target_layer:
                if target_group == 'exc' and bio_type == 'Excitatory':
                    # Ensure L4_exc doesn't swallow L4_ss if both are requested
                    if target_layer == 'L4' and 'L4_ss' in name_list and 'SS' in mtype:
                        is_match = False
                    else:
                        is_match = True
                elif target_group == 'inh' and bio_type == 'Inhibitory':
                    is_match = True
                elif target_group == 'ss' and 'SS' in mtype:
                    is_match = True

            if is_match:
                global_indices.append(raw_idx)

        # 4. Process the matched sub-population
        global_indices = np.array(global_indices)

        if len(global_indices) == 0:
            print(f"Warning: No cells found for macro-population '{name}'")
            continue

        # Extract Coordinates (Made compliant: converted to list of dicts)
        sub_coords_raw = cell_coords[global_indices]
        sub_coords = [{'x': row[0], 'y': row[1], 'z': row[2]} for row in sub_coords_raw]
        
        # NEW: Extract the genetic types for the matched sub-population
        sub_gtypes = cell_gtypes[global_indices]

        # Create Mapping: Local (0 to N) -> Raw Global Index
        local_to_raw_map = {local_idx: raw_idx for local_idx, raw_idx in enumerate(global_indices)}

        # NEW: Filter the single synapse_dict (Keys are LOCAL, Values are RAW)
        sub_synapse_dict = {
            local_idx: synapse_dict[raw_idx] 
            for local_idx, raw_idx in enumerate(global_indices) if raw_idx in synapse_dict
        }

        # 5. Pack strictly the requested keys into the return dictionary
        extracted_data[name] = {
            'cell_coords': sub_coords,
            'local_to_raw_map': local_to_raw_map,
            'synapse_dict': sub_synapse_dict,
            'cell_gtypes': sub_gtypes
        }

    return extracted_data







class MorphoPath:
    '''
    A dictionary-like manager where the key is the subpopulation name 
    and the value is a list of loading paths.
    '''

    def __init__(self):
        self.paths = {}

    def add_subpop(self, name, nids, parent_path):
        '''
        - name: the name of the subpopulation (string)
        - nids: list/array of neurons' identification numbers (integers)
        - parent_path: path to the folder storing the .hoc files
        '''
        morph_paths = [
            os.path.join(parent_path, f"neuron_{nid}_aligned.hoc") 
            for nid in nids
        ]

        # Store in the dictionary
        self.paths[name] = morph_paths


    def construct_dict(self, name_list, name_path, morph_path):
        ''' 
        This method iteratively applies the add_subpop method on the subpopulations
        in the name_list. The nids are stored in the name_path folder txt files.
        '''
        for n in name_list:
            filename = f"{n}_nids.txt"    
            full_path = os.path.join(name_path, filename)
            
            # Use numpy to cleanly read the text file back into an array of integers
            # dtype=int ensures they don't become floats
            nids_array = np.loadtxt(full_path, dtype=int)
            
            # Safety check: np.loadtxt returns a 0-d array if there's only one number in the file.
            # We convert it to a 1D array so it can still be iterated over.
            if nids_array.ndim == 0:
                nids_array = np.array([nids_array])

            self.add_subpop(n, nids_array, morph_path)

            
    @property
    def get_paths(self):
        return self.paths




        
if __name__ == '__main__':
    params = multicompartment_params()
     
    print(dir(params))

