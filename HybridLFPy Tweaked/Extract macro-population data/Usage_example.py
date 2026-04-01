

import numpy as np
from scipy.spatial.distance import cdist
from scipy import sparse
import os


class Connectomics:


   def __init__(self,connectomics_path='',connectomics_output='',Calculate=True):

        self.connectomics_path = connectomics_path
        self.connectomics_output = connectomics_output


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



        self.input_dict = input_dict

        
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

            # Open the file in 'read-binary' mode and load the data
            full_path_conn = os.path.join(self.connectomics_path, 'conn.pkl')
            with open(full_path_conn, 'rb') as f:
                self.conn_data = pickle.load(f)

            # Execute Pipeline
            self.get_lookup_table()
            self.calculate_bbp_relative_presences()
            self.get_ADJ()
            self.extract_connectivity_dicts()



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
                - 'cell_coords': 2D array of shape $(N, 3)$ containing XYZ coordinates in $\mu m$.
                - 'adj_matrix': Sparse CSR matrix representing the synaptic adjacency graph.
                - 'post_to_pre': Afferent dict mapping Post-synaptic IDs to lists of Pre-synaptic IDs.
                - 'pre_to_post': Efferent dict mapping Pre-synaptic IDs to lists of Post-synaptic IDs.
        """
        return {
            'bbp_results': self.bbp_results,
            'bbp_totals': self.bbp_totals,
            'mtype_fast_lookup': self.mtype_fast_lookup,
            'cell_mtypes': self.cell_mtypes,
            'cell_coords': self.cell_coords,
            'adj_matrix': self.adj_matrix,
            'post_to_pre': self.post_to_pre,
            'pre_to_post': self.pre_to_post
        }
        
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

        # We store only the (row, col) indices of successful connections
        # This avoids storing the millions of 'zeros' in a dense matrix
        all_rows = []
        all_cols = []

        # 1. Pre-group indices to avoid calling np.where millions of times
        mtype_indices = {m: np.where(cell_mtypes == m)[0] for m in unique_mtypes}

        for pre_mtype in unique_mtypes:
            pre_idx = mtype_indices[pre_mtype]
            coords_pre = cell_coords[pre_idx]

            for post_mtype in unique_mtypes:
                # Only proceed if this pathway exists in our connectivity data
                if post_mtype not in conn_data.get('best_fit', {}).get(pre_mtype, {}):
                    continue

                post_idx = mtype_indices[post_mtype]
                coords_post = cell_coords[post_idx]

                # 2. Calculate distances ONLY for this sub-block
                # This is the key RAM saver
                d_sub = cdist(coords_pre, coords_post, metric='euclidean')

                fit_type = conn_data['best_fit'][pre_mtype][post_mtype]

                # 3. Calculate probabilities for this sub-block
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
                    # Fallback: Step-function bins
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

                # 4. Roll the dice for this sub-block only
                # random_matrix is now just the size of the population pair
                trial = np.random.rand(*p_sub.shape) < p_sub

                # 5. Extract successful connection coordinates
                # local_rows/cols are indices within the sub-block
                local_rows, local_cols = np.where(trial)

                if len(local_rows) > 0:
                    # Map local indices back to global cell indices
                    global_rows = pre_idx[local_rows]
                    global_cols = post_idx[local_cols]

                    # Check for and remove autapses (self-connections)
                    mask_autapse = global_rows != global_cols
                    all_rows.extend(global_rows[mask_autapse])
                    all_cols.extend(global_cols[mask_autapse])

                # Explicitly clear sub-matrices from memory
                del d_sub, p_sub, trial

        # 6. Final Assembly: Create a Sparse CSR matrix
        # This stores only the '1's. Total RAM used is proportional to number of connections.
        data = np.ones(len(all_rows), dtype=np.uint8)
        adj_matrix = sparse.csr_matrix((data, (all_rows, all_cols)), shape=(N, N))
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
        - cell_mtypes: 1D numpy array of string m-types.
        - cell_coords: 2D numpy array of shape (N, 3) containing (x, y, z).
        """
        input_dict = self.input_dict
        bbp_results = self.bbp_results
        output_folder = self.connectomics_output
        radius = input_dict['Geometry']['radius']

        mtypes_list = []
        coords_list = []

        # Pre-calculate the column's cross-sectional area
        area = np.pi * (radius ** 2)

        if verbose:
            print("BUILDING MICROCIRCUIT...")
            print("="*40)

        for layer, bio_types in input_dict['Cells'].items():
            if layer not in input_dict['Layers']:
                if verbose: print(f"Warning: Z-boundaries for '{layer}' not defined. Skipping.")
                continue

            # Unpack bounds and calculate layer volume
            bounds = input_dict['Layers'][layer]
            z_min, z_max = min(bounds), max(bounds)
            layer_height = z_max - z_min
            volume_um3 = area * layer_height

            if verbose:
                print(f"\n--- {layer} (Height: {layer_height} um) ---")

            for raw_bio_type, density_mm3 in bio_types.items():
                if density_mm3 <= 0:
                    continue

                # 1. Normalize the bio_type key (handles 'inh', 'exc', 'Inhibitory', etc.)
                bio_type = 'Inhibitory' if raw_bio_type.lower().startswith('inh') else 'Excitatory'

                # 2. Convert density (cells/mm^3) to cells/um^3, calculate TOTAL cells for this group
                density_um3 = density_mm3 / 1e9
                total_group_count = int(np.round(density_um3 * volume_um3))

                if total_group_count <= 0:
                    continue

                # 3. Check if we have BBP distribution data for this layer and type
                if layer not in bbp_results or bio_type not in bbp_results[layer]:
                    if verbose: print(f"  Warning: BBP results for {layer} {bio_type} not found. Skipping.")
                    continue

                if verbose:
                    print(f"  {bio_type} (Target Total: ~{total_group_count} cells):")

                # 4. Iterate through the BBP sub-populations
                mtype_data = bbp_results[layer][bio_type]

                for mtype, data in mtype_data.items():
                    percentage = data['percentage']

                    # Calculate the exact number of cells for this specific m-type
                    count = int(np.round(total_group_count * (percentage / 100.0)))

                    if count <= 0:
                        continue

                    if verbose:
                        print(f"    -> {mtype:<12}: {count:>5} cells ({percentage:>5.2f}%)")

                    # 5. Generate uniform random spatial positions
                    r_random = radius * np.sqrt(np.random.rand(count))
                    theta_random = np.random.rand(count) * 2 * np.pi

                    x = r_random * np.cos(theta_random)
                    y = r_random * np.sin(theta_random)
                    z = np.random.uniform(z_min, z_max, count)

                    # Stack the generated data
                    coords = np.column_stack((x, y, z))
                    coords_list.append(coords)

                    mtypes = np.full(count, mtype, dtype=object)
                    mtypes_list.append(mtypes)

        # Concatenate everything into the final format
        if mtypes_list:
            cell_mtypes = np.concatenate(mtypes_list)
            cell_coords = np.vstack(coords_list)
            if verbose:
                print("="*40)
                print(f"SUCCESS: Generated {len(cell_mtypes)} total neurons.")
        else:
            cell_mtypes = np.array([])
            cell_coords = np.empty((0, 3))
            if verbose:
                print("WARNING: No cells were generated. Check your input dictionary.")

        # --- NEW: Save arrays to disk if an output folder is provided ---
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

            # Inhibitory Types
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
            'SAC': ('Small Axon Cell', 'Inhibitory')
        }

        file_path = 'S1-cells-distributions-Rat.txt'
        valid_layers = {"L1", "L23", "L4", "L5", "L6"}


        full_path_conn = os.path.join(self.connectomics_path, file_path)


        # Initialize the fast-readable variable
        mtype_fast_lookup = {}

        if not os.path.exists(full_path_conn):
            print(f"Error: '{full_path_conn}' not found. Please upload it to the 'anatomy' folder.")
        else:
            with open(full_path_conn, 'r') as f:
                for line in f.read().split('\n'):
                    if line.strip():
                        parts = line.split()
                        if len(parts) == 5:
                            metype, mtype, etype, n, m = parts

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


        # --- Quick Test ---
        test_cell = 'L23_LBC'
        if test_cell in mtype_fast_lookup:
            layer, cell_type = mtype_fast_lookup[test_cell]
            print(f"Success! {test_cell} -> Layer: {layer}, Type: {cell_type}")










import pickle



# ==========================================
# Execution Example
# ==========================================

connectomics_path = '/content/drive/MyDrive/Colab Notebooks/Connectomics/'

connectomics_output = '/content/drive/MyDrive/Colab Notebooks/Connectomic_output/'

column_input = {
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



conn =  Connectomics(connectomics_path,connectomics_output,column_input)

conn_dict = conn.get_ConnectomicInfo


name_list = [
    "L23_exc", "L23_inh", 
    "L4_exc", "L4_inh", "L4_ss", 
    "L5_exc", "L5_inh", 
    "L6_exc", "L6_inh"
]
extracted_pops = extract_macro_populations_debug(conn_dict, name_list)
debug_plot_macro_populations(extracted_pops, column_input)













































