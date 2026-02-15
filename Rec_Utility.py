'''
Utility functions used to extract, correct, polish automatically reconstructed neurons from the google project
'''
def validate_reconstruction_quality(df, com_distance_threshold=50000, output_dir="QA_Reports"):
    """
    Evaluates the quality of the final reconstructed neuron (Labled_propagated).
    
    Checks:
      1. Unique Soma root (exactly one p=-1).
      2. Absence of debris (single connected component).
      3. Presence of >= 1 Dendritic arbor.
      4. Presence of exactly 1 Axon arbor (only 1 primary branch leading to an axon).
      5. Soma is centered within com_distance_threshold (nm) of the entire arbor.
    """
    print("\n🔬 Running Quality Assurance Check...")
    
    # 1. Setup OS Logging
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    previous_logs = glob.glob(os.path.join(output_dir, "*.txt"))
    
    is_valid = True
    reasons = []
    
    if df is None or df.empty:
        return False, ["DataFrame is empty."]

    df_clean = df.copy()
    df_clean['annotated_type'] = df_clean['annotated_type'].astype(str)
    
    # --- CHECK 1: Unique Soma Root ---
    roots = df_clean[df_clean['p'] == -1]
    
    if len(roots) == 0:
        is_valid = False
        reasons.append("Topology Error: No root node (p=-1) found.")
        soma_root = None
    elif len(roots) > 1:
        is_valid = False
        reasons.append(f"Topology Error: Multiple roots found ({len(roots)}). Skeleton is fragmented.")
        soma_root = roots.iloc[0] 
    else:
        soma_root = roots.iloc[0]
        if 'Soma' not in soma_root['annotated_type'] and soma_root['annotated_type'] != '3':
            is_valid = False
            reasons.append(f"Annotation Error: Root node is annotated as '{soma_root['annotated_type']}', not 'Soma'.")

    # --- CHECK 2: Absence of Debris (NetworkX & DBSCAN) ---
    G = nx.Graph()
    G.add_nodes_from(df_clean['id'])
    edges = df_clean[df_clean['p'] != -1][['id', 'p']].values
    G.add_edges_from(edges)
    
    # Topological Debris Check
    num_components = nx.number_connected_components(G)
    if num_components > 1:
        is_valid = False
        reasons.append(f"Debris Detected: Skeleton is broken into {num_components} disconnected components.")

    # Spatial Debris Check
    coords = df_clean[['x', 'y', 'z']].values
    clustering = DBSCAN(eps=15000, min_samples=10).fit(coords)
    unique_clusters = len(set(clustering.labels_) - {-1})
    if unique_clusters > 1:
        is_valid = False
        reasons.append(f"Spatial Disconnect: DBSCAN identified {unique_clusters} floating clusters.")

    # --- CHECK 3: Navis Sanity & At least one Dendritic Arbor ---
    # Validate structure using navis
    temp_df = df_clean.rename(columns={'id': 'node_id', 'p': 'parent_id', 'r': 'radius'})
    try:
        n = navis.TreeNeuron(temp_df, name='qa_neuron')
        if not n.is_sane:
            is_valid = False
            reasons.append("Navis Validation Failed: Tree contains cycles or breaks.")
    except Exception as e:
        pass # Handle gracefully if skeleton is severely malformed

    has_dendrite = df_clean['annotated_type'].str.contains('Dendrite|1', case=False, regex=True).any()
    if not has_dendrite:
        is_valid = False
        reasons.append("Morphology Error: No dendritic arbor detected.")

    # --- CHECK 4: Exactly One Single Axon Arbor ---
    axon_branches = 0
    if soma_root is not None:
        soma_id = soma_root['id']
        primary_roots = df_clean[df_clean['p'] == soma_id]['id'].tolist()
        
        DiG = nx.DiGraph()
        DiG.add_edges_from(df_clean[df_clean['p'] != -1][['p', 'id']].values)
        
        for pr_id in primary_roots:
            try:
                descendants = list(nx.descendants(DiG, pr_id)) + [pr_id]
                branch_types = df_clean[df_clean['id'].isin(descendants)]['annotated_type']
                if branch_types.str.contains('Axon|0|Myelinated', case=False, regex=True).any():
                    axon_branches += 1
            except nx.NetworkXError:
                continue
                
        if axon_branches == 0:
            is_valid = False
            reasons.append("Morphology Error: No axon arbor detected branching from the soma.")
        elif axon_branches > 1:
            is_valid = False
            reasons.append(f"Morphology Error: Multiple axon arbors detected ({axon_branches} primary branches contain Axon labels).")

    # --- CHECK 5: Soma is Centered (cKDTree & Numpy) ---
    if soma_root is not None:
        com = np.mean(coords, axis=0) 
        soma_coords = np.array([soma_root['x'], soma_root['y'], soma_root['z']])
        
        # Snap theoretical CoM to nearest skeleton physical node
        tree = cKDTree(coords)
        _, nearest_idx = tree.query(com)
        snapped_com = coords[nearest_idx]

        distance_to_com = np.linalg.norm(soma_coords - snapped_com)
        if distance_to_com > com_distance_threshold:
            is_valid = False
            reasons.append(f"Spatial Error: Soma is heavily off-center. Distance to CoM is {distance_to_com:.1f} nm (Threshold: {com_distance_threshold} nm).")

    # --- TERMINAL OUTPUT ---
    print(f"\n📝 QA Result: {'✅ PASS' if is_valid else '❌ FAIL'}")
    if not is_valid:
        print("Reasons for failure:")
        for r in reasons:
            print(f"  - {r}")

    # --- VISUALIZATION (Plotly & Subplots) ---
    fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])

    # Skeleton
    node_map = df_clean.set_index('id')[['x', 'y', 'z']].to_dict('index')
    x_lines, y_lines, z_lines = [], [], []
    for _, row in df_clean.iterrows():
        pid = row['p']
        if pid != -1 and pid in node_map:
            parent = node_map[pid]
            x_lines.extend([row['x'], parent['x'], None])
            y_lines.extend([row['y'], parent['y'], None])
            z_lines.extend([row['z'], parent['z'], None])
            
    fig.add_trace(go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines', line=dict(color='lightgrey', width=2),
        name='Skeleton', hoverinfo='skip'
    ))

    if soma_root is not None:
        # Plot Soma
        fig.add_trace(go.Scatter3d(
            x=[soma_root['x']], y=[soma_root['y']], z=[soma_root['z']],
            mode='markers', marker=dict(size=12, color='gold', symbol='diamond', line=dict(color='black', width=2)),
            name='Soma Root'
        ))
        # Plot CoM
        fig.add_trace(go.Scatter3d(
            x=[snapped_com[0]], y=[snapped_com[1]], z=[snapped_com[2]],
            mode='markers', marker=dict(size=8, color='magenta', symbol='x', line=dict(width=2)),
            name='Center of Mass (Snapped)'
        ))
        # Trace line between Soma and CoM
        fig.add_trace(go.Scatter3d(
            x=[soma_root['x'], snapped_com[0]], y=[soma_root['y'], snapped_com[1]], z=[soma_root['z'], snapped_com[2]],
            mode='lines', line=dict(color='magenta', width=4, dash='dot'),
            name=f'Soma-CoM Offset ({distance_to_com:.0f} nm)'
        ))

    # Highlight Debris / Disconnected components (Red)
    if soma_root is not None and not is_valid:
        try:
            connected_to_soma = nx.node_connected_component(G, soma_root['id'])
            debris_nodes = df_clean[~df_clean['id'].isin(connected_to_soma)]
            if not debris_nodes.empty:
                fig.add_trace(go.Scatter3d(
                    x=debris_nodes['x'], y=debris_nodes['y'], z=debris_nodes['z'],
                    mode='markers', marker=dict(size=4, color='red'),
                    name=f'Debris ({len(debris_nodes)} nodes)'
                ))
        except:
            pass

    title_color = "green" if is_valid else "red"
    fig.update_layout(
        title=f"<span style='color:{title_color}'><b>Reconstruction Quality Check: {'PASS' if is_valid else 'FAIL'}</b></span>",
        scene=dict(xaxis_title='X (nm)', yaxis_title='Y (nm)', zaxis_title='Z (nm)', aspectmode='data', bgcolor='white'),
        height=800, margin=dict(l=0, r=0, b=0, t=50)
    )
    fig.show()

    return is_valid


def extract_skeleton_data(neuron_id):
    """
    1. Loads Annotation Cloud (Reference).
    2. Loads SWC Skeleton (Target).
    3. Aligns Skeleton to Cloud.
    4. Finds nearest Annotation for each Skeleton node (With 5um Max Dist).
    """
    print(f"📥 Extracting and Mapping data for Neuron {neuron_id}...")

    # --- STEP 1: LOAD REFERENCE CLOUD (Coords + Labels) ---
    try:
        reader = neuroglancer.read_precomputed_annotations.AnnotationReader(LOCAL_URL)
        rel_key = 'skeleton_id' if 'skeleton_id' in reader.relationships else 'associated_segments'
        anns = reader.relationships[rel_key][int(neuron_id)]

        ann_coords = []
        ann_labels = []

        for a in anns:
            try:
                # Get Coordinate
                pt = a.point if hasattr(a, 'point') else a['point']

                # Get Label Code (e.g., "0" for Axon, "5" for AIS)
                props = []
                if hasattr(a, 'props'): props = a.props
                elif isinstance(a, dict): props = a.get('props', [])
                elif hasattr(a, 'properties'): props = getattr(a, 'properties', [])

                # Default to '999' if missing
                label_code = str(props[0]) if props else '999'

                ann_coords.append(pt)
                ann_labels.append(label_code)
            except: continue

        if not ann_coords: raise ValueError("No annotation points found.")

        ann_arr = np.array(ann_coords)
        label_arr = np.array(ann_labels)

        # Calculate Reference Range for Alignment
        ann_min, ann_max = ann_arr.min(axis=0), ann_arr.max(axis=0)
        ann_range = ann_max - ann_min
        ann_center = ann_arr.mean(axis=0)

    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return None

    # --- STEP 2: LOAD SWC FILES ---
    pattern = os.path.join(SKELETON_PATH, f"{neuron_id}*.swc")
    files = glob.glob(pattern)

    if not files:
        print("❌ No SWC files found.")
        return None

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, delim_whitespace=True, comment='#',
                             names=['id', 'type_code', 'x', 'y', 'z', 'radius', 'parent'])
            df['file_source'] = os.path.basename(f)
            dfs.append(df)
        except: pass

    if not dfs: return None
    swc_df = pd.concat(dfs, ignore_index=True)

    # --- STEP 3: ALIGN SKELETON TO ANNOTATION SPACE ---
    swc_min = swc_df[['x', 'y', 'z']].min().values
    swc_max = swc_df[['x', 'y', 'z']].max().values
    swc_range = swc_max - swc_min

    # Calculate Scale
    scale = np.divide(ann_range, swc_range, out=np.zeros_like(ann_range), where=swc_range!=0)
    swc_df['x'] *= scale[0]
    swc_df['y'] *= scale[1]
    swc_df['z'] *= scale[2]

    # Calculate Offset
    swc_center = swc_df[['x', 'y', 'z']].mean().values
    offset = ann_center - swc_center
    swc_df['x'] += offset[0]
    swc_df['y'] += offset[1]
    swc_df['z'] += offset[2]

    print(f"✅ Aligned Skeleton (Scale: {scale.round(2)})")

    # --- STEP 4: NEAREST NEIGHBOR MAPPING (FIXED) ---
    print("🔍 Mapping closest annotations (Max Dist: 5000nm)...")



    # A. Build KDTree
    tree = cKDTree(ann_arr)
    skeleton_pts = swc_df[['x', 'y', 'z']].values

    # B. Query with Distance Limit (5 microns)
    MAX_DIST = 5000.0
    distances, indices = tree.query(skeleton_pts, k=1, distance_upper_bound=MAX_DIST)

    # C. Map Results
    mapped_types = []
    for i, idx in enumerate(indices):
        # If no neighbor found within 5000nm, idx equals len(ann_arr) (infinity)
        if idx == len(ann_arr):
            mapped_types.append('Unknown')
        else:
            code = label_arr[idx]
            # Map "0" -> "Axon", "5" -> "AIS", etc.
            mapped_types.append(ANNOTATION_LABEL_MAP.get(code, f"Unknown ({code})"))

    # Store in DataFrame
    swc_df['annotated_type'] = mapped_types
    swc_df['distance_to_annotation'] = distances

    # --- FINALIZE ---
    final_df = swc_df[['id', 'annotated_type', 'x', 'y', 'z', 'radius', 'distance_to_annotation', 'parent', 'file_source']].copy()

    return final_df






def plot_skeleton_dataframe(df, neuron_id="Unknown"):
    """
    Takes a DataFrame (from extract_skeleton_data) and plots the skeleton
    colored by the NEAREST ANNOTATION TYPE (Axon, AIS, Myelin, Dendrite, Soma).
    """
    if df is None or df.empty:
        print("❌ DataFrame is empty.")
        return

    print(f"📊 Plotting Skeleton for Neuron {neuron_id}...")

    fig = go.Figure()

    # --- 1. DEFINE COLORS (Full Biological Schema) ---
    COLOR_MAP = {
        'Soma': 'green',
        'Axon': 'red',
        'Dendrite': 'blue',
        'AIS': 'gold',              # Axon Initial Segment (Critical)
        'Myelinated Axon': 'darkred',
        'Myelinated Fragment': 'brown',
        'Cilium': 'magenta',
        'Astrocyte': 'purple',
        'Unclassified': 'gray',
        'Unknown': 'black'
    }

    # --- 2. PLOT BY ANNOTATED TYPE ---
    # Group by the 'annotated_type' column derived from your annotations
    if 'annotated_type' not in df.columns:
        print("⚠️ 'annotated_type' column missing. Falling back to original 'type'.")
        group_col = 'type'
    else:
        group_col = 'annotated_type'

    # Iterate through each biological group (e.g., all Axon points, all Dendrite points)
    for seg_type, type_data in df.groupby(group_col):

        # Get color (default to gray if not in map)
        color = COLOR_MAP.get(seg_type, 'gray')

        # Scale marker size by radius
        # Cap large structures (Soma) at 15 to prevent blocking the view
        # Ensure thin structures (Axons) are at least size 2
        if 'radius' in type_data.columns:
            sizes = type_data['radius'].clip(lower=1, upper=15) * 2
        else:
            sizes = 3

        fig.add_trace(go.Scatter3d(
            x=type_data['x'],
            y=type_data['y'],
            z=type_data['z'],
            mode='markers', # Point cloud style is efficient for dense skeletons
            marker=dict(
                size=sizes,
                color=color,
                opacity=0.7,
                sizemode='diameter'
            ),
            name=f"{seg_type} (n={len(type_data)})"
        ))

    # --- 3. LAYOUT ---
    fig.update_layout(
        title=f"Neuron {neuron_id}: Full Class Schema (AIS, Myelin, etc.)",
        width=1000,
        height=800,
        scene=dict(
            xaxis_title='X (nm)',
            yaxis_title='Y (nm)',
            zaxis_title='Z (nm)',
            aspectmode='data', # Critical: Keeps biological proportions
            bgcolor='white'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(title="Structure Type")
    )

    fig.show()


def visualize_comparison_lines(neuron_id, df_processed):
    """
    Side-by-side comparison using LINES to show connectivity.
    [Original SWC] vs [Processed DataFrame]

    Fix: Iterates by 'file' to handle duplicate IDs across multiple SWC parts.
    """
    print(f"👀 Generating Line Comparison for Neuron {neuron_id}...")

    # --- 1. LOAD RAW DATA (Original Voxel Space) ---
    pattern = os.path.join(SKELETON_PATH, f"{neuron_id}*.swc")
    files = glob.glob(pattern)
    if not files:
        print("❌ Original SWC file not found.")
        return

    dfs_raw = []
    for f in files:
        try:
            t = pd.read_csv(f, delim_whitespace=True, comment='#',
                            names=['id', 'type', 'x', 'y', 'z', 'r', 'p'])
            t['file'] = os.path.basename(f) # Store filename
            dfs_raw.append(t)
        except: pass
    df_raw = pd.concat(dfs_raw, ignore_index=True)

    # --- 2. SETUP SUBPLOTS ---
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=(f"Original SWC (Voxel Space)", f"Processed Data (Nanometers)"),
        horizontal_spacing=0.05
    )

    # --- 3. HELPER: DRAW LINES FILE-BY-FILE ---
    def add_lines_to_plot(df, row, col, color_map, label_prefix):
        # We group by 'file' (or 'file_source') to isolate IDs
        # Check which column exists
        file_col = 'file' if 'file' in df.columns else 'file_source'

        # If neither exists (single file), just use a dummy group
        if file_col not in df.columns:
            df[file_col] = 'single_file'

        for filename, file_data in df.groupby(file_col):
            # Create Lookup for THIS file only
            coords = file_data.set_index('id')[['x', 'y', 'z']].to_dict('index')

            # We also group by Type within the file to apply colors
            # Determine type column name
            type_col = 'annotated_type' if 'annotated_type' in df.columns else 'type'

            for seg_type, type_data in file_data.groupby(type_col):
                c = color_map.get(seg_type, 'black')

                x_lines, y_lines, z_lines = [], [], []

                for _, row_data in type_data.iterrows():
                    pid = int(row_data['p'] if 'p' in row_data else row_data['parent'])

                    # Draw line if parent is in THIS file
                    if pid != -1 and pid in coords:
                        parent = coords[pid]
                        x_lines.extend([row_data['x'], parent['x'], None])
                        y_lines.extend([row_data['y'], parent['y'], None])
                        z_lines.extend([row_data['z'], parent['z'], None])

                if x_lines:
                    fig.add_trace(
                        go.Scatter3d(
                            x=x_lines, y=y_lines, z=z_lines,
                            mode='lines',
                            line=dict(color=c, width=3),
                            name=f"{label_prefix}: {seg_type}",
                            legendgroup=f"{label_prefix}_{seg_type}",
                            showlegend=True # Simplifies legend
                        ),
                        row=row, col=col
                    )

    # --- 4. PLOT LEFT: ORIGINAL SWC ---
    RAW_COLORS = {1: 'green', 2: 'red', 3: 'blue', 4: 'magenta', 5: 'gold'}
    add_lines_to_plot(df_raw, row=1, col=1, color_map=RAW_COLORS, label_prefix="Orig")

    # --- 5. PLOT RIGHT: PROCESSED DATA ---
    NEW_COLORS = {
        'Soma': 'green', 'Axon': 'red', 'Dendrite': 'blue',
        'AIS': 'gold', 'Myelinated Axon': 'darkred',
        'Myelinated Fragment': 'brown', 'Cilium': 'magenta',
        'Astrocyte': 'purple', 'Unclassified': 'gray', 'Unknown': 'black'
    }
    add_lines_to_plot(df_processed, row=1, col=2, color_map=NEW_COLORS, label_prefix="New")

    # --- 6. FINALIZE ---
    fig.update_layout(
        title=f"Connectivity Check: Neuron {neuron_id}",
        height=800, width=1500,
        scene=dict(aspectmode='data', bgcolor='white'),
        scene2=dict(aspectmode='data', bgcolor='white')
    )

    fig.show()


def check_coordinate_consistency():
    print(f"⚖️ Checking Consistency for Neuron {TARGET_NEURON_ID}...\n")

    # --- A. GET ANNOTATION STATS (The "Cloud") ---
    try:
        reader = neuroglancer.read_precomputed_annotations.AnnotationReader(LOCAL_URL)
        rel_key = 'skeleton_id' if 'skeleton_id' in reader.relationships else 'associated_segments'

        # Get all points
        anns = reader.relationships[rel_key][TARGET_NEURON_ID]
        ann_coords = []
        for a in anns:
            pt = a.point if hasattr(a, 'point') else a['point']
            ann_coords.append(pt)

        ann_arr = np.array(ann_coords)

        print("🔵 ANNOTATION DATA (Neuroglancer):")
        if len(ann_arr) > 0:
            print(f"   Count: {len(ann_arr)} points")
            print(f"   X Range: {ann_arr[:,0].min():.1f} to {ann_arr[:,0].max():.1f}")
            print(f"   Y Range: {ann_arr[:,1].min():.1f} to {ann_arr[:,1].max():.1f}")
            print(f"   Z Range: {ann_arr[:,2].min():.1f} to {ann_arr[:,2].max():.1f}")
            print(f"   Sample Point: {ann_arr[0]}")
        else:
            print("   ❌ No annotation points found.")
            return

    except Exception as e:
        print(f"   ❌ Error reading annotations: {e}")
        return

    print("-" * 40)

    # --- B. GET SKELETON STATS (The SWC) ---
    pattern = os.path.join(SKELETON_PATH, f"{TARGET_NEURON_ID}*.swc")
    files = glob.glob(pattern)

    print(f"⚫ SKELETON DATA (SWC Files: {len(files)} found):")

    all_swc_coords = []

    for f in files:
        try:
            # Read X, Y, Z columns (index 2, 3, 4)
            df = pd.read_csv(f, delim_whitespace=True, comment='#', header=None)
            # Filter valid rows (some SWC files have headers)
            coords = df.iloc[:, 2:5].values
            all_swc_coords.extend(coords)
        except Exception:
            pass

    if all_swc_coords:
        swc_arr = np.array(all_swc_coords)
        print(f"   Count: {len(swc_arr)} nodes")
        print(f"   X Range: {swc_arr[:,0].min():.1f} to {swc_arr[:,0].max():.1f}")
        print(f"   Y Range: {swc_arr[:,1].min():.1f} to {swc_arr[:,1].max():.1f}")
        print(f"   Z Range: {swc_arr[:,2].min():.1f} to {swc_arr[:,2].max():.1f}")
        print(f"   Sample Point: {swc_arr[0]}")

        # --- C. DIAGNOSIS ---
        print("\n🧐 DIAGNOSIS:")

        # Check X axis ratio
        x_ratio = ann_arr[:,0].mean() / swc_arr[:,0].mean()
        print(f"   Ratio (Annotation / SWC): ~{x_ratio:.2f}")

        if 900 < x_ratio < 1100:
            print("   ⚠️ ISSUE DETECTED: Annotations are in Nanometers, SWC is in Microns.")
            print("   👉 FIX: Multiply SWC coordinates by 1000.")
        elif 3 < x_ratio < 5 or 0.2 < x_ratio < 0.3:
            print("   ⚠️ ISSUE DETECTED: Likely a Voxel Resolution mismatch.")
            print("   👉 CHECK: The 'info' file for 'scales'. SWC might be in voxels (e.g. 4nm pixels).")
        elif 0.9 < x_ratio < 1.1:
            print("   ✅ Coordinates seem consistent (Unit 1:1).")
        else:
            print("   ⚠️ Complex mismatch. Check if SWC is raw voxels and Annotations are physical nm.")

    else:
        print("   ❌ No SWC coordinates found.")

def load_and_align():
    print(f"⚖️ Aligning Neuron {TARGET_NEURON_ID}...")

    # --- A. Load Annotations (Target Space) ---
    try:
        reader = neuroglancer.read_precomputed_annotations.AnnotationReader(LOCAL_URL)
        rel_key = 'skeleton_id' if 'skeleton_id' in reader.relationships else 'associated_segments'
        annotations = reader.relationships[rel_key][TARGET_NEURON_ID]

        groups = {}
        all_ann_coords = []

        for ann in annotations:
            try:
                coord = ann.point if hasattr(ann, 'point') else ann['point']

                # Extract Label Code ("0", "1", etc.)
                props = []
                if hasattr(ann, 'props'): props = ann.props
                elif isinstance(ann, dict): props = ann.get('props', [])
                elif hasattr(ann, 'properties'): props = getattr(ann, 'properties', [])

                code = str(props[0]) if props else "999"

                # Convert Code -> Text Label (e.g., "0" -> "Axon")
                label_text = LABELS_MAP.get(code, f"Unknown ({code})")

                if label_text not in groups: groups[label_text] = []
                groups[label_text].append(coord)
                all_ann_coords.append(coord)
            except: continue

        if not all_ann_coords: return None, None
        ann_arr = np.array(all_ann_coords)

    except Exception as e:
        print(f"❌ Annotation Error: {e}")
        return None, None

    # --- B. Load SWC (Source Space) ---
    pattern = os.path.join(SKELETON_PATH, f"{TARGET_NEURON_ID}*.swc")
    files = glob.glob(pattern)
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, delim_whitespace=True, comment='#',
                             names=['id', 'type', 'x', 'y', 'z', 'r', 'parent'])
            df['file'] = os.path.basename(f)
            dfs.append(df)
        except: pass

    if not dfs: return None, None
    swc_df = pd.concat(dfs, ignore_index=True)

    # --- C. CALCULATE ALIGNMENT ---
    ann_min, ann_max = ann_arr.min(axis=0), ann_arr.max(axis=0)
    swc_min, swc_max = swc_df[['x', 'y', 'z']].min().values, swc_df[['x', 'y', 'z']].max().values

    ann_range = ann_max - ann_min
    swc_range = swc_max - swc_min

    # Scale
    scale = np.divide(ann_range, swc_range, out=np.zeros_like(ann_range), where=swc_range!=0)
    print(f"📏 Detected Scale Factors: {scale.round(2)}")

    swc_df['x'] *= scale[0]; swc_df['y'] *= scale[1]; swc_df['z'] *= scale[2]

    # Offset
    swc_center = swc_df[['x', 'y', 'z']].mean().values
    ann_center = ann_arr.mean(axis=0)
    offset = ann_center - swc_center

    swc_df['x'] += offset[0]; swc_df['y'] += offset[1]; swc_df['z'] += offset[2]

    return groups, swc_df

# ==========================================
# 3. PLOT
# ==========================================
grouped_points, swc_df = load_and_align()

if grouped_points and swc_df is not None:
    print("📊 Generating Aligned Plot...")
    fig = go.Figure()

    # Plot Skeleton (Black Lines)
    for filename, data in swc_df.groupby('file'):
        coords = data.set_index('id')[['x', 'y', 'z']].to_dict('index')
        x_lines, y_lines, z_lines = [], [], []

        for _, row in data.iterrows():
            pid = int(row['parent'])
            if pid != -1 and pid in coords:
                parent = coords[pid]
                x_lines.extend([row['x'], parent['x'], None])
                y_lines.extend([row['y'], parent['y'], None])
                z_lines.extend([row['z'], parent['z'], None])

        fig.add_trace(go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode='lines',
            line=dict(color='black', width=3),
            name='Skeleton',
            opacity=0.6,
            showlegend=False
        ))

    # Plot Annotations (Colored by Class)
    for label_text, points in grouped_points.items():
        pts_np = np.array(points)

        # Get color based on label text (Axon, Dendrite, etc.)
        c = COLOR_MAP.get(label_text, 'gray')

        fig.add_trace(go.Scatter3d(
            x=pts_np[:, 0],
            y=pts_np[:, 1],
            z=pts_np[:, 2],
            mode='markers',
            marker=dict(size=4, color=c, opacity=0.8),
            name=f"{label_text} (n={len(points)})"
        ))

    fig.update_layout(
        title=f"Neuron {TARGET_NEURON_ID} (New Schema)",
        scene=dict(
            xaxis_title='X (nm)',
            yaxis_title='Y (nm)',
            zaxis_title='Z (nm)',
            aspectmode='data',
            bgcolor='white'
        ),
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()
else:
    print("❌ Failed to align data.")




def plot_soma_skeleton(df, soma_ids, centroid):
    """
    Plots the internal skeleton of the Soma.
    FIXED: Handles duplicate IDs by creating a unique 'file_id' key.
    """
    if df is None or not soma_ids:
        print("❌ No Soma data to plot.")
        return

    print(f"🧬 Plotting Soma Skeleton ({len(soma_ids)} nodes)...")

    # 1. Create a Unique Key for Lookup
    # We combine 'file_source' and 'id' to ensure uniqueness
    # (e.g. "file1.swc_1" vs "file2.swc_1")
    df_safe = df.copy()
    if 'file_source' not in df_safe.columns:
        df_safe['file_source'] = 'single_file'

    df_safe['uid'] = df_safe['file_source'].astype(str) + "_" + df_safe['id'].astype(str)

    # Also create the Parent UID so we know who to look for
    df_safe['p_uid'] = df_safe['file_source'].astype(str) + "_" + \
                       (df_safe['p'] if 'p' in df_safe.columns else df_safe['parent']).astype(str)

    # 2. Filter Data for Soma Only
    # We filter based on the original IDs passed in 'soma_ids'
    soma_df = df_safe[df_safe['id'].isin(soma_ids)].copy()

    # Create the fast lookup using the UNIQUE ID
    coords = df_safe.set_index('uid')[['x', 'y', 'z']].to_dict('index')

    fig = go.Figure()

    # --- 3. BUILD SOMA SKELETON (LINES) ---
    x_lines, y_lines, z_lines = [], [], []

    for _, row in soma_df.iterrows():
        p_uid = row['p_uid']
        parent_id = int(row['p'] if 'p' in row else row['parent'])

        # Draw line if:
        # 1. Parent exists in our unique lookup
        # 2. Parent's original ID is also in the Soma Cluster (Internal connection)
        if p_uid in coords and parent_id in soma_ids:
            parent = coords[p_uid]
            x_lines.extend([row['x'], parent['x'], None])
            y_lines.extend([row['y'], parent['y'], None])
            z_lines.extend([row['z'], parent['z'], None])

    # Trace 1: The Skeleton Lines
    fig.add_trace(go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines',
        line=dict(color='green', width=4),
        name='Soma Skeleton',
        opacity=0.8
    ))

    # Trace 2: The Nodes (Context)
    fig.add_trace(go.Scatter3d(
        x=soma_df['x'], y=soma_df['y'], z=soma_df['z'],
        mode='markers',
        marker=dict(size=3, color='green', opacity=0.3),
        name='Soma Nodes'
    ))

    # --- 4. PLOT CENTROID MARKER ---
    if centroid is not None:
        fig.add_trace(go.Scatter3d(
            x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
            mode='markers',
            marker=dict(
                size=15,
                color='black',
                symbol='cross',
                line=dict(width=2, color='white')
            ),
            name='Centroid'
        ))

    # --- LAYOUT ---
    fig.update_layout(
        title="Soma Skeleton & Centroid Check",
        scene=dict(
            aspectmode='data',
            xaxis_title='X (nm)', yaxis_title='Y (nm)', zaxis_title='Z (nm)',
            bgcolor='white',
            xaxis=dict(showbackground=False, showgrid=False, showticklabels=False),
            yaxis=dict(showbackground=False, showgrid=False, showticklabels=False),
            zaxis=dict(showbackground=False, showgrid=False, showticklabels=False),
        ),
        width=1000, height=800
    )
    fig.show()
def find_stable_soma_centroid(df, min_samples=5, step_eps=500, max_eps=25000, REQUIRED_STABILITY=10):
    """
    1. Filters Soma points.
    2. Adaptive DBSCAN: Finds the largest cluster that remains stable in size.
    3. PLOTS the Stability Curve (Epsilon vs Size).
    4. Returns centroid of the stable cluster (SNAPPED to the nearest skeleton node).
    """
    print("📍 Searching for Stable Soma Cluster...")

    # --- 1. Filter Soma Points ---
    soma_df = df[ (df['annotated_type'] == 'Soma') | (df['annotated_type'] == '3') ].copy()
    coords = soma_df[['x', 'y', 'z']].values

    if len(coords) < min_samples:
        print(f"⚠️ Small Soma ({len(coords)} pts). Skipping clustering.")
        return np.median(coords, axis=0), set(soma_df['id'])

    # --- 2. Adaptive Stability Loop ---
    history = []
    prev_size = 0
    stability_count = 0

    best_mask = None
    best_eps = 0
    found_plateau = False

    for eps in range(1000, max_eps + 1, step_eps):
        # Run Clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = clustering.labels_

        # Get sizes of valid clusters (ignore -1 noise)
        valid_labels = labels[labels != -1]

        if len(valid_labels) == 0:
            current_size = 0
        else:
            # Find size of the LARGEST cluster
            counts = np.bincount(valid_labels)
            largest_cluster_id = np.argmax(counts)
            current_size = counts[largest_cluster_id]

            # Save mask
            current_mask = (labels == largest_cluster_id)

        # Record History
        history.append({'eps': eps, 'size': current_size})

        # Check Stability
        if current_size > 0 and current_size == prev_size:
            stability_count += 1
        else:
            stability_count = 0

        # Found Plateau?
        if stability_count >= REQUIRED_STABILITY and not found_plateau:
            best_eps = eps - (step_eps * REQUIRED_STABILITY)
            best_mask = current_mask
            print(f"✅ Stable Cluster Found: Eps={best_eps}nm, Size={current_size} nodes")
            found_plateau = True
            # We continue the loop briefly just to fill the plot, but result is locked
            if len(history) > 15: break

        prev_size = current_size

    # Fallback
    if best_mask is None:
        print("⚠️ No plateau found. Using largest cluster from max_eps.")
        best_mask = (labels != -1) # Use all non-noise points
        best_eps = max_eps

    # --- 3. EMBEDDED PLOT: STABILITY CURVE ---
    df_hist = pd.DataFrame(history)

    fig = go.Figure()

    # The Curve
    fig.add_trace(go.Scatter(
        x=df_hist['eps'], y=df_hist['size'],
        mode='lines+markers',
        line=dict(color='blue', width=2),
        marker=dict(size=6),
        name='Cluster Size'
    ))

    # The Selected Point
    selected_size = df_hist[df_hist['eps'] == best_eps]['size'].iloc[0] if best_eps in df_hist['eps'].values else 0
    fig.add_trace(go.Scatter(
        x=[best_eps], y=[selected_size],
        mode='markers',
        marker=dict(size=14, color='red', symbol='circle-open', line=dict(width=3)),
        name=f'Selected (Eps={best_eps})'
    ))

    fig.update_layout(
        title="Soma Stability Analysis (DBSCAN)",
        xaxis_title="Search Radius (Epsilon nm)",
        yaxis_title="Points in Largest Cluster",
        template="plotly_white",
        height=400, width=800
    )
    fig.show()

    # --- 4. RETURN RESULT (SNAPPED) ---
    strict_cluster_df = soma_df[best_mask]
    points = strict_cluster_df[['x', 'y', 'z']].values

    # 1. Calculate Geometric Median (Virtual Center)
    geo_centroid = np.median(points, axis=0)

    # 2. Find the closest actual skeleton node to the geometric median
    dists = np.linalg.norm(points - geo_centroid, axis=1)
    closest_idx = np.argmin(dists)

    snapped_centroid = points[closest_idx]

    print(f"📍 Geometric Median: {geo_centroid}")
    print(f"📍 Snapped Centroid: {snapped_centroid} (Dist: {dists[closest_idx]:.2f} nm)")

    return snapped_centroid, set(strict_cluster_df['id'])   





def reorient_fragment(df, new_root_id):
    """
    Uses Navis to re-root the skeleton fragment at 'new_root_id'.
    Returns a DataFrame with updated parent-child relationships.
    """
    try:
        # 1. Prepare Data for Navis
        # Navis expects specific column names (node_id, parent_id)
        temp_df = df.copy()
        temp_df.rename(columns={'id': 'node_id', 'p': 'parent_id', 'r': 'radius'}, inplace=True)

        # 2. Create Neuron Object
        # checks_enabled=False makes it faster; we assume input is valid SWC chunks
        n = navis.TreeNeuron(temp_df, name='fragment')

        # 3. Reroot
        # This function flips the parent-child flow to start from new_root_id
        # inplace=True modifies the object directly
        n.reroot(new_root_id, inplace=True)

        # 4. Extract Data back to Pandas
        new_df = n.nodes.copy()

        # 5. Restore original column format
        new_df.rename(columns={'node_id': 'id', 'parent_id': 'p', 'radius': 'r'}, inplace=True)

        # Ensure we return only the standard columns in the right order
        desired_cols = ['id', 'type', 'x', 'y', 'z', 'r', 'p']
        return new_df[desired_cols]

    except Exception as e:
        print(f"⚠️ Navis Reroot Failed for ID {new_root_id}: {e}")
        # Fallback: Return original if Navis fails (prevents crash)
        return df

def stitch_neuron_fragments_smart(neuron_id, skeleton_path):
    """
    UPDATED: Adds a 'segment_id' column to track which file each node came from.
    Main Arbor = 0, Fragments = 1, 2, 3...
    """

    # --- 1. Load Main Arbor ---
    main_file = os.path.join(skeleton_path, f"{neuron_id}.0.swc")
    if not os.path.exists(main_file):
        print(f"❌ Main file missing: {main_file}")
        return None

    main_df = pd.read_csv(main_file, delim_whitespace=True, comment='#',
                          names=['id', 'type', 'x', 'y', 'z', 'r', 'p'])
    main_df['id'] = main_df['id'].astype(int)
    main_df['p'] = main_df['p'].astype(int)

    # <--- NEW: Tag the Main Arbor as Segment 0
    main_df['segment_id'] = 0

    print(f"🔹 Main Arbor: {len(main_df)} nodes")

    # --- 2. Find Fragments ---
    pattern = os.path.join(skeleton_path, f"{neuron_id}*.swc")
    all_files = sorted(glob.glob(pattern))
    fragment_files = [f for f in all_files if f != main_file]

    if not fragment_files:
        return main_df

    print(f"🧩 Processing {len(fragment_files)} fragments...")

    for i, frag_path in enumerate(fragment_files):
        frag_name = os.path.basename(frag_path)
        frag_df = pd.read_csv(frag_path, delim_whitespace=True, comment='#',
                              names=['id', 'type', 'x', 'y', 'z', 'r', 'p'])

        if frag_df.empty: continue

        # --- A. GEOMETRIC SEARCH ---
        main_tree = cKDTree(main_df[['x', 'y', 'z']].values)
        frag_coords = frag_df[['x', 'y', 'z']].values
        dists, main_indices = main_tree.query(frag_coords)

        best_idx_in_frag = np.argmin(dists)
        min_dist = dists[best_idx_in_frag]

        new_root_id = int(frag_df.iloc[best_idx_in_frag]['id'])
        target_main_idx = main_indices[best_idx_in_frag]
        target_parent_id = int(main_df.iloc[target_main_idx]['id'])

        print(f"   🔄 Re-rooting {frag_name}: New Root {new_root_id} -> Connects to Main {target_parent_id} (Dist: {min_dist:.2f} nm)")

        # --- B. RE-ORIENT HIERARCHY ---
        frag_df = reorient_fragment(frag_df, new_root_id)

        # <--- NEW: Tag this fragment (1, 2, 3...)
        # We do this AFTER reorient_fragment because that function might drop custom columns
        current_seg_id = i + 1
        frag_df['segment_id'] = current_seg_id

        # --- C. RE-INDEXING ---
        max_id = int(main_df['id'].max())
        offset = max_id + 100

        frag_df['id'] += offset
        mask = frag_df['p'] != -1
        frag_df.loc[mask, 'p'] += offset

        # --- D. LINK ---
        shifted_root_id = new_root_id + offset
        root_row_idx = frag_df.index[frag_df['id'] == shifted_root_id].tolist()

        if root_row_idx:
            frag_df.at[root_row_idx[0], 'p'] = target_parent_id

        # --- E. MERGE ---
        main_df = pd.concat([main_df, frag_df], ignore_index=True)

    # --- 3. FINAL CLEANUP ---
    main_df.reset_index(drop=True, inplace=True)

    if main_df['id'].duplicated().any():
        print("⚠️ Sanitizing IDs...")
        old_ids = main_df['id'].values
        new_ids = np.arange(1, len(main_df) + 1)
        id_map = dict(zip(old_ids, new_ids))

        main_df['p'] = main_df['p'].map(id_map).fillna(-1).astype(int)
        main_df['id'] = new_ids

    print(f"✅ Merged {len(main_df)} nodes.")
    return main_df

import plotly.graph_objects as go
import plotly.colors as pc

def plot_merged_neuron(df, title="Merged Neuron Verification"):
    """
    Plots the final merged DataFrame.
    Highlights:
    - Segments colored by ID (Main Arbor vs Fragments).
    - MAIN ROOT (Soma) marked with a large GOLD STAR.
    """
    if df is None or df.empty:
        print("❌ Dataframe is empty.")
        return

    print(f"📊 Plotting Merged Neuron ({len(df)} nodes)...")

    # 1. Build Fast Lookup (Global map)
    coords = df.set_index('id')[['x', 'y', 'z']].to_dict('index')

    fig = go.Figure()

    # --- 2. SEGMENT LOOP (Color by Fragment) ---
    if 'segment_id' in df.columns:
        segments = sorted(df['segment_id'].unique())
    else:
        segments = [0]
        df['segment_id'] = 0

    palette = pc.qualitative.Dark24 * (len(segments) // 24 + 1)

    for i, seg_id in enumerate(segments):
        seg_df = df[df['segment_id'] == seg_id]

        x_lines, y_lines, z_lines = [], [], []

        for _, row in seg_df.iterrows():
            parent_id = row['p']
            if parent_id in coords:
                parent = coords[parent_id]
                x_lines.extend([row['x'], parent['x'], None])
                y_lines.extend([row['y'], parent['y'], None])
                z_lines.extend([row['z'], parent['z'], None])

        # Color Logic
        if seg_id == 0:
            name = "Main Arbor"
            color = "black"
            width = 4
        else:
            name = f"Fragment {seg_id}"
            color = palette[i % len(palette)]
            width = 3

        fig.add_trace(go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode='lines',
            line=dict(color=color, width=width),
            name=name,
            hoverinfo='name'
        ))

    # --- 3. HIGHLIGHT THE MAIN ROOT ---
    # The root is the node with parent = -1
    roots = df[df['p'] == -1]

    if not roots.empty:
        print(f"📍 Found {len(roots)} root(s). Highlighting...")

        fig.add_trace(go.Scatter3d(
            x=roots['x'], y=roots['y'], z=roots['z'],
            mode='markers+text',
            text=["MAIN ROOT" if i==0 else "Disconnected Root" for i in range(len(roots))],
            textposition="top center",
            marker=dict(
                size=15,
                color='gold',
                symbol='diamond',
                line=dict(width=2, color='black')
            ),
            name='Main Root (Soma)',
            hovertemplate="<b>Root Node</b><br>ID: %{text}<br>X: %{x}<br>Y: %{y}<br>Z: %{z}<extra></extra>"
        ))

    # --- LAYOUT ---
    fig.update_layout(
        title=f"{title} <br><sup>{len(df)} nodes | {len(segments)} stitched segments</sup>",
        scene=dict(
            xaxis_title='X (nm)', yaxis_title='Y (nm)', zaxis_title='Z (nm)',
            aspectmode='data',
            bgcolor='white',
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        height=800,
        margin=dict(l=0, r=0, b=0, t=60)
    )

    fig.show()




def highlight_close_non_adjacent_points(df, threshold=1000, min_graph_hops=5):
    """
    Finds and plots pairs of nodes that are spatially close (< threshold)
    but topologically distant (not parent-child and not immediate neighbors).

    Parameters:
    - min_graph_hops: Ignores pairs connected by fewer than this many steps.
      (Prevents highlighting sharp bends or high-curvature segments as "gaps").
    """
    if df is None or df.empty:
        print("❌ DataFrame is empty.")
        return

    print(f"🔍 Scanning for close but non-adjacent points (< {threshold} nm)...")

    # --- 1. BUILD GRAPH (For Topological Distance) ---
    # We need to know if nodes are "neighbors" in the graph sense
    G = nx.Graph()
    G.add_nodes_from(df['id'])
    edges = df[df['p'] != -1][['id', 'p']].values
    G.add_edges_from(edges)

    # --- 2. SPATIAL QUERY ---
    coords = df[['x', 'y', 'z']].values
    ids = df['id'].values
    tree = cKDTree(coords)

    # Query all pairs within threshold
    # returns a set of (i, j) tuples where i < j
    pairs_idx = tree.query_pairs(r=threshold, output_type='ndarray')

    if len(pairs_idx) == 0:
        print("   ✅ No close points found.")
        return

    print(f"   🔹 Processed {len(pairs_idx)} spatial candidates. Filtering topology...")

    # --- 3. FILTERING ---
    close_calls = []

    # Map index back to ID
    idx_to_id = dict(enumerate(ids))

    for i, j in pairs_idx:
        id_a = idx_to_id[i]
        id_b = idx_to_id[j]

        # Check 1: Direct Parent-Child (The user's specific request)
        if G.has_edge(id_a, id_b):
            continue

        # Check 2: Graph Distance (The "Curvature" Filter)
        # If nodes are just 2-3 steps apart, it's just a bend, not a separate branch.
        # We only care if they are topologically FAR (or disconnected).
        try:
            # We use a cutoff to speed up 'shortest_path_length'
            # If path is longer than min_graph_hops, it raises NetworkXNoPath or returns length
            # Optimization: distinct components raise NetworkXNoPath immediately
            dist_hops = nx.shortest_path_length(G, source=id_a, target=id_b)

            if dist_hops < min_graph_hops:
                continue # Too close in the graph (just a bend)

        except nx.NetworkXNoPath:
            # They are in disconnected fragments! This is a PRIME candidate for a merge.
            pass

        # If we get here, it's a valid "Close Call"
        pos_a = coords[i]
        pos_b = coords[j]
        dist_euclid = np.linalg.norm(pos_a - pos_b)

        close_calls.append({
            'id_a': id_a, 'pos_a': pos_a,
            'id_b': id_b, 'pos_b': pos_b,
            'dist': dist_euclid
        })

    print(f"🚀 Found {len(close_calls)} pairs of close non-adjacent points.")

    # --- 4. VISUALIZATION ---
    fig = go.Figure()

    # A. Plot Skeleton (Context)
    # Using a fast line collection strategy
    node_map = df.set_index('id')[['x', 'y', 'z']].to_dict('index')
    x_skel, y_skel, z_skel = [], [], []
    for _, row in df.iterrows():
        pid = row['p']
        if pid != -1 and pid in node_map:
            p = node_map[pid]
            x_skel.extend([row['x'], p['x'], None])
            y_skel.extend([row['y'], p['y'], None])
            z_skel.extend([row['z'], p['z'], None])

    fig.add_trace(go.Scatter3d(
        x=x_skel, y=y_skel, z=z_skel,
        mode='lines',
        line=dict(color='lightgrey', width=5),
        opacity=1, # CHANGED: Increased from 0.5 to 0.7 for less transparency
        name='Skeleton'
    ))

    # B. Plot "Close Calls" (Red Lines)
    if close_calls:
        x_gap, y_gap, z_gap = [], [], []

        for item in close_calls:
            p1, p2 = item['pos_a'], item['pos_b']
            x_gap.extend([p1[0], p2[0], None])
            y_gap.extend([p1[1], p2[1], None])
            z_gap.extend([p1[2], p2[2], None])

        fig.add_trace(go.Scatter3d(
            x=x_gap, y=y_gap, z=z_gap,
            mode='lines',
            line=dict(color='red', width=4),
            name=f'Proximities (< {threshold}nm)'
        ))

        # Add markers at the tips
        fig.add_trace(go.Scatter3d(
            x=[item['pos_a'][0] for item in close_calls] + [item['pos_b'][0] for item in close_calls],
            y=[item['pos_a'][1] for item in close_calls] + [item['pos_b'][1] for item in close_calls],
            z=[item['pos_a'][2] for item in close_calls] + [item['pos_b'][2] for item in close_calls],
            mode='markers',
            marker=dict(size=4, color='red', symbol='circle-open'),
            showlegend=False
        ))

    fig.update_layout(
        title=f"Close Non-Adjacent Points (Threshold: {threshold}nm, Min Hops: {min_graph_hops})",
        scene=dict(aspectmode='data', bgcolor='white'),
        height=800
    )
    fig.show()

    return pd.DataFrame(close_calls)



def annotate_stitched_neuron(stitched_df, neuron_id, annotation_url=LOCAL_URL):
    """
    Takes the stitched DataFrame, aligns it to the Neuroglancer Annotation Cloud,
    and maps the closest labels to the skeleton nodes.

    Args:
        stitched_df (pd.DataFrame): Output from stitch_neuron_fragments_smart.
        neuron_id (str/int): The ID to look up in the annotation layer.
        annotation_url (str): Path to the precomputed annotations.

    Returns:
        pd.DataFrame: The original stitched_df with a new 'annotated_type' column.
    """
    if stitched_df is None or stitched_df.empty:
        print("❌ Input DataFrame is empty.")
        return stitched_df

    print(f"📥 Loading Annotations for Neuron {neuron_id}...")

    # --- 1. LOAD REFERENCE ANNOTATIONS (The Cloud) ---
    try:
        reader = neuroglancer.read_precomputed_annotations.AnnotationReader(annotation_url)
        # Handle different relationship key names often found in Precomputed formats
        rel_key = 'skeleton_id' if 'skeleton_id' in reader.relationships else 'associated_segments'

        # specific lookup for this neuron_id
        anns = reader.relationships[rel_key][int(neuron_id)]

        ann_coords = []
        ann_labels = []

        for a in anns:
            try:
                # Extract Point
                pt = a.point if hasattr(a, 'point') else a['point']

                # Extract Label Code
                props = []
                if hasattr(a, 'props'): props = a.props
                elif isinstance(a, dict): props = a.get('props', [])
                elif hasattr(a, 'properties'): props = getattr(a, 'properties', [])

                label_code = str(props[0]) if props else '999'

                ann_coords.append(pt)
                ann_labels.append(label_code)
            except:
                continue

        if not ann_coords:
            print("⚠️ No annotation points found for this ID.")
            stitched_df['annotated_type'] = "Unknown"
            return stitched_df

        ann_arr = np.array(ann_coords)
        label_arr = np.array(ann_labels)

        # Calculate Annotation Bounds (Target Space)
        ann_min, ann_max = ann_arr.min(axis=0), ann_arr.max(axis=0)
        ann_range = ann_max - ann_min
        ann_center = ann_arr.mean(axis=0)

        print(f"🔹 Found {len(ann_arr)} annotation points.")

    except Exception as e:
        print(f"❌ Error loading annotations: {e}")
        return stitched_df

    # --- 2. CALCULATE ALIGNMENT (Source -> Target) ---
    # We use temporary coordinates so we don't distort the actual skeleton structure
    # in the output DataFrame.

    # Get Skeleton Bounds (Source Space)
    skel_coords = stitched_df[['x', 'y', 'z']].values.astype(float)
    skel_min = skel_coords.min(axis=0)
    skel_max = skel_coords.max(axis=0)
    skel_range = skel_max - skel_min

    # A. Calculate Scale Factor (Annotation Range / Skeleton Range)
    # Avoid division by zero
    scale = np.divide(ann_range, skel_range, out=np.zeros_like(ann_range), where=skel_range!=0)

    # B. Apply Scale to Temporary Coords
    aligned_coords = skel_coords * scale

    # C. Calculate Offset (Center to Center)
    aligned_center = aligned_coords.mean(axis=0)
    offset = ann_center - aligned_center

    # D. Apply Offset
    aligned_coords += offset

    print(f"✅ Alignment Calculated | Scale: {scale.round(2)}")

    # --- 3. KDTREE NEAREST NEIGHBOR SEARCH ---
    print("🔍 Mapping labels (Max Dist: 5000nm)...")

    # Build Tree on the Annotation Cloud
    tree = cKDTree(ann_arr)

    # Query: Find closest annotation for every skeleton node
    MAX_DIST = 5000.0  # 5 microns
    distances, indices = tree.query(aligned_coords, k=1, distance_upper_bound=MAX_DIST)

    # --- 4. ASSIGN LABELS ---
    mapped_types = []

    for idx in indices:
        if idx == len(ann_arr):
            # No neighbor found within MAX_DIST
            mapped_types.append('Unknown')
        else:
            code = label_arr[idx]
            text_label = ANNOTATION_LABEL_MAP.get(code, f"Unknown ({code})")
            mapped_types.append(text_label)

    # Apply to DataFrame
    stitched_df['annotated_type'] = mapped_types

    # Optional: Save the distance so you can debug alignment issues later
    stitched_df['dist_to_ann'] = distances

    # Stats
    counts = stitched_df['annotated_type'].value_counts()
    print(f"📊 Annotation Results:\n{counts}")

    return stitched_df


def plot_annotated_neuron(df, title="Annotated Neuron Skeleton"):
    """
    Plots the neuron skeleton color-coded by 'annotated_type'.

    Colors:
    - Axon: Red
    - Dendrite: Blue
    - Soma: Gold (with large marker)
    - AIS: Green
    - Myelinated: Orange
    - Unknown: Grey
    """
    if df is None or df.empty:
        print("❌ Dataframe is empty.")
        return

    print(f"📊 Plotting Annotated Neuron ({len(df)} nodes)...")

    # 1. Define Color Map
    COLOR_MAP = {
        'Axon': 'crimson',
        'Dendrite': 'royalblue',
        'Soma': 'gold',
        'AIS': 'limegreen',
        'Cilium': 'magenta',
        'Astrocyte': 'cyan',
        'Myelinated Axon': 'darkorange',
        'Myelinated Fragment': 'orange',
        'Unknown': 'lightgrey'
    }

    # 2. Build Fast Lookup for Parents
    # We need coordinates indexed by ID to draw lines from child to parent
    coords = df.set_index('id')[['x', 'y', 'z']].to_dict('index')

    fig = go.Figure()

    # 3. Group by Type and Plot
    # Plotting by group is much faster than adding a trace per line
    unique_types = df['annotated_type'].unique()

    for ann_type in unique_types:
        # Filter data for this type
        type_df = df[df['annotated_type'] == ann_type]

        x_lines, y_lines, z_lines = [], [], []

        for _, row in type_df.iterrows():
            parent_id = row['p']

            # Skip root nodes (parent = -1) or missing parents
            if parent_id != -1 and parent_id in coords:
                parent = coords[parent_id]

                # Add line segment (Child -> Parent -> None)
                x_lines.extend([row['x'], parent['x'], None])
                y_lines.extend([row['y'], parent['y'], None])
                z_lines.extend([row['z'], parent['z'], None])

        # Select color (default to grey if not in map)
        color = COLOR_MAP.get(ann_type, 'grey')

        # Add Lines Trace
        fig.add_trace(go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode='lines',
            line=dict(color=color, width=3),
            name=ann_type,
            hoverinfo='name'
        ))

    # 4. Highlight Somas specifically with markers
    somas = df[df['annotated_type'] == 'Soma']
    if not somas.empty:
        fig.add_trace(go.Scatter3d(
            x=somas['x'], y=somas['y'], z=somas['z'],
            mode='markers',
            marker=dict(size=12, color='gold', line=dict(width=2, color='black')),
            name='Soma (Nodes)',
            hoverinfo='name'
        ))

    # --- LAYOUT ---
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (nm)', yaxis_title='Y (nm)', zaxis_title='Z (nm)',
            aspectmode='data', # Keeps physical proportions correct
            bgcolor='white',
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        height=800,
        margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(title="Compartment Type")
    )

    fig.show()


def scale_dataframe_to_nm(df, resolution=[8, 8, 33]):
    """
    Converts a DataFrame from Voxel Indices -> Physical Nanometers.

    Input columns expected: ['x', 'y', 'z', 'r']
    Resolution: [res_x, res_y, res_z] in nm/voxel (e.g., [8, 8, 32])
    """
    if df is None or df.empty:
        print("❌ DataFrame is empty.")
        return df

    print(f"📏 Scaling data by resolution: {resolution} nm/voxel...")
    print(f"   Before (Head): X={df['x'].iloc[0]:.1f}, Y={df['y'].iloc[0]:.1f}, Z={df['z'].iloc[0]:.1f}")

    # Create a copy so we don't corrupt the original data
    df_nm = df.copy()

    # 1. Apply Anisotropic Scaling
    # X_nm = X_voxel * 8
    # Y_nm = Y_voxel * 8
    # Z_nm = Z_voxel * 32
    df_nm['x'] = df_nm['x'] * resolution[0]
    df_nm['y'] = df_nm['y'] * resolution[1]
    df_nm['z'] = df_nm['z'] * resolution[2]

    # 2. Scale the Radius
    # Radius is a 2D cross-section measurement.
    # In anisotropic data, radius is usually calculated in the high-res plane (XY).
    # We scale 'r' by the average of X and Y resolution.
    xy_scale = (resolution[0] + resolution[1]) / 2.0
    df_nm['r'] = df_nm['r'] * xy_scale

    # 3. Add metadata so you know it's converted
    df_nm.attrs['unit'] = 'nm'
    df_nm.attrs['resolution'] = resolution

    print(f"   After  (Head): X={df_nm['x'].iloc[0]:.1f}, Y={df_nm['y'].iloc[0]:.1f}, Z={df_nm['z'].iloc[0]:.1f}")
    print("✅ Conversion to Nanometers complete.")

    return df_nm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

def plot_voxel_vs_physical(df_voxel, df_physical, title="Voxel vs Physical Space"):
    """
    Plots the raw Voxel dataframe (Left) and the scaled Physical dataframe (Right)
    to visualize the Anisotropic correction (the "stretching" of Z).
    """
    if df_voxel is None or df_physical is None:
        print("❌ Missing DataFrames.")
        return

    print("📊 Generating comparison plot...")

    # Create Subplots: 1 Row, 2 Columns
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Raw Voxel Space (Squashed Z)", "Physical Space (Corrected)")
    )

    # --- HELPER TO EXTRACT LINES ---
    def get_lines(df):
        # Fast dictionary lookup
        coords = df.set_index('id')[['x', 'y', 'z']].to_dict('index')
        x, y, z = [], [], []
        for _, row in df.iterrows():
            pid = row['p']
            if pid != -1 and pid in coords:
                parent = coords[pid]
                x.extend([row['x'], parent['x'], None])
                y.extend([row['y'], parent['y'], None])
                z.extend([row['z'], parent['z'], None])
        return x, y, z

    # --- 1. LEFT PANEL: VOXELS ---
    vx, vy, vz = get_lines(df_voxel)
    fig.add_trace(go.Scatter3d(
        x=vx, y=vy, z=vz,
        mode='lines',
        line=dict(color='blue', width=1),
        name='Voxels (Raw)',
        hoverinfo='skip'
    ), row=1, col=1)

    # --- 2. RIGHT PANEL: PHYSICAL ---
    px, py, pz = get_lines(df_physical)
    fig.add_trace(go.Scatter3d(
        x=px, y=py, z=pz,
        mode='lines',
        line=dict(color='red', width=1),
        name='Physical (nm)',
        hoverinfo='skip'
    ), row=1, col=2)

    # --- LAYOUT SETTINGS ---
    # Crucial: We force 'data' aspect mode so Plotly respects the actual numbers.
    # In Voxel space, Z is small (e.g. 700), so it will look flat.
    # In Physical space, Z is big (e.g. 25000), so it will look tall.

    fig.update_layout(
        title=title,
        height=800,
        showlegend=False,
        scene=dict(
            aspectmode='data',
            xaxis_title='X (px)', yaxis_title='Y (px)', zaxis_title='Z (slices)'
        ),
        scene2=dict(
            aspectmode='data',
            xaxis_title='X (nm)', yaxis_title='Y (nm)', zaxis_title='Z (nm)'
        )
    )

    fig.show()




def plot_soma_skeleton(df, soma_ids, centroid):
    """
    Plots the internal skeleton of the Soma.
    FIXED: Handles duplicate IDs by creating a unique 'file_id' key.
    """
    if df is None or not soma_ids:
        print("❌ No Soma data to plot.")
        return

    print(f"🧬 Plotting Soma Skeleton ({len(soma_ids)} nodes)...")

    # 1. Create a Unique Key for Lookup
    # We combine 'file_source' and 'id' to ensure uniqueness
    # (e.g. "file1.swc_1" vs "file2.swc_1")
    df_safe = df.copy()
    if 'file_source' not in df_safe.columns:
        df_safe['file_source'] = 'single_file'

    df_safe['uid'] = df_safe['file_source'].astype(str) + "_" + df_safe['id'].astype(str)

    # Also create the Parent UID so we know who to look for
    df_safe['p_uid'] = df_safe['file_source'].astype(str) + "_" + \
                       (df_safe['p'] if 'p' in df_safe.columns else df_safe['parent']).astype(str)

    # 2. Filter Data for Soma Only
    # We filter based on the original IDs passed in 'soma_ids'
    soma_df = df_safe[df_safe['id'].isin(soma_ids)].copy()

    # Create the fast lookup using the UNIQUE ID
    coords = df_safe.set_index('uid')[['x', 'y', 'z']].to_dict('index')

    fig = go.Figure()

    # --- 3. BUILD SOMA SKELETON (LINES) ---
    x_lines, y_lines, z_lines = [], [], []

    for _, row in soma_df.iterrows():
        p_uid = row['p_uid']
        parent_id = int(row['p'] if 'p' in row else row['parent'])

        # Draw line if:
        # 1. Parent exists in our unique lookup
        # 2. Parent's original ID is also in the Soma Cluster (Internal connection)
        if p_uid in coords and parent_id in soma_ids:
            parent = coords[p_uid]
            x_lines.extend([row['x'], parent['x'], None])
            y_lines.extend([row['y'], parent['y'], None])
            z_lines.extend([row['z'], parent['z'], None])

    # Trace 1: The Skeleton Lines
    fig.add_trace(go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines',
        line=dict(color='green', width=4),
        name='Soma Skeleton',
        opacity=0.8
    ))

    # Trace 2: The Nodes (Context)
    fig.add_trace(go.Scatter3d(
        x=soma_df['x'], y=soma_df['y'], z=soma_df['z'],
        mode='markers',
        marker=dict(size=3, color='green', opacity=0.3),
        name='Soma Nodes'
    ))

    # --- 4. PLOT CENTROID MARKER ---
    if centroid is not None:
        fig.add_trace(go.Scatter3d(
            x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
            mode='markers',
            marker=dict(
                size=15,
                color='black',
                symbol='cross',
                line=dict(width=2, color='white')
            ),
            name='Centroid'
        ))

    # --- LAYOUT ---
    fig.update_layout(
        title="Soma Skeleton & Centroid Check",
        scene=dict(
            aspectmode='data',
            xaxis_title='X (nm)', yaxis_title='Y (nm)', zaxis_title='Z (nm)',
            bgcolor='white',
            xaxis=dict(showbackground=False, showgrid=False, showticklabels=False),
            yaxis=dict(showbackground=False, showgrid=False, showticklabels=False),
            zaxis=dict(showbackground=False, showgrid=False, showticklabels=False),
        ),
        width=1000, height=800
    )
    fig.show()
def find_stable_soma_centroid(df, min_samples=50, step_eps=100, max_eps=5000, REQUIRED_STABILITY=5):
    """
    1. Filters Soma points.
    2. Adaptive DBSCAN: Finds the largest cluster that remains stable in size.
    3. PLOTS the Stability Curve.
    4. PLOTS the Resulting 3D Cluster + Centroid + Epsilon Sphere.
    5. Returns centroid of the stable cluster (SNAPPED to the nearest skeleton node).
    """
    print("📍 Searching for Stable Soma Cluster...")

    # --- 1. Filter Soma Points ---
    soma_df = df[ (df['annotated_type'] == 'Soma') | (df['annotated_type'] == '3') ].copy()
    coords = soma_df[['x', 'y', 'z']].values

    if len(coords) < min_samples:
        print(f"⚠️ Small Soma ({len(coords)} pts). Skipping clustering.")
        return np.median(coords, axis=0), set(soma_df['id'])

    # --- 2. Adaptive Stability Loop ---
    history = []
    prev_size = 0
    stability_count = 0

    best_mask = None
    best_eps = max_eps # Default fallback
    found_plateau = False

    # We need to track the actual mask corresponding to the best_eps
    final_cluster_mask = np.zeros(len(coords), dtype=bool)

    start_eps = max(100, step_eps)

    for eps in range(start_eps, max_eps + 1, step_eps):
        # Run Clustering
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(coords)
        labels = clustering.labels_

        # Identify Largest Cluster
        valid_labels = labels[labels != -1]

        if len(valid_labels) == 0:
            current_size = 0
            current_mask = np.zeros(len(labels), dtype=bool)
        else:
            counts = np.bincount(valid_labels)
            largest_cluster_id = np.argmax(counts)
            current_size = counts[largest_cluster_id]
            current_mask = (labels == largest_cluster_id)

        history.append({'eps': eps, 'size': current_size})

        # Check Stability
        if current_size > 0 and abs(current_size - prev_size) < (prev_size * 0.01): #Allow <1% fluctuation
             stability_count += 1
        else:
             stability_count = 0

        # Found Plateau?
        if stability_count >= REQUIRED_STABILITY and not found_plateau:
            # We select the START of the plateau as the optimal point (tightest fit)
            best_eps = eps - (step_eps * (REQUIRED_STABILITY // 2))

            # Important: We must re-run or cache the mask for THIS specific epsilon
            # For simplicity, we assume current_mask is close enough since it's stable
            final_cluster_mask = current_mask
            print(f"✅ Stable Cluster Found: Eps={best_eps}nm, Size={current_size} nodes")

            found_plateau = True
            # Continue briefly to show plateau
            if len(history) > (len(history) + 5): break

        prev_size = current_size

    # Fallback
    if not found_plateau:
        print("⚠️ No plateau found. Using max_eps.")
        best_eps = max_eps
        final_cluster_mask = (labels != -1)

    # --- 3. PLOT STABILITY CURVE ---
    df_hist = pd.DataFrame(history)
    fig_curve = go.Figure()
    fig_curve.add_trace(go.Scatter(x=df_hist['eps'], y=df_hist['size'], mode='lines+markers', name='Cluster Size'))

    # Highlight Selected
    sel_row = df_hist.iloc[(df_hist['eps'] - best_eps).abs().argsort()[:1]]
    if not sel_row.empty:
        fig_curve.add_trace(go.Scatter(
            x=[sel_row['eps'].values[0]], y=[sel_row['size'].values[0]],
            mode='markers+text', text=[f"<b>Selected: {int(sel_row['size'].values[0])}</b>"], textposition="top left",
            marker=dict(size=12, color='red', symbol='circle-open', line=dict(width=2))
        ))

    fig_curve.update_layout(title="Stability Analysis", xaxis_title="Epsilon (nm)", yaxis_title="Cluster Size", height=400)
    fig_curve.show()

    # --- 4. CALCULATE CENTROID ---
    strict_cluster_df = soma_df[final_cluster_mask]
    if strict_cluster_df.empty:
        print("❌ Error: Cluster Empty.")
        return None, set()

    points = strict_cluster_df[['x', 'y', 'z']].values
    geo_centroid = np.median(points, axis=0)

    # Snap to nearest node
    dists = np.linalg.norm(points - geo_centroid, axis=1)
    closest_idx = np.argmin(dists)
    snapped_centroid = points[closest_idx]

    # --- 5. PLOT 3D RESULT (SOMA + SPHERE) ---
    print(f"📊 Plotting Result with Epsilon Sphere ({best_eps} nm)...")
    fig_3d = go.Figure()

    # A. The Soma Points (Green)
    fig_3d.add_trace(go.Scatter3d(
        x=strict_cluster_df['x'], y=strict_cluster_df['y'], z=strict_cluster_df['z'],
        mode='markers', marker=dict(size=3, color='green', opacity=0.5),
        name='Stable Soma Cluster'
    ))

    # B. The Centroid (Black X)
    fig_3d.add_trace(go.Scatter3d(
        x=[snapped_centroid[0]], y=[snapped_centroid[1]], z=[snapped_centroid[2]],
        mode='markers', marker=dict(size=10, color='black', symbol='x'),
        name='Centroid'
    ))

    # C. The Epsilon Sphere (Wireframe)
    # Visualizes the "Reach" of DBSCAN at the chosen Epsilon
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 10)
    x_sphere = snapped_centroid[0] + best_eps * np.outer(np.cos(u), np.sin(v))
    y_sphere = snapped_centroid[1] + best_eps * np.outer(np.sin(u), np.sin(v))
    z_sphere = snapped_centroid[2] + best_eps * np.outer(np.ones(np.size(u)), np.cos(v))

    fig_3d.add_trace(go.Surface(
        x=x_sphere, y=y_sphere, z=z_sphere,
        opacity=0.2, showscale=False, colorscale=[[0, 'blue'], [1, 'blue']],
        name=f'Epsilon Sphere (R={best_eps})'
    ))

    fig_3d.update_layout(
        title=f"Soma Cluster & Epsilon Sphere (R={best_eps} nm)",
        scene=dict(aspectmode='data'),
        height=700
    )
    fig_3d.show()

    return snapped_centroid, set(strict_cluster_df['id'])


def plot_soma_points(df, soma_ids, title="Highlighted Soma Points"):
    """
    Plots the entire neuron (Ghost) and highlights the specific 'soma_ids' in Gold.
    Useful for debugging DBSCAN or Expansion results.
    """
    if df is None or not soma_ids:
        print("❌ Missing data or soma_ids.")
        return

    print(f"📍 Plotting {len(soma_ids)} highlighted soma nodes...")

    # Ensure efficient lookup
    target_ids = set(soma_ids)

    fig = go.Figure()

    # --- 1. CONTEXT: GHOST SKELETON (Faint) ---
    # Plotting lines is heavy, so we just plot dots for context if it's large,
    # or lines if it's manageable. Here we do lines for clarity.

    coords = df.set_index('id')[['x', 'y', 'z']].to_dict('index')
    x_ghost, y_ghost, z_ghost = [], [], []

    for _, row in df.iterrows():
        pid = row['p']
        if pid != -1 and pid in coords:
            parent = coords[pid]
            x_ghost.extend([row['x'], parent['x'], None])
            y_ghost.extend([row['y'], parent['y'], None])
            z_ghost.extend([row['z'], parent['z'], None])

    fig.add_trace(go.Scatter3d(
        x=x_ghost, y=y_ghost, z=z_ghost,
        mode='lines',
        line=dict(color='lightgrey', width=1),
        opacity=0.3,
        name='Context (Skeleton)'
    ))

    # --- 2. HIGHLIGHT: SOMA IDS ---
    # Filter the dataframe for just the target IDs
    soma_df = df[df['id'].isin(target_ids)]

    if not soma_df.empty:
        fig.add_trace(go.Scatter3d(
            x=soma_df['x'], y=soma_df['y'], z=soma_df['z'],
            mode='markers',
            marker=dict(
                size=5,
                color='gold',
                line=dict(width=1, color='darkgoldenrod')
            ),
            name=f'Soma Points ({len(soma_df)})',
            hovertemplate="ID: %{text}<br>Type: %{customdata}<extra></extra>",
            text=soma_df['id'],
            customdata=soma_df['annotated_type']
        ))
    else:
        print("⚠️ None of the provided soma_ids were found in the DataFrame.")

    # --- LAYOUT ---
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (nm)', yaxis_title='Y (nm)', zaxis_title='Z (nm)',
            aspectmode='data',
            bgcolor='white'
        ),
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()


def find_exits_by_sphere_intersection(df, centroid, radius, tolerance=2000):
    """
    Finds exits via sphere intersection AND re-roots the dataframe at the soma center.

    Returns:
        unique_exits (pd.DataFrame): The exit points found.
        df (pd.DataFrame): The modified dataframe, re-rooted at the soma center.
    """
    if df is None or df.empty:
        print("❌ DataFrame is empty.")
        return None, None

    print(f"🔍 Finding exits via Sphere Intersection (Radius: {radius} ± {tolerance} nm)...")

    # --- 1. SETUP GRAPH ---
    # We use an UNDIRECTED graph to allow traversal regardless of current root
    G = nx.Graph()
    G.add_nodes_from(df['id'])
    edges = df[df['p'] != -1][['id', 'p']].values
    G.add_edges_from(edges)

    node_map = df.set_index('id').to_dict('index')
    coords = df[['x', 'y', 'z']].values
    node_ids = df['id'].values

    # Find Root (Soma Center) based on geometric distance
    dists_to_center = np.linalg.norm(coords - centroid, axis=1)
    root_idx = np.argmin(dists_to_center)
    root_id = node_ids[root_idx]

    # --- 2. FIND SHELL INTERSECTIONS ---
    mask_shell = (dists_to_center >= (radius - tolerance)) & (dists_to_center <= (radius + tolerance))
    shell_ids = node_ids[np.where(mask_shell)[0]]

    if len(shell_ids) == 0:
        print("   ⚠️ No nodes found in shell. Increase tolerance.")
        return None, df

    # --- 3. CLUSTER & SELECT TARGETS ---
    G_shell = G.subgraph(shell_ids)
    components = list(nx.connected_components(G_shell))
    targets = []

    for comp in components:
        comp_list = list(comp)
        comp_dists = [np.linalg.norm(np.array([node_map[n]['x'], node_map[n]['y'], node_map[n]['z']]) - centroid) for n in comp_list]
        targets.append(comp_list[np.argmin(comp_dists)])

    print(f"   🔹 Identified {len(targets)} unique branch crossings.")

    # --- 4. TRACE PATHS ---
    exit_candidates = []

    for target in targets:
        try:
            # Shortest path uses the graph G (undirected)
            path = nx.shortest_path(G, source=root_id, target=target)

            for i in range(len(path) - 1):
                curr_node = path[i]
                next_node = path[i+1]

                curr_type = node_map[curr_node].get('annotated_type', 'Unknown')
                next_type = node_map[next_node].get('annotated_type', 'Unknown')

                is_curr_soma = (str(curr_type) in ['Soma', '1', '3'])
                is_next_soma = (str(next_type) in ['Soma', '1', '3'])

                if is_curr_soma and not is_next_soma:
                    exit_candidates.append({
                        'id': next_node,
                        'exit_to_type': next_type,
                        'target_used': target
                    })
                    break

        except nx.NetworkXNoPath:
            continue

    # --- 5. DEDUPLICATE ---
    exit_df = pd.DataFrame(exit_candidates)

    unique_exits = pd.DataFrame()
    if not exit_df.empty:
        unique_exits = exit_df.drop_duplicates(subset='id').copy()
        unique_exits['x'] = unique_exits['id'].apply(lambda x: node_map[x]['x'])
        unique_exits['y'] = unique_exits['id'].apply(lambda x: node_map[x]['y'])
        unique_exits['z'] = unique_exits['id'].apply(lambda x: node_map[x]['z'])
        print(f"🚀 Found {len(unique_exits)} Unique Branch Starts.")
    else:
        print("   ⚠️ No transitions found.")

    # --- 6. PLOT RESULTS ---
    if not unique_exits.empty:
        plot_sphere_exits(df, unique_exits, centroid, radius, targets)

    # --- 7. RE-ROOT DATAFRAME AT SOMA CENTER ---
    print(f"🔧 Re-rooting DataFrame at closest soma node (ID: {root_id})...")

    # A. Determine new parentage flow via BFS from new root
    bfs_tree = nx.bfs_tree(G, source=root_id) # Returns a DiGraph rooted at root_id

    # Create a mapping: Node -> Parent
    # In a DiGraph (bfs_tree), edges go Parent -> Child.
    # So for edge (u, v), u is parent of v.
    new_parents = {child: parent for parent, child in bfs_tree.edges()}

    # B. Apply updates safely
    # We map the IDs to their new parents using the dictionary
    # Nodes not in the BFS tree (unreachable) remain unchanged
    df['p'] = df['id'].map(new_parents).fillna(df['p']).astype(int)

    # Explicitly set the new Root's parent to -1
    df.loc[df['id'] == root_id, 'p'] = -1

    print(f"   ✅ Tree topology updated. Node {root_id} is now the Root (p=-1).")

    return unique_exits, df


def plot_sphere_exits(df, exits, centroid, radius, targets):
    """
    Helper function to visualize the sphere intersection results.
    """
    print("📊 Plotting Sphere Intersections...")
    fig = go.Figure()

    # 1. Soma Cloud (Context) - Filter for Soma/1/3
    soma_df = df[df['annotated_type'].isin(['Soma', '1', '3'])]
    fig.add_trace(go.Scatter3d(
        x=soma_df['x'], y=soma_df['y'], z=soma_df['z'],
        mode='markers', marker=dict(size=2, color='green', opacity=0.1),
        name='Soma Cloud'
    ))

    # 2. The Intersection Sphere Hits (Cyan Circles)
    # These are the points out at ~20 microns
    target_coords = df[df['id'].isin(targets)]
    fig.add_trace(go.Scatter3d(
        x=target_coords['x'], y=target_coords['y'], z=target_coords['z'],
        mode='markers', marker=dict(size=4, color='cyan', symbol='circle-open'),
        name=f'Sphere Hits (R={radius})'
    ))

    # 3. The Exit Points (Red Diamonds)
    # These are the actual branch starts we calculated
    fig.add_trace(go.Scatter3d(
        x=exits['x'], y=exits['y'], z=exits['z'],
        mode='markers',
        marker=dict(size=6, color='red', symbol='diamond'),
        name='Calculated Branch Starts'
    ))

    # 4. Centroid (Black X)
    fig.add_trace(go.Scatter3d(
        x=[centroid[0]], y=[centroid[1]], z=[centroid[2]],
        mode='markers', marker=dict(size=5, color='black', symbol='x'),
        name='Centroid'
    ))

    # 5. Radial Vectors (Visual Check)
    # Draw faint lines from centroid to the exits to verify radial path
    x_lines, y_lines, z_lines = [], [], []
    for _, row in exits.iterrows():
        x_lines.extend([centroid[0], row['x'], None])
        y_lines.extend([centroid[1], row['y'], None])
        z_lines.extend([centroid[2], row['z'], None])

    fig.add_trace(go.Scatter3d(
        x=x_lines, y=y_lines, z=z_lines,
        mode='lines',
        line=dict(color='red', width=1, dash='dot'),
        opacity=0.3,
        name='Radial Paths'
    ))

    fig.update_layout(
        title="Sphere Intersection Analysis",
        scene=dict(aspectmode='data', bgcolor='white'),
        height=800
    )
    fig.show()



def collapse_soma_to_root(df_, soma_ids, centroid, exit_df):
    """
    Collapses the soma cluster by selecting the segment closest to the centroid
    and promoting it to be the 'Virtual Soma' (ID 0).
    """
    df = df_.copy(deep=True)
    if df is None or exit_df is None:
        print("❌ Missing input DataFrames.")
        return None

    print(f"📉 Collapsing Soma (Existing Segment Promotion Mode)...")


    # --- 1. Calculate Virtual Radius ---
    exit_coords = exit_df[['x', 'y', 'z']].values
    dists = np.linalg.norm(exit_coords - centroid, axis=1)
    visual_radius = np.mean(dists) if len(dists) > 0 else 500.0
    print(f"   🔹 Virtual Soma Size: {visual_radius:.2f} nm")

    # --- 2. FIND CLOSEST SEGMENT (New Step) ---
    # We look for the node within soma_ids that is geometrically closest to the centroid
    soma_nodes = df[df['id'].isin(soma_ids)].copy()

    if soma_nodes.empty:
        print("❌ Error: No soma nodes found in DataFrame.")
        return None

    # Calculate distance from each soma node to the centroid
    node_dists = np.linalg.norm(soma_nodes[['x', 'y', 'z']].values - centroid, axis=1)
    best_idx = np.argmin(node_dists)
    best_soma_id = soma_nodes.iloc[best_idx]['id']

    print(f"   🎯 Selected existing segment ID {best_soma_id} as the new Root.")

    # --- 3. Nuclear Deletion (Modified) ---
    ids_to_delete = set(soma_ids)
    exit_ids = set(exit_df['id'].values)

    # CRITICAL: Do NOT delete the chosen root candidate!
    ids_to_delete.discard(best_soma_id)

    # Safety: Do not delete exits
    ids_to_delete = ids_to_delete - exit_ids

    print(f"   🗑️ Deleting {len(ids_to_delete)} soma nodes.")

    # Create Clean DataFrame
    new_df = df[~df['id'].isin(ids_to_delete)].copy()

    # --- 4. Handle ID 0 Conflicts ---
    # If 0 exists in the remaining data (and it's NOT our chosen root), move it.
    # Note: If best_soma_id IS 0, we handle it in the extraction step below.
    if 0 in new_df['id'].values and best_soma_id != 0:
        safe_id = int(new_df['id'].max()) + 9999
        new_df.loc[new_df['id'] == 0, 'id'] = safe_id
        new_df.loc[new_df['p'] == 0, 'p'] = safe_id

    # --- 5. Extract & Promote Root ---
    # Isolate the chosen node row
    root_row = new_df[new_df['id'] == best_soma_id].copy()

    # Remove it from the main body (so we can put it at the top as ID 0)
    new_df = new_df[new_df['id'] != best_soma_id].copy()

    # Update properties to make it the Root (ID 0)
    root_row['id'] = 0
    root_row['p'] = -1
    root_row['r'] = visual_radius # Assign visual radius
    root_row['type'] = 1          # Ensure it is labeled as Soma
    root_row['annotated_type'] = 'Soma'
    root_row['file_source'] = 'promoted_root'

    # This becomes our root_df
    root_df = root_row

    # --- 6. REWIRE EXITS ---
    # Force the 'exit points' to point to Root (0)
    # Note: If the exit point was the best_soma_id, it is now in root_df (p=-1),
    # so it won't be found in new_df, which is correct (Root shouldn't point to itself).
    mask_exits = new_df['id'].isin(exit_ids)

    if mask_exits.any():
        new_df.loc[mask_exits, 'p'] = 0
        print(f"   🔗 Connected {mask_exits.sum()} exit points to Root 0.")

    # --- 7. Final Merge & Cleanup ---
    collapsed_df = pd.concat([root_df, new_df], ignore_index=True)

    collapsed_df.sort_values(by='id', inplace=True)
    collapsed_df.reset_index(drop=True, inplace=True)

    # --- 8. Reindex (Standardize IDs) ---
    print("   🔄 Renumbering IDs...")
    old_ids = collapsed_df['id'].values
    new_ids = np.arange(0, len(collapsed_df)) # Start at 1

    # Create mapping: Old ID -> New ID
    id_map = dict(zip(old_ids, new_ids))

    collapsed_df['id'] = new_ids

    # Remap parents
    collapsed_df['p'] = collapsed_df['p'].map(id_map).fillna(-1).astype(int)

    # Ensure Root (ID 1 now) is -1
    collapsed_df.loc[collapsed_df['id'] == 1, 'p'] = -1

    print(f"✅ Soma Collapsed. Final Node Count: {len(collapsed_df)}")
    return collapsed_df




def plot_collapsed_neuron(df_, title="Collapsed Soma Skeleton"):
    """
    Plots the skeleton with:
    1. The 'Virtual Soma' (ID 0) as a large Gold Sphere.
    2. ANY other disconnected roots (p=-1) as Red X markers (Orphans).
    3. The rest of the skeleton color-coded by type.
    """
    df = df_.copy(deep=True)
    if df is None or df.empty:
        print("❌ DataFrame is empty.")
        return

    print(f"📊 Plotting Collapsed Neuron ({len(df)} nodes)...")

    # --- 1. SETUP COLORS & LOOKUP ---
    COLOR_MAP = {
        'Axon': 'crimson',
        'Dendrite': 'royalblue',
        'Soma': 'gold',
        'AIS': 'limegreen',
        'Cilium': 'magenta',
        'Myelinated Axon': 'darkorange',
        'Unknown': 'lightgrey'
    }

    # Fast Coordinate Lookup
    coords = df.set_index('id')[['x', 'y', 'z']].to_dict('index')

    fig = go.Figure()

    # --- 2. DRAW SKELETON LINES ---
    unique_types = df['annotated_type'].unique()

    for ann_type in unique_types:
        #if ann_type == 'Soma': continue

        type_df = df[df['annotated_type'] == ann_type]
        x_lines, y_lines, z_lines = [], [], []

        for _, row in type_df.iterrows():
            parent_id = row['p']
            if parent_id in coords:
                parent = coords[parent_id]
                x_lines.extend([row['x'], parent['x'], None])
                y_lines.extend([row['y'], parent['y'], None])
                z_lines.extend([row['z'], parent['z'], None])

        color = COLOR_MAP.get(ann_type, 'grey')

        fig.add_trace(go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode='lines',
            line=dict(color=color, width=3),
            name=ann_type,
            hoverinfo='name'
        ))

    # --- 3. DRAW ALL ROOTS (p == -1) ---
    roots = df[df['p'] == -1]

    # Separation: Main Virtual Root (0) vs Orphans
    virtual_root = roots[roots['id'] == 0]
    orphans = roots[roots['id'] != 0]

    # A. Draw Virtual Root (0) - Gold Sphere
    if not virtual_root.empty:
        rx, ry, rz = virtual_root.iloc[0]['x'], virtual_root.iloc[0]['y'], virtual_root.iloc[0]['z']
        radius = virtual_root.iloc[0]['r']

        # Center Marker
        fig.add_trace(go.Scatter3d(
            x=[rx], y=[ry], z=[rz],
            mode='markers',
            marker=dict(size=12, color='gold', symbol='diamond', line=dict(color='black', width=2)),
            name='Virtual Root (0)'
        ))

        # Wireframe Sphere (Volume)
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        x_sphere = rx + radius * np.outer(np.cos(u), np.sin(v))
        y_sphere = ry + radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = rz + radius * np.outer(np.ones(np.size(u)), np.cos(v))

        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            opacity=0.3,
            colorscale=[[0, 'gold'], [1, 'gold']],
            showscale=False,
            name='Root Volume'
        ))

    # B. Draw Orphans (Other -1s) - Red Crosses
    if not orphans.empty:
        print(f"   ⚠️ Found {len(orphans)} orphaned roots (disconnected segments).")
        fig.add_trace(go.Scatter3d(
            x=orphans['x'], y=orphans['y'], z=orphans['z'],
            mode='markers',
            marker=dict(size=6, color='red', symbol='x', line=dict(width=2)),
            name=f'Orphaned Roots ({len(orphans)})',
            hovertemplate="<b>Orphan Root</b><br>ID: %{text}<extra></extra>",
            text=orphans['id']
        ))

    # --- 4. HIGHLIGHT BRANCH EXITS ---
    # Nodes connected to Root 0
    exits = df[df['p'] == 0]
    if not exits.empty:
        fig.add_trace(go.Scatter3d(
            x=exits['x'], y=exits['y'], z=exits['z'],
            mode='markers',
            marker=dict(size=5, color='limegreen', symbol='circle'),
            name=f'Branch Exits ({len(exits)})'
        ))

    # --- LAYOUT ---
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (nm)', yaxis_title='Y (nm)', zaxis_title='Z (nm)',
            aspectmode='data',
            bgcolor='white',
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False),
        ),
        height=800,
        margin=dict(l=0, r=0, b=0, t=40)
    )

    fig.show()




def treat_orphan_roots(df_input_, soma_centroid, distance_threshold=15000, length_threshold=2000, plot_result=True):
    """
    Analyzes disconnected 'Orphan Roots' around the soma.

    SAFEGUARDS:
    1. Deep Copy: Original DF is untouched.
    2. Soma Integrity: Ensures Soma (ID 0) is always Root (p=-1).
    3. Re-Parenting: Orphans are attached specifically to ID 0.
    """
    df_input = df_input_.copy(deep=True)
    if df_input is None or df_input.empty:
        print("❌ DataFrame is empty.")
        return None

    # --- 0. SAFETY COPY ---
    df = df_input.copy()

    print(f"🔧 Treating Orphans (Dist < {distance_threshold} | Len > {length_threshold})...")

    # --- 1. SETUP GRAPH ---
    G = nx.DiGraph()
    # Add edges (Parent -> Child). Exclude roots (-1).
    edges = df[df['p'] != -1][['p', 'id']].values
    G.add_edges_from(edges)

    node_map = df.set_index('id').to_dict('index')

    # Identify Orphans: Roots (-1) that are NOT Soma (0)
    orphans = df[(df['p'] == -1) & (df['id'] != 0)]

    if orphans.empty:
        print("   ✅ No orphans found.")
        # Ensure Soma is valid before returning
        df.loc[df['id'] == 0, 'p'] = -1
        return df

    decision_log = []
    nodes_to_delete = set()
    all_orphan_nodes = set()

    # --- 2. PROCESS EACH ORPHAN ---
    for _, row in orphans.iterrows():
        orphan_id = row['id']
        root_coords = np.array([row['x'], row['y'], row['z']])

        # A. Trace Full Arbor
        try:
            arbor_nodes = list(nx.descendants(G, orphan_id))
        except nx.NetworkXError:
            arbor_nodes = []
        arbor_nodes.append(orphan_id)

        all_orphan_nodes.update(arbor_nodes)

        # B. Calculate Length
        total_length = 0.0
        for nid in arbor_nodes:
            if nid == orphan_id: continue
            node = node_map[nid]
            parent = node_map.get(node['p'])
            if parent:
                dist = np.linalg.norm(np.array([node['x'], node['y'], node['z']]) -
                                      np.array([parent['x'], parent['y'], parent['z']]))
                total_length += dist

        # C. Distance Check
        dist_to_soma = np.linalg.norm(root_coords - soma_centroid)

        # D. DECISION LOGIC
        if dist_to_soma > distance_threshold:
            action = "Ignored (Far)"
            color = "darkgrey"

        elif total_length > length_threshold:
            action = "Connected (Kept)"
            color = "limegreen"

            # --- CRITICAL: CONNECT TO SOMA (ID 0) ---
            df.loc[df['id'] == orphan_id, 'p'] = 0

        else:
            action = "Deleted (Noise)"
            color = "crimson"
            nodes_to_delete.update(arbor_nodes)

        decision_log.append({
            'id': orphan_id,
            'root_coords': root_coords,
            'arbor_nodes': arbor_nodes,
            'length': total_length,
            'dist': dist_to_soma,
            'action': action,
            'color': color
        })

    # --- 3. NESTED PLOTTING ---
    def plot_orphan_decisions(log, full_node_map, all_orphans_set):
        print("📊 Plotting Skeleton & Decisions...")
        fig = go.Figure()

        # A. Main Skeleton (Grey)
        all_ids = set(full_node_map.keys())
        main_ids = all_ids - all_orphans_set
        main_x, main_y, main_z = [], [], []

        for nid in main_ids:
            node = full_node_map[nid]
            pid = node['p']
            if pid != -1 and pid in full_node_map:
                parent = full_node_map[pid]
                main_x.extend([node['x'], parent['x'], None])
                main_y.extend([node['y'], parent['y'], None])
                main_z.extend([node['z'], parent['z'], None])

        fig.add_trace(go.Scatter3d(
            x=main_x, y=main_y, z=main_z,
            mode='lines', line=dict(color='lightgrey', width=1), opacity=0.4, name='Main Skeleton'
        ))

        # B. Orphans (Colored)
        for entry in log:
            ox, oy, oz = [], [], []
            for nid in entry['arbor_nodes']:
                node = full_node_map[nid]
                pid = node['p']
                if pid in full_node_map: # Works for parent 0 or internal parent
                    parent = full_node_map[pid]
                    ox.extend([node['x'], parent['x'], None])
                    oy.extend([node['y'], parent['y'], None])
                    oz.extend([node['z'], parent['z'], None])

            fig.add_trace(go.Scatter3d(
                x=ox, y=oy, z=oz,
                mode='lines', line=dict(color=entry['color'], width=4),
                name=f"{entry['action']}", text=f"ID: {entry['id']}", hoverinfo='text+name'
            ))

        # C. Soma Center
        fig.add_trace(go.Scatter3d(
            x=[soma_centroid[0]], y=[soma_centroid[1]], z=[soma_centroid[2]],
            mode='markers', marker=dict(size=6, color='gold', symbol='diamond'), name='Soma Center'
        ))

        fig.update_layout(title="Orphan Decisions", scene=dict(aspectmode='data', bgcolor='white'), height=800)
        fig.show()

    # --- 4. EXECUTE PLOT ---
    if plot_result:
        plot_orphan_decisions(decision_log, node_map, all_orphan_nodes)

    # --- 5. CLEANUP ---
    if nodes_to_delete:
        df = df[~df['id'].isin(nodes_to_delete)].copy()
        print(f"   🗑️ Deleted {len(nodes_to_delete)} nodes (Noise).")

    df.reset_index(drop=True, inplace=True)

    # --- CRITICAL: FORCE SOMA (0) TO BE ROOT (-1) ---
    # This guarantees the file topology is valid
    df.loc[df['id'] == 0, 'p'] = -1

    print(f"✅ Orphan treatment complete. Final node count: {len(df)}")
    return df




def manual_reroot(df, root_id, component_ids):
    """
    Re-orients a fragment so 'root_id' becomes the new root (p = -1).
    Used to ensure the distal fragment flows AWAY from the new connection.
    """
    subset = df[df['id'].isin(component_ids)].copy()

    # Build local graph
    G_sub = nx.Graph()
    G_sub.add_edges_from(subset[subset['p'] != -1][['id', 'p']].values)
    G_sub.add_nodes_from(subset['id'])

    # BFS to define new directionality
    bfs_tree = nx.bfs_tree(G_sub, source=root_id)

    # Map children to parents
    new_parents = {root_id: -1}
    for parent, child in bfs_tree.edges():
        new_parents[child] = parent

    # Apply changes
    df.loc[df['id'].isin(new_parents.keys()), 'p'] = df['id'].map(new_parents)
    return df

# --- MAIN FUNCTION ---
def highlight_and_stitch_points(df_, soma_center, threshold=1000, min_graph_hops=5, stitch=True):
    """
    1. Finds pairs of points that are spatially close but topologically distant.
    2. PLOTS them.
    3. WIRES them (Stitches) if they are in disconnected fragments.
       Logic: The furthest point (Distal) becomes the child of the closest (Proximal).
    """
    df = df_.copy(deep=True)
    if df is None or df.empty:
        print("❌ DataFrame is empty.")
        return df

    print(f"🔍 Scanning for close points (< {threshold} nm)...")

    # Make a copy to modify
    df_stitched = df.copy()

    # --- 1. BUILD GRAPH ---
    G = nx.Graph()
    G.add_nodes_from(df_stitched['id'])
    edges = df_stitched[df_stitched['p'] != -1][['id', 'p']].values
    G.add_edges_from(edges)

    # --- 2. SPATIAL QUERY ---
    coords = df_stitched[['x', 'y', 'z']].values
    ids = df_stitched['id'].values
    tree = cKDTree(coords)

    # Query pairs
    pairs_idx = tree.query_pairs(r=threshold, output_type='ndarray')

    if len(pairs_idx) == 0:
        print("   ✅ No close points found.")
        return df_stitched

    # --- 3. FILTERING & STITCHING ---
    close_calls = []
    merges_made = []
    idx_to_id = dict(enumerate(ids))
    node_map = df_stitched.set_index('id').to_dict('index')

    count_stitched = 0

    for i, j in pairs_idx:
        id_a = idx_to_id[i]
        id_b = idx_to_id[j]

        # A. Filter: Immediate Neighbors
        if G.has_edge(id_a, id_b): continue

        # B. Filter: Graph Distance (Curvature vs Gap)
        is_disconnected = False
        try:
            dist_hops = nx.shortest_path_length(G, source=id_a, target=id_b)
            if dist_hops < min_graph_hops: continue # Too close topologically (bend)
        except nx.NetworkXNoPath:
            is_disconnected = True # Prime candidate for stitching

        # Store for plotting
        pos_a = coords[i]
        pos_b = coords[j]
        dist_euclid = np.linalg.norm(pos_a - pos_b)

        close_calls.append({
            'pos_a': pos_a, 'pos_b': pos_b, 'dist': dist_euclid,
            'disconnected': is_disconnected
        })

        # --- C. WIRING SECTION ---
        if stitch and is_disconnected:
            # We only stitch if they are in DIFFERENT fragments (to avoid loops)

            # 1. Determine Hierarchy (Soma Distance)
            dist_a_soma = np.linalg.norm(pos_a - soma_center)
            dist_b_soma = np.linalg.norm(pos_b - soma_center)

            if dist_a_soma < dist_b_soma:
                proximal, distal = id_a, id_b
            else:
                proximal, distal = id_b, id_a

            # 2. Re-check path (in case a previous loop iteration already connected them)
            if nx.has_path(G, proximal, distal):
                continue

            try:
                # 3. Reroot the Distal Fragment
                # We need the distal node to be the 'Root' of its chunk so we can plug it in
                comp_ids = list(nx.node_connected_component(G, distal))
                df_stitched = manual_reroot(df_stitched, distal, comp_ids)

                # 4. Connect
                df_stitched.loc[df_stitched['id'] == distal, 'p'] = proximal

                # 5. Update Graph (so subsequent checks know they are connected)
                G.add_edge(proximal, distal)

                merges_made.append((pos_a, pos_b))
                count_stitched += 1
                print(f"   🔗 Stitched {distal} -> {proximal} (Gap: {dist_euclid:.1f} nm)")

            except Exception as e:
                print(f"   ⚠️ Stitch failed: {e}")

    print(f"🚀 Found {len(close_calls)} proximities. Performed {count_stitched} stitches.")

    # --- 4. VISUALIZATION ---
    fig = go.Figure()

    # Plot Skeleton
    x_skel, y_skel, z_skel = [], [], []
    # Re-map from the NEW df
    curr_map = df_stitched.set_index('id')[['x', 'y', 'z']].to_dict('index')
    for _, row in df_stitched.iterrows():
        pid = row['p']
        if pid != -1 and pid in curr_map:
            p = curr_map[pid]
            x_skel.extend([row['x'], p['x'], None])
            y_skel.extend([row['y'], p['y'], None])
            z_skel.extend([row['z'], p['z'], None])

    fig.add_trace(go.Scatter3d(
        x=x_skel, y=y_skel, z=z_skel, mode='lines',
        line=dict(color='lightgrey', width=3), opacity=0.7, name='Skeleton'
    ))

    # Plot "Close Calls" (Red = Ignored/Loops, Cyan = Stitched)
    # 1. Plot Stitched (Cyan)
    if merges_made:
        mx, my, mz = [], [], []
        for (p1, p2) in merges_made:
            mx.extend([p1[0], p2[0], None])
            my.extend([p1[1], p2[1], None])
            mz.extend([p1[2], p2[2], None])
        fig.add_trace(go.Scatter3d(
            x=mx, y=my, z=mz, mode='lines',
            line=dict(color='cyan', width=6), name='Stitched Gaps'
        ))

    # 2. Plot Unstitched Proximities (Red) - likely loops or sharp bends
    unstitched = [c for c in close_calls if not c['disconnected']]
    if unstitched:
        ux, uy, uz = [], [], []
        for item in unstitched:
            p1, p2 = item['pos_a'], item['pos_b']
            ux.extend([p1[0], p2[0], None])
            uy.extend([p1[1], p2[1], None])
            uz.extend([p1[2], p2[2], None])
        fig.add_trace(go.Scatter3d(
            x=ux, y=uy, z=uz, mode='lines',
            line=dict(color='red', width=2, dash='dot'), name='Ignored (Loops/Bends)'
        ))

    # Plot Soma Center
    fig.add_trace(go.Scatter3d(
        x=[soma_center[0]], y=[soma_center[1]], z=[soma_center[2]],
        mode='markers', marker=dict(size=8, color='black', symbol='x'), name='Soma Center'
    ))

    fig.update_layout(title=f"Stitching Result (Threshold: {threshold}nm)", scene=dict(aspectmode='data'), height=800)
    fig.show()

    return df_stitched




def plot_soma_and_roots(df, title="Soma & Disconnected Roots Check"):
    """
    Visualizes the skeleton, highlighting:
    1. The SOMA (ID 0) as a large Green Diamond.
    2. ANY OTHER ROOTS (p=-1) as Red 'X' markers (broken segments).
    3. The rest of the skeleton as grey lines.
    """
    if df is None or df.empty:
        print("❌ DataFrame is empty.")
        return

    print(f"📊 Plotting Roots Check ({len(df)} nodes)...")

    # --- 1. SEPARATE DATA ---
    # Find all roots (parent is -1)
    roots = df[df['p'] == -1]

    # Identify the True Soma (ID 0) vs. Broken Roots
    soma_root = roots[roots['id'] == 0]
    broken_roots = roots[roots['id'] != 0]

    # Create coordinate map for drawing lines
    node_map = df.set_index('id')[['x', 'y', 'z']].to_dict('index')

    fig = go.Figure()

    # --- 2. DRAW SKELETON LINES (Context) ---
    # Draw all segments that represent valid connections (p != -1)
    # We use a simple grey line for context
    lines_x, lines_y, lines_z = [], [], []

    # Filter for valid parents
    valid_segments = df[df['p'] != -1]

    for _, row in valid_segments.iterrows():
        pid = row['p']
        if pid in node_map:
            parent = node_map[pid]
            lines_x.extend([row['x'], parent['x'], None])
            lines_y.extend([row['y'], parent['y'], None])
            lines_z.extend([row['z'], parent['z'], None])

    fig.add_trace(go.Scatter3d(
        x=lines_x, y=lines_y, z=lines_z,
        mode='lines',
        line=dict(color='lightgrey', width=1),
        opacity=0.5,
        name='Skeleton',
        hoverinfo='skip'
    ))

    # --- 3. PLOT TRUE SOMA (ID 0) ---
    if not soma_root.empty:
        fig.add_trace(go.Scatter3d(
            x=soma_root['x'], y=soma_root['y'], z=soma_root['z'],
            mode='markers',
            marker=dict(size=12, color='limegreen', symbol='diamond', line=dict(color='black', width=2)),
            name='Soma (Root 0)',
            hovertemplate="<b>Soma Root</b><br>ID: 0<extra></extra>"
        ))
    else:
        print("⚠️ Warning: No Soma (ID 0) found with p=-1.")

    # --- 4. PLOT BROKEN ROOTS (Orphans) ---
    if not broken_roots.empty:
        print(f"   ⚠️ Found {len(broken_roots)} disconnected roots/orphans.")
        fig.add_trace(go.Scatter3d(
            x=broken_roots['x'], y=broken_roots['y'], z=broken_roots['z'],
            mode='markers',
            marker=dict(size=6, color='crimson', symbol='x', line=dict(width=2)),
            name=f'Disconnected Roots ({len(broken_roots)})',
            hovertemplate="<b>Disconnected</b><br>ID: %{text}<extra></extra>",
            text=broken_roots['id']
        ))
    else:
        print("   ✅ Perfect Topology: No disconnected roots found.")

    # --- 5. LAYOUT ---
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
            aspectmode='data',
            bgcolor='white'
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        height=700
    )

    fig.show()

def reroot_entire_neuron_navis(df_):
    """
    1. Identifies the SINGLE root node (p = -1).
    2. Validates that ONLY ONE root exists.
    3. Uses Navis to re-orient the skeleton away from this root.

    Ensures the hierarchy flows: Soma (Root) -> Branches -> Tips
    """
    df = df_.copy(deep=True)
    if df is None or df.empty:
        print("❌ DataFrame is empty.")
        return df

    # --- 1. FIND & VALIDATE ROOT ---
    # Strict check: There must be exactly one node with parent = -1
    root_rows = df[df['p'] == -1]

    if len(root_rows) == 0:
        print("❌ Error: No root found (No node has p = -1). The skeleton might be a closed loop or empty.")
        return None

    if len(root_rows) > 1:
        print(f"❌ Error: Multiple roots found ({len(root_rows)}).")
        print(f"   Root IDs: {root_rows['id'].tolist()}")
        print("   The skeleton is fragmented (multiple disconnected trees). Please stitch them first.")
        return None

    # Extract the single valid Root ID
    root_id = int(root_rows.iloc[0]['id'])
    print(f"🌲 Identified Single Root ID: {root_id}")
    print(f"   Re-orienting entire neuron flow away from ID {root_id}...")

    # --- 2. PREPARE DATA FOR NAVIS ---
    temp_df = df.copy()
    # Map to Navis standard
    rename_map = {'id': 'node_id', 'p': 'parent_id', 'r': 'radius', 'x': 'x', 'y': 'y', 'z': 'z'}
    temp_df.rename(columns=rename_map, inplace=True)

    # --- 3. CREATE NEURON OBJECT ---
    try:
        # Navis will automatically preserve extra columns like 'annotated_type' in the .nodes attribute
        n = navis.TreeNeuron(temp_df, name='neuron_clean')
    except Exception as e:
        print(f"❌ Navis Import Failed: {e}")
        return df

    # --- 4. EXECUTE REROOT ---
    # This flips any edges that are pointing the "wrong way" (towards the soma)
    try:
        n.reroot(root_id, inplace=True)
        print(f"✅ Directionality enforced successfully.")
    except Exception as e:
        print(f"❌ Reroot Failed: {e}")
        return df

    # --- 5. RESTORE DATAFRAME FORMAT ---
    new_df = n.nodes.copy()

    # Map back to your standard
    reverse_map = {'node_id': 'id', 'parent_id': 'p', 'radius': 'r'}
    new_df.rename(columns=reverse_map, inplace=True)

    # Ensure Integer Types
    new_df['id'] = new_df['id'].astype(int)
    new_df['p'] = new_df['p'].fillna(-1).astype(int)

    # --- 6. PRESERVE COLUMN ORDER ---
    # Keep the output looking exactly like the input
    original_cols = df.columns.tolist()

    # Identify columns that were preserved vs new ones
    final_cols = [c for c in original_cols if c in new_df.columns] + \
                 [c for c in new_df.columns if c not in original_cols]

    return new_df[final_cols]




def plot_final_neuron(df, title="Final Reconstructed Neuron"):
    """
    Plots the complete neuron skeleton with segments colored by 'annotated_type'.

    Features:
    - Fast rendering (uses None-separated line batches).
    - distinct colors for Soma, Axon, Dendrite, AIS, etc.
    - Highlights the Root Node (Soma Center).
    """
    if df is None or df.empty:
        print("❌ DataFrame is empty.")
        return

    print("🎨 Generating Final Skeleton Plot...")

    # --- 1. SETUP COLOR MAP ---
    # Standard SWC colors + Custom Types
    COLOR_MAP = {
        'Soma': 'green',       '1': 'green',
        'Axon': 'red',         '2': 'red',
        'Dendrite': 'blue',    '3': 'blue',
        'Apical': 'magenta',   '4': 'magenta',
        'AIS': 'gold',         'myelin': 'orange',
        'Unknown': 'grey'
    }

    # Normalize type column to string to match keys safely
    df['type_str'] = df['annotated_type'].astype(str)

    # Create Fast Lookup for Coordinates
    # We need to look up Parent Coordinates instantly
    node_map = df.set_index('id')[['x', 'y', 'z']].to_dict('index')

    fig = go.Figure()

    # --- 2. PLOT SEGMENTS BY TYPE ---
    # We loop through each unique type found in the data (e.g., 'Axon', 'Dendrite')
    for type_name in df['type_str'].unique():

        # Filter nodes of this type
        # We only care about nodes that have a valid parent (p != -1)
        subset = df[(df['type_str'] == type_name) & (df['p'] != -1)]

        if subset.empty: continue

        x_lines, y_lines, z_lines = [], [], []

        # Build the line segments (Child -> Parent)
        for _, row in subset.iterrows():
            pid = row['p']

            # Check if parent exists in the map (it should, unless data is broken)
            if pid in node_map:
                parent = node_map[pid]

                # Add Line Segment: [Child_X, Parent_X, None]
                x_lines.extend([row['x'], parent['x'], None])
                y_lines.extend([row['y'], parent['y'], None])
                z_lines.extend([row['z'], parent['z'], None])

        # Pick Color
        color = COLOR_MAP.get(type_name, 'grey') # Default to grey if unknown

        # Add Trace
        fig.add_trace(go.Scatter3d(
            x=x_lines, y=y_lines, z=z_lines,
            mode='lines',
            line=dict(color=color, width=3),
            name=f'{type_name} ({len(subset)} segs)',
            hoverinfo='name'
        ))

    # --- 3. HIGHLIGHT THE ROOT (SOMA) ---
    # The root is the node with p = -1 (usually ID 0 or 1 after collapsing)
    root_node = df[df['p'] == -1]

    if not root_node.empty:
        fig.add_trace(go.Scatter3d(
            x=root_node['x'], y=root_node['y'], z=root_node['z'],
            mode='markers',
            marker=dict(size=12, color='black', symbol='diamond', line=dict(color='white', width=2)),
            name='Root / Soma Center'
        ))

    # --- 4. LAYOUT SETTINGS ---
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (nm)',
            yaxis_title='Y (nm)',
            zaxis_title='Z (nm)',
            aspectmode='data', # Crucial for 3D realism
            bgcolor='white'
        ),
        legend=dict(x=0, y=1),
        margin=dict(l=0, r=0, b=0, t=40),
        height=800
    )

    fig.show()



def clean_branch_labels(df):
    """
    Standardizes 'annotated_type' across entire branches.
    Assumes the input dataframe is already correctly rooted (Parent -> Child flow).

    Logic:
    1. Finds the Soma (Root, p == -1).
    2. Identifies Primary Roots (nodes directly attached to the Soma).
    3. Traces all downstream descendants for each Primary Root.
    4. Calculates the most frequent label in that branch.
    5. Overwrites all nodes in that branch with the predominant label.
    """
    if df is None or df.empty:
        print("❌ DataFrame is empty.")
        return df

    if 'annotated_type' not in df.columns:
        print("⚠️ 'annotated_type' column missing. Cannot clean labels.")
        return df

    print("🧹 Cleaning and unifying branch labels...")
    df_clean = df.copy()

    # Ensure string type for labels to avoid mixed-type errors
    df_clean['annotated_type'] = df_clean['annotated_type'].astype(str)

    # --- 1. BUILD DIRECTED GRAPH ---
    # Edges MUST go Parent -> Child so we can safely trace downstream
    G = nx.DiGraph()
    G.add_nodes_from(df_clean['id'])

    # Filter out the root's parent (-1) before adding edges
    valid_edges = df_clean[df_clean['p'] != -1][['p', 'id']].values
    G.add_edges_from(valid_edges)

    # --- 2. FIND ROOT AND PRIMARY ROOTS ---
    # Since we already ran the Navis reroot, we know exactly where the root is
    root_nodes = df_clean[df_clean['p'] == -1]['id'].values
    if len(root_nodes) == 0:
        print("❌ Error: No Root node found (p = -1).")
        return df_clean

    soma_id = root_nodes[0]

    # Primary roots are the direct children of the Soma
    primary_roots = df_clean[df_clean['p'] == soma_id]['id'].tolist()
    print(f"   🌱 Found {len(primary_roots)} primary branches attached to Soma (ID {soma_id}).")

    # --- 3. EVALUATE AND APPLY PREDOMINANT LABELS ---
    nodes_changed = 0

    for pr_id in primary_roots:
        # nx.descendants gets ALL nodes flowing downstream from the primary root to the tips
        branch_nodes = list(nx.descendants(G, pr_id)) + [pr_id]

        # Extract the labels for this specific branch
        branch_df = df_clean[df_clean['id'].isin(branch_nodes)]
        labels = branch_df['annotated_type']

        # Find the absolute majority label
        predominant_label = labels.mode()[0]

        # Count how many are changing for the console log
        num_changing = (branch_df['annotated_type'] != predominant_label).sum()
        nodes_changed += num_changing

        # Apply the winning label to the entire branch
        df_clean.loc[df_clean['id'].isin(branch_nodes), 'annotated_type'] = predominant_label

        print(f"      🔹 Branch at {pr_id}: {len(branch_nodes)} nodes -> Consolidated to '{predominant_label}' ({num_changing} nodes fixed)")

    print(f"✅ Label cleaning complete. Unified {nodes_changed} anomalous nodes across all branches.")
    return df_clean


def plot_skeleton_comparison(df_old, df_new, title="Skeleton Comparison: Before vs. After Cleaning"):
    """
    Plots two skeleton dataframes side-by-side in 3D to compare label smoothing.
    Both subplots are color-coded by 'annotated_type'.
    """
    if df_old is None or df_new is None:
        print("❌ Missing one or both DataFrames.")
        return

    print("📊 Generating side-by-side comparison plot...")

    # --- 1. SETUP SUBPLOTS ---
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=("Original Labels", "Cleaned Labels (Majority Rule)")
    )

    # --- 2. COLOR MAP ---
    COLOR_MAP = {
        'Soma': 'green',       '1': 'green',
        'Axon': 'red',         '2': 'red',
        'Dendrite': 'blue',    '3': 'blue',
        'Apical': 'magenta',   '4': 'magenta',
        'AIS': 'gold',         'myelin': 'orange',
        'Unknown': 'grey'
    }

    # --- 3. HELPER FUNCTION TO BUILD TRACES ---
    def add_traces_to_subplot(df, col_idx):
        # Normalize type string
        if 'annotated_type' in df.columns:
            df['type_str'] = df['annotated_type'].astype(str)
        else:
            df['type_str'] = 'Unknown'

        # Fast coordinate lookup map
        node_map = df.set_index('id')[['x', 'y', 'z']].to_dict('index')

        # Keep track of added types so we only show them in the legend once
        types_added = set()

        # Iterate over each unique label
        for type_name in df['type_str'].unique():
            subset = df[(df['type_str'] == type_name) & (df['p'] != -1)]
            if subset.empty: continue

            x_lines, y_lines, z_lines = [], [], []

            # Build line segments (Child -> Parent) separated by None
            for _, row in subset.iterrows():
                pid = row['p']
                if pid in node_map:
                    parent = node_map[pid]
                    x_lines.extend([row['x'], parent['x'], None])
                    y_lines.extend([row['y'], parent['y'], None])
                    z_lines.extend([row['z'], parent['z'], None])

            color = COLOR_MAP.get(type_name, 'grey')

            # Show legend only for the first subplot to keep it clean
            show_leg = (col_idx == 1) and (type_name not in types_added)
            types_added.add(type_name)

            # Add to the specific subplot
            fig.add_trace(go.Scatter3d(
                x=x_lines, y=y_lines, z=z_lines,
                mode='lines',
                line=dict(color=color, width=3),
                name=type_name,
                showlegend=show_leg,
                hoverinfo='name'
            ), row=1, col=col_idx)

        # Add the Root (Soma Center) as a distinct black diamond
        root_node = df[df['p'] == -1]
        if not root_node.empty:
            fig.add_trace(go.Scatter3d(
                x=root_node['x'], y=root_node['y'], z=root_node['z'],
                mode='markers',
                marker=dict(size=8, color='black', symbol='diamond', line=dict(color='white', width=1)),
                name='Virtual Root',
                showlegend=False,
                hoverinfo='skip'
            ), row=1, col=col_idx)

    # --- 4. POPULATE SUBPLOTS ---
    add_traces_to_subplot(df_old, col_idx=1) # Left Panel
    add_traces_to_subplot(df_new, col_idx=2) # Right Panel

    # --- 5. SYNCHRONIZE LAYOUT ---
    # aspectmode='data' ensures 1 nm in X looks identical to 1 nm in Z
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X (nm)', yaxis_title='Y (nm)', zaxis_title='Z (nm)',
            aspectmode='data', bgcolor='white'
        ),
        scene2=dict(
            xaxis_title='X (nm)', yaxis_title='Y (nm)', zaxis_title='Z (nm)',
            aspectmode='data', bgcolor='white'
        ),
        legend=dict(title="Branch Types", x=1.05, y=0.5), # Move legend to the right
        margin=dict(l=0, r=0, b=0, t=50),
        height=700,
        width=1200
    )

    # Optional: Link the camera views so rotating one rotates the other
    # fig.update_layout(scene_camera=fig.layout.scene.camera, scene2_camera=fig.layout.scene.camera)

    fig.show()




















