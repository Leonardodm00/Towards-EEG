def save_nids(nids, output_dir, prefix, extension=".txt"):
    """
    Saves a list of integer IDs to a specified directory.
    
    Args:
        nids (list or array): The list of integer IDs to save.
        output_dir (str): The folder path where the file should be saved.
        prefix (str): The tunable name (the 'X' in 'X_nids').
        extension (str): The file extension (defaults to '.txt', but could be '.gdf').
    """
    # 1. Ensure the target directory exists to prevent FileNotFoundError
    os.makedirs(output_dir, exist_ok=True)
    
    # 2. Construct the exact filename (e.g., 'subpopA_nids.txt')
    filename = f"{prefix}_nids{extension}"
    
    # 3. Create the safe, full path
    full_path = os.path.join(output_dir, filename)
    
    # 4. Save the list. 
    # fmt='%d' ensures they remain integers, not floats.
    np.savetxt(full_path, nids, fmt='%d')
