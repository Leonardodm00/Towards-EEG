try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False

def save_probability_fields(axon_field, dend_field, output_dir, base_name="population_probability", save_tiff=True):
    """
    Saves 3D probability fields to a specified directory as both .npy and .tif formats.
    """
    if axon_field is None or dend_field is None:
        print("⚠️ Warning: Empty fields provided. Nothing to save.")
        return

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # --- 1. Save as native NumPy arrays ---
    axon_npy_path = os.path.join(output_dir, f"{base_name}_axon.npy")
    dend_npy_path = os.path.join(output_dir, f"{base_name}_dend.npy")
    
    np.save(axon_npy_path, axon_field)
    np.save(dend_npy_path, dend_field)
    print(f"✅ Saved NumPy data:\n  -> {axon_npy_path}\n  -> {dend_npy_path}")
    
    # --- 2. Save as Multi-page TIFF stacks ---
    if save_tiff:
        if HAS_TIFFFILE:
            axon_tif_path = os.path.join(output_dir, f"{base_name}_axon.tif")
            dend_tif_path = os.path.join(output_dir, f"{base_name}_dend.tif")
            
            # Spatial formatting for imaging software
            # histogramdd outputs (X, Y, Z). 
            # ImageJ/napari expect the first dimension to be the Z-stack (Z, Y, X).
            axon_img = axon_field.transpose(2, 1, 0).astype(np.float32)
            dend_img = dend_field.transpose(2, 1, 0).astype(np.float32)
            
            tifffile.imwrite(axon_tif_path, axon_img, imagej=True)
            tifffile.imwrite(dend_tif_path, dend_img, imagej=True)
            print(f"✅ Saved TIFF stacks:\n  -> {axon_tif_path}\n  -> {dend_tif_path}")
        else:
            print("⚠️ 'tifffile' library not found. Skipping TIFF export.")
            print("💡 To enable TIFF export, run: pip install tifffile")
