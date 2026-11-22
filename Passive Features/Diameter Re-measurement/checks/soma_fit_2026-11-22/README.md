# Can the brightfield stack give an ellipsoid soma's height? (2026-11-22; D-216.2)

The check before Block 45-46's fit of a soma model (D-215, D-216): the Fisher information of the stack for an
ellipsoid soma rendered with the pipeline's partition absorption and the pilot's calibrated kernel table
(`cleaning_config.json` of the user's Drive: sigma 0.08, 0.154, 0.392, 0.562 um at 0, 0.28, 0.56, 0.84 um; the
configured continuation, linear 0.79 um/um, not calibrated on this cell). In-plane semi-axes 6.6 and 5.8 um at 20 deg;
heights c = 3.5, 5, 6.5, 8 um; absorption set so that the centre's transmittance would be 0.5; background 110 gl, noise
3 gl; planes every 0.56 um within +-10 um; a 13 um disk of pixels; per-plane backgrounds as nuisance parameters.
Simplifications: the pixel grid itself, the kernel floored at half a pixel, slabs of 0.14 um.

`soma_identifiability.py` **[run, sandbox]** (`soma_identifiability.out`, `.json`, about 10 min):
- SE(c) about 0.005 um at every height; corr(c, mu) -0.96 (c 3.5) to -0.60 (c 8); corr(c, z0) about +0.36.
- Data rendered with the far slope x 1.2, fitted with the configured one: bias(c) -0.98, -0.76, -0.63, -0.50 um;
  bias(S) -35, -29, -24, -20 um2 (S 350, 422, 499, 579 um2): -10 to -3 %.
- An apical stem (radius 1.3 um) leaking outside its mask: bias(c) about +0.1 um, bias(S) +8 to +12 um2 (2 %).

`soma_identifiability_slope.py` **[run, sandbox]** (`soma_identifiability_slope.out`, about 4 min): with the far
slope gamma fitted jointly, SE(c) 0.006-0.007 um, SE(gamma) 0.0005-0.0011 um/um, corr(c, gamma) about +0.6: in this
model the soma's own stack separates its height from the kernel's far growth.

Not in it (precision, not accuracy): a soma that is not an ellipsoid (the pilot's is pyramidal), a far-defocus point
spread that is not Gaussian, neuropil and background structure, JPEG.
