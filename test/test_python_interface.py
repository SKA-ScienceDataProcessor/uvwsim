import pyuvwsim as uvw
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import time

# ---------------------------------------------------------
filename   = '@PROJECT_BINARY_DIR@/test/VLA_A_hor_xyz.txt'
lon        = 0.0  * (np.pi/180.)
lat        = 90.0 * (np.pi/180.)
alt        = 0.0
ra0        = 0.0  * (np.pi/180.)
dec0       = 90.0 * (np.pi/180.)
ntimes     = 10
obs_length = 0.2 # days
# ---------------------------------------------------------

# Generate coordinates.
mjd_start  = uvw.datetime_to_mjd(2014, 7, 28, 14, 26, 1.321);
(x,y,z) = uvw.load_station_coords(filename)
(x,y,z) = uvw.convert_enu_to_ecef(x, y, z, lon, lat, alt)
t0 = time.time()
for i in range(0, ntimes):
    mjd = mjd_start
    (uu_m,vv_m,ww_m) = uvw.evaluate_baseline_uvw(x,y,z, ra0, dec0, mjd)
print '- Time taken to evaluate uww coordinates = %.3f ms' % ((time.time()-t0)*1.e3)

# Scatter plot of baseline coordinates.
plt.ioff()
plt_filename = 'test.png'
fig = plt.figure(figsize=(6,6))
matplotlib.rcParams.update({'font.size': 10})
scatter_size = 3
fig.add_subplot(1,1,1, aspect='equal')
plt.scatter(uu_m, vv_m, c='b', lw = 0, s=scatter_size);
plt.scatter(-uu_m, -vv_m, c='r', lw = 0, s=scatter_size);
plt.xlabel('uu [metres]', fontsize=8)
plt.ylabel('vv [metres]', fontsize=8)
plt.title(filename[-17:])
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=6)
if os.path.isfile(plt_filename): os.remove(plt_filename)
plt.savefig(plt_filename)
plt.close()

print '- uvw coordinates plotted to file: %s' % os.path.abspath(plt_filename)
