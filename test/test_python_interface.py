
# TODO put try, catch around imports ...
import numpy as np
import pyuvwsim as uvw

import matplotlib
import matplotlib.pyplot as plt
import time

filename='@PROJECT_BINARY_DIR@/test/VLA_A_hor_xyz.txt'
lon  = 0.0  * (np.pi/180.)
lat  = 90.0 * (np.pi/180.)
alt  = 0.0
ra0  = 0.0  * (np.pi/180.)
dec0 = 90.0 * (np.pi/180.)
mjd_start  = uvw.datetime_to_mjd(2014, 7, 28, 14, 26, 1.321);
ntimes     = 10
obs_length = 0.2 # days

(x,y,z) = uvw.load_station_coords(filename)
(x,y,z) = uvw.convert_enu_to_ecef(x,y,z,lon,lat,alt)

nant = len(x)
print 'nstations =',nant

t0 = time.time()
for i in range(0, ntimes):
    mjd = mjd_start
    (uu_m,vv_m,ww_m) = uvw.evaluate_baseline_uvw(x,y,z, ra0, dec0, mjd)
print 'Time taken to evaluate uww points = %.3f ms' % ((time.time()-t0)*1.e3)

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
plt.title(filename)
plt.grid(True)
plt.tick_params(axis='both', which='major', labelsize=6)
plt.savefig(plt_filename)
plt.close()
