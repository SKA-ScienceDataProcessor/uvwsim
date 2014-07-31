#include <uvwsim.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#defined M_PI 3.14159265358979323846264338327950288
#endif

int main() 
{
    // Specify layout file and check that it exists.
    const char* filename = "VLA_A_hor_xyz.txt";
    if (!uvwsim_file_exists(filename)) {
        fprintf(stderr, "Unable to find specified layout file: %s\n", filename);
        exit(1);
    } 
    
    // Read the number of stations in the layout file.
    int nant = uvwsim_get_num_stations(filename);
    
    // Allocate memory for antenna coordinates.
    double* x = (double*)malloc(nant * sizeof(double));
    double* y = (double*)malloc(nant * sizeof(double));
    double* z = (double*)malloc(nant * sizeof(double));
    
    // Load the antenna coordinates, checking that the expected number have been read.
    if (uvwsim_load_station_coords(filename, nant, x, y, z) != nant) {
        fprintf(stderr, "Failed to read antenna coordinates!\n");
        exit(1);
    }
    
    // Convert coordinates to ECEF (using the coordinates of the VLA)
    double lon = -107.6184 * (M_PI/180.);
    double lat =  34.0790 * (M_PI/180.);
    double alt = 0.0;
    uvwsim_convert_enu_to_ecef(nant, x, y, z, x, y, z, lon, lat, alt);
    
    // Define observation parameters. Observation of Ra,Dec=20,60 deg. 
    // for 15 times starting at 20/03/2014 14:57:1.3 for 1 hour.
    double ra0  = 0.0 * (M_PI/180.);
    double dec0 = 90.0 * (M_PI/180.);
    int ntimes = 15;
    double start_time_mjd = uvwsim_datetime_to_mjd(2014, 03, 20, 01, 57, 1.3);
    double obs_length_days = 1.0 / 24.0;
    int nbaselines = uvwsim_num_baselines(nant);
    
    // Allocate memory for baseline coordinates.
    int ncoords = ntimes * nbaselines;
    double* uu = (double*)malloc(ncoords * sizeof(double));
    double* vv = (double*)malloc(ncoords * sizeof(double));
    double* ww = (double*)malloc(ncoords * sizeof(double));
    // Evaluate baseline uvw coordinates.
    for (int t = 0; t < ntimes; ++t) {
        double time_mjd = start_time_mjd + t * (obs_length_days/(double)ntimes);
        size_t offset = t * nbaselines;
        uvwsim_evaluate_baseline_uvw(&uu[offset], &vv[offset], &ww[offset],
            nant, x, y, z, ra0, dec0, time_mjd);
    }

    // Convert baseline coordinates from metres to wavelengths.
    double c0 = 299792458.0;
    int nchan = 5;
    double start_freq = 500.0e6; // Hz
    double freq_inc   = 5.0e6; // Hz
    double* uu_wavelengths = (double*)malloc(ncoords * nchan * sizeof(double));
    double* vv_wavelengths = (double*)malloc(ncoords * nchan * sizeof(double));
    double* ww_wavelengths = (double*)malloc(ncoords * nchan * sizeof(double));
    for (int c = 0, j = 0; c < nchan; ++c) {
        double freq = start_freq + (double)c * freq_inc;
        double freq_scale = freq / c0;
        for (int i = 0; i < ncoords; ++i, ++j) {
            uu_wavelengths[j] = uu[i] * freq_scale;
            vv_wavelengths[j] = vv[i] * freq_scale;
            ww_wavelengths[j] = ww[i] * freq_scale;
        }
    }

    // Write the coordinates to file
    FILE* fp = fopen("example_baselines_vla_a.txt", "w");
    if (!fp) {
        fprintf(stderr, "unable to open output file for writing.\n");
        exit(1);
    }
    for (int i = 0; i < nchan * ncoords; ++i) {
        fprintf(fp, "%e,%e,%e\n", uu_wavelengths[i], vv_wavelengths[i],
                ww_wavelengths[i]);
        // Also write the coordinates of the mirror baseline (ie. the coordinates of
        // baseline for antennas 1 & 2 has coordinates for 1-2 and 2-1.
        // This is really only useful for making symetric plots!
        fprintf(fp, "%e,%e,%e\n", -uu_wavelengths[i], -vv_wavelengths[i],
                -ww_wavelengths[i]);
    }
    fclose(fp);

    // Clean up memory.
    free(x); free(y); free(z);
    free(uu); free(vv); free(ww);
    free(uu_wavelengths); free(vv_wavelengths); free(ww_wavelengths);
}
