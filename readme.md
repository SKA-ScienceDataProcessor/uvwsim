# The uvwsim library (version 1.0)#


## Introduction ##
The **uvwsim** library provides a simple C API for generating interferometric baseline (uvw) coordinates. This library has been developed for use with various SKA.SDP tasks.

The code is based on the specification described in [JIRA ticket PRENG-44](https://jira.ska-sdp.org/browse/PRENG-44).

## Building and installing ##

### Dependencies ###
- CMake 1.8.0 or higher. ([www.cmake.org](http://www.cmake.org))

### Build command ###
In order to build the library run the following commands:

```
mkdir build
cd build
cmake [build options] ..
make
```

In order to then install the library, if required, run the command:

```
make install
```

### Build options ###
A number of build options can be provided when constructing the makefiles with the `cmake` command, which alter the behaviour of the generated Makefiles.

- **-DCMAKE_BUILD_TYPE=[release or debug]** : Build the library with release or debug compiler options (default=release).
- **-DBUILD_SHARED_LIBS=[ON or OFF]** : Specify building a shared or static library (default=static).
- **-DCMAKE_INSTALL_PREFIX=[path]** : Set the installation prefix (default=/usr/local).
- **-DCMAKE_C_COMPILER=[path to C compiler]** : Specifies the C compiler.
- **-DCMAKE_CXX_COMPILER=[path to C++ compiler]** : Specifies the C++ compiler (used for unit tests only).

### Testing the library ###
A number of unit tests are built along with the library and can be run by issuing the following command from the top level build directory:

```
ctest --verbose
```

(Please note that the unit tests contain hard-coded relative paths to data files so must be run from the top level build directory.)

All tests are expected to pass, if you find any failures please contact us at uvwsim@oerc.ox.ac.uk with a copy of the failed output, details of your operating system and any build variables you may have specified.


## Using the library ##

### The API ###

The **uvwsim** library currently consists of 7 public functions all prefixed with `uvwsim_`. These are listed below:

--------------------------------------------------------------------------------

```C
int uvwsim_file_exists(const char* filename);
```
Returns true if the specified file exists.

- **[in] filename** : File name (path) to check.
- **[rtn]**         : 1 (true) if the file exists, 0 (false) otherwise.

--------------------------------------------------------------------------------
```C
int uvwsim_get_num_stations(const char* filename)
```
Returns the number of antennas or stations coordinates in a specified layout file.

- **[in] filename** : File name (path) of a layout file.
- **[rtn]**         : Number of coordinates in the specified file.

--------------------------------------------------------------------------------

```C
int uvwsim_load_station_coords(const char* filename, int n, double* x,
    double* y, double* z)
```
Loads antenna or station coordinates from the specified file and returns the number of coordinates read.

- **[in] filename** : File name (path) of a layout file.
- **[in] n**        : Number of stations or antennas to read from the file.
- **[out] x**       : Array of station x coordinates.
- **[out] y**       : Array of station y coordinates.
- **[out] z**       : Array of station z coordinates.
- **[rtn]**         : Number of coordinates read from the specified file.

--------------------------------------------------------------------------------

```C
void uvwsim_convert_enu_to_ecef(int n, double* x_ecef,
    double* y_ecef, double* z_ecef, const double* x_enu,
    const double* y_enu, const double* z_enu, double lon, double lat,
    double alt);
```
Converts from East-North-Up (ENU), local tangent plane coordinates to Earth centred, Earth fixed (ECEF) coordiantes. Note: by specifying the same memory for input and output coordinate arrays the transform can be performed in-place.

- **[in] n**       : Number of stations or antennas to read from the file.
- **[out] x_ecef** : Array of ECEF station x coordinates, in metres.
- **[out] y_ecef** : Array of ECEF station y coordinates, in metres.
- **[out] z_ecef** : Array of ECEF station z coordinates, in metres.
- **[in] x_enu**   : Array of ENU station x coordinates, in metres.
- **[in] y_enu**   : Array of ENU station y coordinates, in metres.
- **[in] z_enu**   : Array of ENU station z coordinates, in metres.
- **[in] lon**     : East Longitude of the reference point, in radians.
- **[in] lat**     : North Latitude of the reference point, in radians.
- **[in] alt**     : Altitude of the reference point, in metres.

--------------------------------------------------------------------------------

```C
int uvwsim_num_baselines(int n);
```
Returns the number of baslines for the specified number of antennas.

- **[in] n**  : Number of stations or antennas.
- **[rtn]** : Number of baselines.

--------------------------------------------------------------------------------

```C
void uvwsim_evaluate_baseline_uvw(double* uu, double* vv, double* ww,
    int n, const double* x_ecef, const double* y_ecef, const double* z_ecef,
    double ra0, double dec0, double time_mjd);
```
Computes baseline uvw coordiantes from the specified array of antenna (station) coordinates and observation parameters.

- **[out] uu**      : Array of baseline uu coordinates, in metres.
- **[out] vv**      : Array of baseline vv coordinates, in metres.
- **[out] ww**      : Array of baseline ww coordinates, in metres.
- **[in] nant**     : Number of stations or antennas.
- **[in] x_ecef**   : Array of ECEF station x coordinates, in metres.
- **[in] y_ecef**   : Array of ECEF station y coordinates, in metres.
- **[in] z_ecef**   : Array of ECEF station z coordinates, in metres.
- **[in] ra0**      : Right Ascension of the observation direction, in radians.
- **[in] dec0**     : Declination of the observation direction, in radians.
- **[in] time_mjd** : Time centroid of the observation in Modified Julian Days.

--------------------------------------------------------------------------------


```C
double uvwsim_datetime_to_mjd(int year, int month, int day, int hour,
    int minute, double seconds);
```
Converts the specified date and time into Modified Julian Days.

- **[in] year**     : The year.
- **[in] month**    : The month.
- **[in] day**      : The day.
- **[in] hour**     : The hour.
- **[in] minute**   : The minute.
- **[in] seconds**  : Fractional seconds.
- **[rtn]**         : Date in Modified Julians Days.

--------------------------------------------------------------------------------



### Antenna (station) Layout files ###
Utility functions for loading antenna or station coordinates, require plain ASCII files where the coordinates to be loaded are stored column wise in the first two (2-dimensional data) or three columns (3-dimensional data). The coordinates for an antenna must be presented as a row in the file and consist of either two or three comma, tab or space separated values.

Empty rows that are empty or starting with a '#' character will be ignored. Note this format is compatible with all know CASA ASCII antenna configuration (*.cfg) files.

**Example:**

```
# Example layout file consisting of 4, antennas.
0.0,  100.0, 0.0
10.0, -20.0, 0.0
-5.0e-1, -70.0, 0.0
45.0,  1.0,  0.0
```

### Antenna (station) coordinate systems ###
The **uvwsim** library can make use of antenna coordinates in two different [frames of reference][ref1]:

- East-North-Up (ENU), Horizon, or local tangent plane coordinates. In this
  frame, the x-coordinate is the Eastwards direction, the y-coordinate points
  is in the Northwards direction, and the z-coordinate points towards towards
  the local zenith.
- Earth centred, Earth fixed ([ECEF][ref2]), or ITRF coordinates.

The function(s) in the **uvwsim** library for converting to baseline uvw coordinates (eg. `uvwsim_evaluate_baseline_uvw`) require antenna coordinates to be in an ECEF frame. A function called `uvwsim_convert_enu_to_ecef` is therefore provided to convert from ENU coordinates if needed.

[ref1]: http://en.wikipedia.org/wiki/North_East_Down#mediaviewer/File:ECEF_ENU_Longitude_Latitude_relationships.svg
[ref2]: http://en.wikipedia.org/wiki/ECEF

### Code Example: ###
The following code demonstrates the use of the **uvwsim** library.

This example, can be found in the doc folder and built with the command `gcc example.c -luvwsim -o uvsim_example` (or `gcc ../doc/example.c -I../src -L. -luvwsim -o uvwsim_example` from the build directory, if the library not yet installed.) In order to sucessfully run the example the VLA_A_hor_xyz.txt layout file must be present. This can be found in the '*test/data*' directory of the library source tree.

```C
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
```

If run sucessfully, this example produces a ASCII CSV file consisting of baseline coordinates in wavelengths called *example_baselines_vla_a.txt*. This can be plotted in your favorite plotting program. The following plot is a scatter plot of these results for the baseline coordinates *uu*, in wavelengths against *vv*, in wavelengths for the first time and channel.

![Figure showing scatter plot of example baselines. see: doc/example.png in the library source tree](http://www.oerc.ox.ac.uk/~ska/uvwsim/example.png "Scatter plot of baselines from the example code.")
