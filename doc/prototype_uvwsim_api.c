/*******************************************************************************
 * uvw coordinate simulator prototype API v1
 *******************************************************************************/

 // Proposed API Functions (in no particular order): 
 int uvwsim_get_num_stations(const char* layout_file);
 int uvwsim_load_station_coords(const char* layout_file, int nant, 
    double* x, double*y, double* z, char sep);
 void uvwsim_convert_enu_to_ecef(int nant, double* x, double* y, double* z,
     double lon, double lat, double alt);
 int uvwsim_num_baselines(int nant);
 void uvwsim_evaluate_basaeline_uvw(double* uu, double* vv, double* vv,
            int nant, double* x, double* y, double* z, double ra0, double dec0,
            double time_mjd);
 double uvwsim_datetime_to_mjd(int year, int month, int day, int hour, 
    int minute, double seconds);
 
// Example main.
int main()
{
    //-------------------------------------------------------------------------
    const char* filename = "layout.txt"; // Station layout file. ASCII CSV format or similar
    double lon  =  10.0 * (M_PI/180.);   // radians (for horizon)
    double lat  = -30.0 * (M_PI/180.);   // radians (for horizon)
    double alt  = 0.0;                   // metres (for horizon)
    int ntimes  = 100;                   // Number of times at which to evaluate
                                         // baselines
    double ra0  = 10.0 * (M_PI/180.);    // radians
    double dec0 = -25.0 * (M_PI/180.);   // radians
    double start_time_mjd = 12345.0;     // Start time MJD, in seconds
                                         // --> provide datetime to MJD method
    double interval   = 10.0 / 86400.0;  // days (uvw coordinate interval)
    int nchannels     = 100;
    double start_freq = 100.0e6;         // Hz
    double freq_inc   = 100.0e3;         // Hz
    //-------------------------------------------------------------------------

    /***************************************************************************
     * Obtain station coordinates in an Earth centred frame.
     * - A method is provided to load station coordinates from the typical
     *   CSV style ASCII layout file as provided by the SKAO
     * - If the coordinates are loaded in horizon plane coordinates 
     *   (x=East, y=North, z=Up), a method is provided to perform the 
     *   coordiante conversion.
     * - If coordinates are already in an Earth centred frame (such as ITRF)
     *   then they can be used directly.
     **************************************************************************/

    // Obtain the number of antennas in the station layout file
    // -> This function is required unless the loader were to allocate 
    //    coordinate arrays or the number of antennas obtained elsewhere.
    //    We feel that having hidden memory allocation in the loader would make
    //    the API less transparent and therefore harder to use.
    int nant = uvwsim_get_num_stations("layout.txt")

    // Allocate arrays for station coordiantes.
    double* x = (double*)malloc(nant * sizeof(double));
    double* y = (double*)malloc(nant * sizeof(double));
    double* z = (double*)malloc(nant * sizeof(double));

    // Load station coordinates.
    // -> Note that by returning the number of coordinates this can be used
    //    for error checking.
    int num_read = uvwsim_load_station_coords(filename, nant, x, y, z, sep=',');
    if (num_read != nant) return EXIT_FAILURE;
    
    // Convert from Horizon coordinates (East, North, Up) to Earth centred
    // coordinate (ECEF).
    // -> Note the prototype here is for an in-place conversion but this 
    //    might not be a good idea in general as it limits flexibility a bit.
    uvwsim_convert_enu_to_ecef(nant, x, y, z, lon, lat, alt)

    /**************************************************************************
     * Convert station coordinates to baseline coordinates.
     * - The API shown below is very low level (and flexible) but we could 
     *   certainly consider wrapping some of the loops into higher level 
     *   functions.
     **************************************************************************/

    // Allocate arrays for baseline uvw coordinates.
    int nbaselines = uvwsim_num_baselines(nant);
    int ncoords    = nbaselines * ntimes;
    double* uu     = (double*)malloc(ncoords * sizeof(double));
    double* vv     = (double*)malloc(ncoords * sizeof(double));
    double* ww     = (double*)malloc(ncoords * sizeof(double));

    // Loop over times to generate baseline coordinates in metres.
    // -> Note if performance is critical the function to evaluate 
    //    baseline uvw coordinates will either need to take a work array
    //    or be split into two functions: A function that computes station uvw,
    //    followed by convertion from station uvw to baseline uvw.
    //    This would avoid otherwise unnessary internal memory operations.
    for (int i = 0; i < ntimes; ++i) {
        int offset = i * nbaselines;
        double time_mjd = start_time_mjd + internal * i;
        uvwsim_evaluate_baseline_uvw(&uu[offset], &vv[offset], &vv[offset],
            nant, x, y, z, ra0, dec0, time_mjd);
    }

    // Scale baselines into wavelengths (if required)
    const double c0 = 299792458.0;
    double* uu_wavelengths = (double*)malloc(ncoords * nfreq * sizeof(double));
    double* vv_wavelengths = (double*)malloc(ncoords * nfreq * sizeof(double));
    double* ww_wavelengths = (double*)malloc(ncoords * nfreq * sizeof(double));
    for (int idx = 0, j = 0; j < nfreqs; ++j) {
        double freq = start_freq + freq_inc * j;
        for (int i = 0; i < ncoords; ++i) {
            uu_wavelengths[idx] = uu[i] * (freq / c0);
            vv_wavelengths[idx] = vv[i] * (freq / c0);
            ww_wavelengths[idx] = ww[i] * (freq / c0);
        }
    }

    // Cleanup.
    free(x);
    free(y);
    free(z);
    free(uu);
    free(vv);
    free(ww);
    free(uu_wavelengths);
    free(vv_wavelengths);
    free(ww_wavelengths);
}
