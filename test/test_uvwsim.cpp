/*
 * Copyright (c) 2014, The University of Oxford
 * All rights reserved.
 *
 * This file is part of the uvwsim library.
 * Contact: uvwsim at oerc.ox.ac.uk
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <gtest/gtest.h>

#include <uvwsim.h>

#include <cstdlib>
#include <cstdio>
#include <ctime>
#include <cmath>

/* Bool flag that can be set to true(1) if testing with CASA layout files */
#define HAVE_CASA_CFG_FILES 0

static double frand(double min, double max)
{
    double f = (double)rand() / RAND_MAX;
    return min + f * (max - min);
}

TEST(uvwsim, get_num_stations)
{
    bool deterministic = true;
    if (!deterministic) srand(time(NULL));

    // Create a temporary station file with 3 columns.
    {
        int nant = deterministic ? 500 : (rand() % 500 + 1);
        const char* filename = "TEMP_layout_xyz.txt";
        FILE* fp = fopen(filename, "w");
        fprintf(fp, "# This is a comment\n");
        fprintf(fp, "# This is a comment\n");
        fprintf(fp, "# This is a comment\n");
        for (int i = 0; i < nant; ++i) {
            double x,y,z;
            if (deterministic) {
                x = -200.0 + 0.35*(double)i;
                y = -182.0 + 0.20*(double)i;
                z = -174.0 + 0.18*(double)i;
            } else {
                x = frand(-200, 200);
                y = frand(-200, 200);
                z = frand(-200, 200);
            }
            fprintf(fp, "%f,%f,%f\n", x, y, z);
        }
        fprintf(fp, "# This is a comment");
        fclose(fp);

        ASSERT_TRUE(uvwsim_file_exists(filename))
        << "Unable to locate input layout file: " << filename << "\n";
        ASSERT_EQ(nant, uvwsim_get_num_stations(filename));

        // Remove the temporary layout file
        remove(filename);
    }

    // Create a temporary station file with 2 columns.
    {
        int nant = deterministic ? 500 : (rand() % 500 + 1);
        const char* filename = "TEMP_layout_xy.txt";
        FILE* fp = fopen(filename, "w");
        for (int i = 0; i < nant; ++i) {
            double x, y;
            if (deterministic) {
                x = -200.0 + 0.35*(double)i;
                y = -182.0 + 0.20*(double)i;
            } else {
                if (rand()%2) fprintf(fp, "# This is a comment\n");
                x = frand(-200, 200);
                y = frand(-200, 200);
            }
            fprintf(fp, "%f,%f\n", x, y);
        }
        fclose(fp);

        ASSERT_TRUE(uvwsim_file_exists(filename))
        << "Unable to locate input layout file: " << filename << "\n";
        ASSERT_EQ(nant, uvwsim_get_num_stations(filename));

        // Remove the temporary layout file
        remove(filename);
    }

#if HAVE_CASA_CFG_FILES
    {
        const char* filename = "WSRT.cfg";
        ASSERT_TRUE(uvwsim_file_exists(filename))
        << "Unable to locate input layout file: " << filename << "\n";
        ASSERT_EQ(14, uvwsim_get_num_stations(filename));
    }

    {
        const char* filename = "meerkat.cfg";
        ASSERT_TRUE(uvwsim_file_exists(filename))
        << "Unable to locate input layout file: " << filename << "\n";
        ASSERT_EQ(80, uvwsim_get_num_stations(filename));
    }
#endif
}

TEST(uvwsim, load_station_coords)
{
    bool deterministic = true;
    if (!deterministic) srand(time(NULL));

    // Create a temporary station file with 3 columns.
    {
        int nant = deterministic ? 500 : (rand() % 500 + 1);
        const char* filename = "TEMP_layout_xyz.txt";
        double* x_in = (double*)malloc(nant * sizeof(double));
        double* y_in = (double*)malloc(nant * sizeof(double));
        double* z_in = (double*)malloc(nant * sizeof(double));
        FILE* fp = fopen(filename, "w");
        for (int i = 0; i < nant; ++i)
        {
            if (deterministic) {
                x_in[i] = -200.0 + 0.35*(double)i;
                y_in[i] = -182.0 + 0.20*(double)i;
                z_in[i] = -174.0 + 0.18*(double)i;
            } else {
                x_in[i] = frand(-200, 200);
                y_in[i] = frand(-200, 200);
                z_in[i] = frand(-200, 200);
            }
            fprintf(fp, "%.16e,%.16e,%.16e\n", x_in[i], y_in[i], z_in[i]);
        }
        fclose(fp);

        double* x = (double*)malloc(nant * sizeof(double));
        double* y = (double*)malloc(nant * sizeof(double));
        double* z = (double*)malloc(nant * sizeof(double));
        ASSERT_EQ(nant, uvwsim_load_station_coords(filename, nant, x, y, z));

        /* check results */
        for (int i = 0; i < nant; ++i)
        {
            ASSERT_DOUBLE_EQ(x_in[i], x[i]);
            ASSERT_DOUBLE_EQ(y_in[i], y[i]);
            ASSERT_DOUBLE_EQ(z_in[i], z[i]);
        }

        // Remove the temporary layout file
        remove(filename);
        free(x);
        free(y);
        free(z);
        free(x_in);
        free(y_in);
        free(z_in);
    }

    // Create a temporary station file with 2 columns.
    {
        int nant = deterministic ? 500 : (rand() % 500 + 1);
        const char* filename = "TEMP_layout_xy.txt";
        double* x_in = (double*)malloc(nant * sizeof(double));
        double* y_in = (double*)malloc(nant * sizeof(double));
        FILE* fp = fopen(filename, "w");
        for (int i = 0; i < nant; ++i)
        {
            if (deterministic) {
                x_in[i] = -200.0 + 0.35*(double)i;
                y_in[i] = -182.0 + 0.20*(double)i;
            } else {
                if (rand()%2) fprintf(fp, "# This is a comment\n");
                x_in[i] = frand(-200, 200);
                y_in[i] = frand(-200, 200);
            }
            fprintf(fp, "%.16e,%.16e\n", x_in[i], y_in[i]);
        }
        fclose(fp);

        double* x = (double*)malloc(nant * sizeof(double));
        double* y = (double*)malloc(nant * sizeof(double));
        double* z = (double*)malloc(nant * sizeof(double));
        ASSERT_EQ(nant, uvwsim_load_station_coords(filename, nant, x, y, z));

        /* check results */
        for (int i = 0; i < nant; ++i)
        {
            ASSERT_DOUBLE_EQ(x_in[i], x[i]);
            ASSERT_DOUBLE_EQ(y_in[i], y[i]);
            ASSERT_DOUBLE_EQ(0.0, z[i]);
        }

        // Remove the temporary layout file
        remove(filename);
        free(x);
        free(y);
        free(z);
        free(x_in);
        free(y_in);
    }

    // Configuration file using various allowed separators.
    {
        const char* filename = "test_sep.cfg";
        int nant = 5;
        FILE* fp = fopen(filename, "w");
        // single comma & single space.
        fprintf(fp, "%f,%f %f\n",1.0,2.0,3.0);
        // single tab, comma-space
        fprintf(fp, "%f\t%f, %f\n",4.0,5.0,6.0);
        // comma-tab, tab-space
        fprintf(fp, "%f,\t%e\t %f\n",7.0,8.0,9.0);
        // space-tab, space-comma
        fprintf(fp, "%f \t%f ,%e\n",10.0,11.0,12.0);
        // space-comma-tab, spacex3-comma
        fprintf(fp, "%e ,\t%f   ,%f\n",13.0,14.0,15.0);
        fclose(fp);

        double* x = (double*)malloc(nant * sizeof(double));
        double* y = (double*)malloc(nant * sizeof(double));
        double* z = (double*)malloc(nant * sizeof(double));
        ASSERT_EQ(nant, uvwsim_load_station_coords(filename, nant, x, y, z));

        /* check results */
        for (int i = 0; i < nant; ++i) {
            ASSERT_DOUBLE_EQ((double)i*3.0+1.0, x[i]);
            ASSERT_DOUBLE_EQ((double)i*3.0+2.0, y[i]);
            ASSERT_DOUBLE_EQ((double)i*3.0+3.0, z[i]);
        }

        // Cleanup and remove temporary layout file.
        free(x);
        free(y);
        free(z);
        remove(filename);
    }
}

TEST(uvwsim, convert_enu_to_ecef)
{
    bool deterministic = true;
    if (!deterministic) srand(time(NULL));

    /* Note: This is not exactly a bullet-proof test at least checks nothing
     * has changed ... */
    /* By setting lon, lat to 270.0, 90.0 the rotation matrix going from enu
     * to ecef is an identity matrix
     */
    double lon = 270.0 * (M_PI/180.);
    double lat = 90.0  * (M_PI/180.);
    double alt = 0.0;
    int nant = 500;
    double* x_in = (double*)malloc(nant * sizeof(double));
    double* y_in = (double*)malloc(nant * sizeof(double));
    double* z_in = (double*)malloc(nant * sizeof(double));
    for (int i = 0; i < nant; ++i) {
        if (deterministic) {
            x_in[i] = -200.0 + 0.35*(double)i;
            y_in[i] = -182.0 + 0.20*(double)i;
            z_in[i] = -174.0 + 0.18*(double)i;
        } else {
            x_in[i] = frand(-200, 200);
            y_in[i] = frand(-200, 200);
            z_in[i] = frand(-200, 200);
        }
    }
    double* x = (double*)malloc(nant * sizeof(double));
    double* y = (double*)malloc(nant * sizeof(double));
    double* z = (double*)malloc(nant * sizeof(double));

    uvwsim_convert_enu_to_ecef(nant, x, y, z, x_in, y_in, z_in, lon, lat, alt);
    for (int i = 0; i < nant; ++i) {
        ASSERT_NEAR(x[i], x_in[i], 1.0e-8);
        ASSERT_NEAR(y[i], y_in[i], 1.0e-8);
        ASSERT_NEAR(z[i], z_in[i]+6.3567523140e+06, 1.0e-8);
    }

    lon = 0.0 * (M_PI/180.);
    lat = 90.0 * (M_PI/180.);
    uvwsim_convert_enu_to_ecef(nant, x, y, z, x_in, y_in, z_in, lon, lat, alt);
    for (int i = 0; i < nant; ++i) {
        ASSERT_NEAR(x[i], -y_in[i], 1.0e-8);
        ASSERT_NEAR(y[i], x_in[i], 1.0e-8);
        ASSERT_NEAR(z[i], z_in[i]+6.3567523140e+06, 1.0e-8);
    }

    free(x_in);
    free(y_in);
    free(z_in);
    free(x);
    free(y);
    free(z);
}

TEST(uvwsim, num_baselines)
{
    bool deterministic = true;
    if (!deterministic) srand(time(NULL));
    int nant = deterministic ? 500 : (rand() % 500 + 1);
    ASSERT_EQ(uvwsim_num_baselines(nant), (nant*(nant-1)/2));
}

TEST(uvwsim, evaluate_uvw_vla_a_hor)
{
    srand(time(NULL));

    {
        const char* layout_file = "VLA_A_hor_xyz.txt";
        ASSERT_TRUE(uvwsim_file_exists(layout_file))
            << "Unable to locate input layout file: " << layout_file << "\n";

        double lon  = -107.6184 * (M_PI/180.);
        double lat  = 34.0790 * (M_PI/180.);
        double alt  = 0.0;
        double ra0  = 0.0 * (M_PI/180.);
        double dec0 = 90.0 * (M_PI/180.);
        int ntimes = 1;
        double start_mjd = uvwsim_datetime_to_mjd(2014, 03, 20, 14, 57, 0.0);
        double length_days = 0.1;
        int nchan = 20;
        double start_freq = 500.0e6;
        double end_freq = 1500.0e6;
        bool save_hermitian = true;

        int nant = uvwsim_get_num_stations(layout_file);
        ASSERT_EQ(27, nant);
        double* x = (double*)malloc(nant * sizeof(double));
        double* y = (double*)malloc(nant * sizeof(double));
        double* z = (double*)malloc(nant * sizeof(double));
        ASSERT_EQ(nant, uvwsim_load_station_coords(layout_file, nant, x, y, z));
        int nbaselines = uvwsim_num_baselines(nant);
        int ncoords = nbaselines * ntimes;
        double* uu = (double*)malloc(ncoords * sizeof(double));
        double* vv = (double*)malloc(ncoords * sizeof(double));
        double* ww = (double*)malloc(ncoords * sizeof(double));
        uvwsim_convert_enu_to_ecef(nant, x, y, z, x, y, z, lon, lat, alt);

        for (int t = 0; t < ntimes; ++t)
        {
            double time_mjd = start_mjd + t * (length_days/(double)ntimes);
            int offset = nbaselines * t;
            uvwsim_evaluate_baseline_uvw(&uu[offset], &vv[offset], &ww[offset],
                    nant, x, y, z, ra0, dec0, time_mjd);
        }

        /* Save baseline coordinates in metres */
        {
            const char* uvw_file = "TEMP_VLA_A_uvw_metres.csv";
            FILE* fp = fopen(uvw_file, "w");
            for (int i = 0; i < ncoords; ++i) {
                fprintf(fp, "%f,%f,%f\n", uu[i], vv[i], ww[i]);
                /* Write Hermitian coordinates to symmetrise the plot */
                if (save_hermitian)
                    fprintf(fp, "%f,%f,%f\n", -uu[i], -vv[i], -ww[i]);
            }
            fclose(fp);
        }

        /* Save frequency scaled baseline coordinates */
        {
            const char* uvw_file = "TEMP_VLA_A_uvw_wavelengths.csv";
            FILE* fp = fopen(uvw_file, "w");
            double freq_inc = (end_freq-start_freq)/(double)nchan;
            const double c0 = 299792458.0;
            for (int c = 0; c < nchan; ++c) {
                double freq = start_freq + c * freq_inc;
                double freq_scale = (freq / c0);
                for (int i = 0; i < ncoords; ++i) {
                    double uu_ = uu[i] * freq_scale;
                    double vv_ = vv[i] * freq_scale;
                    double ww_ = ww[i] * freq_scale;
                    fprintf(fp, "%f,%f,%f\n", uu_, vv_, ww_);
                    /* Write Hermitian coordinates to symmetrise the plot */
                    if (save_hermitian)
                        fprintf(fp, "%f,%f,%f\n", -uu_, -vv_, -ww_);
                }
            }
            fclose(fp);
        }

        free(uu);
        free(vv);
        free(ww);
        free(x);
        free(y);
        free(z);
    }

    /* Plotting command with GNUPLOT:

        gnuplot
        set term x11
        set datafile separator ","
        plot 'test/TEMP_VLA_A_uvw_metres.csv' using 1:2

     */
}

#if HAVE_CASA_CFG_FILES
TEST(uvwsim, evaluate_uvw_vla_a_itrf)
{
    srand(time(NULL));

    {
        const char* layout_file = "vla.a.cfg";
        ASSERT_TRUE(uvwsim_file_exists(layout_file))
        << "Unable to locate input layout file: " << layout_file << "\n";

        double ra0  = 0.0 * (M_PI/180.);
        double dec0 = 90.0 * (M_PI/180.);
        int ntimes = 1;
        double start_mjd = uvwsim_datetime_to_mjd(2014, 03, 20, 14, 57, 0.0);
        double length_days = 0.1;
        int nchan = 20;
        double start_freq = 500.0e6;
        double end_freq = 1500.0e6;
        bool save_hermitian = true;

        int nant = uvwsim_get_num_stations(layout_file);
        ASSERT_EQ(27, nant);
        double* x = (double*)malloc(nant * sizeof(double));
        double* y = (double*)malloc(nant * sizeof(double));
        double* z = (double*)malloc(nant * sizeof(double));
        ASSERT_EQ(nant, uvwsim_load_station_coords(layout_file, nant, x, y, z));
        int nbaselines = uvwsim_num_baselines(nant);
        int ncoords = nbaselines * ntimes;
        double* uu = (double*)malloc(ncoords * sizeof(double));
        double* vv = (double*)malloc(ncoords * sizeof(double));
        double* ww = (double*)malloc(ncoords * sizeof(double));

        for (int t = 0; t < ntimes; ++t)
        {
            double time_mjd = start_mjd + t * (length_days/(double)ntimes);
            int offset = nbaselines * t;
            uvwsim_evaluate_baseline_uvw(&uu[offset], &vv[offset], &ww[offset],
                    nant, x, y, z, ra0, dec0, time_mjd);
        }

        /* Save baseline coordinates in metres */
        {
            const char* uvw_file = "TEMP_VLA_A_2_uvw_metres.csv";
            FILE* fp = fopen(uvw_file, "w");
            for (int i = 0; i < ncoords; ++i) {
                fprintf(fp, "%f,%f,%f\n", uu[i], vv[i], ww[i]);
                /* Write Hermitian coordinates to symmetrise the plot */
                if (save_hermitian)
                    fprintf(fp, "%f,%f,%f\n", -uu[i], -vv[i], -ww[i]);
            }
            fclose(fp);
        }

        /* Save frequency scaled baseline coordinates */
        {
            const char* uvw_file = "TEMP_VLA_A_2_uvw_wavelengths.csv";
            FILE* fp = fopen(uvw_file, "w");
            double freq_inc = (end_freq-start_freq)/(double)nchan;
            const double c0 = 299792458.0;
            for (int c = 0; c < nchan; ++c) {
                double freq = start_freq + c * freq_inc;
                double freq_scale = (freq / c0);
                for (int i = 0; i < ncoords; ++i) {
                    double uu_ = uu[i] * freq_scale;
                    double vv_ = vv[i] * freq_scale;
                    double ww_ = ww[i] * freq_scale;
                    fprintf(fp, "%f,%f,%f\n", uu_, vv_, ww_);
                    /* Write Hermitian coordinates to symmetrise the plot */
                    if (save_hermitian)
                        fprintf(fp, "%f,%f,%f\n", -uu_, -vv_, -ww_);
                }
            }
            fclose(fp);
        }

        free(uu);
        free(vv);
        free(ww);
        free(x);
        free(y);
        free(z);
    }
}
#endif


TEST(uvwsim, datetime_to_mjd)
{
    ASSERT_DOUBLE_EQ(56856.0, uvwsim_datetime_to_mjd(2014, 7, 18, 0, 0, 0.0));
    ASSERT_DOUBLE_EQ(56856.713021990843, uvwsim_datetime_to_mjd(2014, 7, 18, 17, 06, 45.1));
}
