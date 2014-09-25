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

#include <Python.h>
#include <uvwsim.h>
#include <pyuvwsim.h>

/* http://docs.scipy.org/doc/numpy-dev/reference/c-api.deprecations.html */
#define NPY_NO_DEPRECATED_API NPY_1_8_API_VERSION
#include <numpy/arrayobject.h>

#include <stdio.h>
#include <math.h>
#include <string.h>

/* Error objects */
static PyObject* pyuvwsimError;

static PyObject* load_station_coords(PyObject* self, PyObject* args)
{
    /*
     * http://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html
     * http://nedbatchelder.com/text/whirlext.html
     * https://docs.python.org/2/extending/extending.html#intermezzo-errors-and-exceptions
     */
    int n, nread;
    npy_intp dims;
    const char* filename_ = 0;
    PyObject *x_, *y_, *z_;
    double *x, *y, *z;

    if (!PyArg_ParseTuple(args, "s", &filename_)) {
        return NULL;
    }
    /*printf("filename = %s\n", filename_);*/

    /* Check if the file exists */
    if (!uvwsim_file_exists(filename_)) {
        PyErr_SetString(pyuvwsimError, "Specified station (antenna) "
            "layout file doesn't exist!");
        return NULL;
    }

    /* Find number of stations and load station coordinates */
    n = uvwsim_get_num_stations(filename_);

    /* Allocate arrays to hold coordinates. */
    dims = n;
    x_ = PyArray_SimpleNew(1, &dims, NPY_DOUBLE);
    y_ = PyArray_SimpleNew(1, &dims, NPY_DOUBLE);
    z_ = PyArray_SimpleNew(1, &dims, NPY_DOUBLE);
    x = (double*)PyArray_DATA((PyArrayObject*)x_);
    y = (double*)PyArray_DATA((PyArrayObject*)y_);
    z = (double*)PyArray_DATA((PyArrayObject*)z_);

    /* Read station coordinates. */
    nread = uvwsim_load_station_coords(filename_, n, x, y, z);
    if (nread != n) {
        PyErr_SetString(pyuvwsimError, "Layout file read error. Incorrect "
            "number of station coordinates read.");
        return NULL;
    }

    return Py_BuildValue("OOO", x_, y_, z_);
}

static PyObject* convert_enu_to_ecef(PyObject* self, PyObject* args)
{
    int typenum, requirements, nd, n;
    double lon, lat, alt;
    PyObject *x_enu_o=NULL, *y_enu_o=NULL, *z_enu_o=NULL;
    PyObject *x_enu_, *y_enu_, *z_enu_;
    npy_intp* dims;
    double *x_enu, *y_enu, *z_enu;
    PyObject *x_ecef_, *y_ecef_, *z_ecef_;
    double *x_ecef, *y_ecef, *z_ecef;

    /* Read input arguments */
    if (!PyArg_ParseTuple(args, "O!O!O!ddd", &PyArray_Type, &x_enu_o,
        &PyArray_Type, &y_enu_o, &PyArray_Type, &z_enu_o,
        &lon, &lat, &alt)) return NULL;
    /* printf("lon: %lf deg.\n", lon*180./M_PI); */
    /* printf("lat: %lf deg.\n", lat*180./M_PI); */
    /* printf("alt: %lf m\n", alt); */

    /* Convert Python objects to array of specified built-in data-type. */
    typenum = NPY_DOUBLE;
    /*requirements = NPY_IN_ARRAY; FIXME Python version check macro? */
    requirements = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
    x_enu_ = PyArray_FROM_OTF(x_enu_o, typenum, requirements);
    if (!x_enu_) return NULL;
    y_enu_ = PyArray_FROM_OTF(y_enu_o, typenum, requirements);
    if (!y_enu_) return NULL;
    z_enu_ = PyArray_FROM_OTF(z_enu_o, typenum, requirements);
    if (!z_enu_) return NULL;

    /* Extract dimensions and pointers. */
    /* TODO Require input arrays be 1D, and check dimension consistency. */
    nd = PyArray_NDIM((PyArrayObject*)x_enu_);
    dims = PyArray_DIMS((PyArrayObject*)x_enu_);
    x_enu = (double*)PyArray_DATA((PyArrayObject*)x_enu_);
    y_enu = (double*)PyArray_DATA((PyArrayObject*)y_enu_);
    z_enu = (double*)PyArray_DATA((PyArrayObject*)z_enu_);
    /* printf("x_enu: nd      = %i\n", nd); */
    /* printf("x_enu: dims[0] = %li\n", dims[0]); */
    /* printf("x_enu: [1-3]   = %lf, %lf, %lf\n", x_enu[0], x_enu[1], x_enu[2]); */

    /* Create New arrays for ECEF coordinates. */
    /* TODO in-place option?! */
    x_ecef_ = PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
    y_ecef_ = PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
    z_ecef_ = PyArray_SimpleNew(nd, dims, NPY_DOUBLE);
    x_ecef = (double*)PyArray_DATA((PyArrayObject*)x_ecef_);
    y_ecef = (double*)PyArray_DATA((PyArrayObject*)y_ecef_);
    z_ecef = (double*)PyArray_DATA((PyArrayObject*)z_ecef_);

    /* Call conversion function. */
    n = dims[0];
    uvwsim_convert_enu_to_ecef(n, x_ecef, y_ecef, z_ecef, x_enu,
            y_enu, z_enu, lon, lat, alt);

    /* Decrement references to temporary array objects. */
    Py_DECREF(x_enu_);
    Py_DECREF(y_enu_);
    Py_DECREF(z_enu_);

    /* Return station ECEF coordinates. */
    return Py_BuildValue("OOO", x_ecef_, y_ecef_, z_ecef_);

fail:  /* On fail, ...? */
    Py_XDECREF(x_enu_);
    Py_XDECREF(y_enu_);
    Py_XDECREF(z_enu_);
    return NULL;
}

static PyObject* evaluate_baseline_uvw(PyObject* self, PyObject* args)
{
    int typenum, requirements, n;
    PyObject *x_ecef_o=NULL, *y_ecef_o=NULL, *z_ecef_o=NULL;
    double ra0, dec0, mjd;
    PyObject *x_ecef_, *y_ecef_, *z_ecef_;
    npy_intp* dims;
    double *x_ecef, *y_ecef, *z_ecef;
    npy_intp nb;
    PyObject *uu_, *vv_, *ww_;
    double *uu, *vv, *ww;

    /* Read input arguments */
    if (!PyArg_ParseTuple(args, "O!O!O!ddd", &PyArray_Type, &x_ecef_o,
        &PyArray_Type, &y_ecef_o, &PyArray_Type, &z_ecef_o,
        &ra0, &dec0, &mjd)) return NULL;
    /* printf("ra0 : %lf deg.\n", ra0*180./M_PI); */
    /* printf("dec0: %lf deg.\n", dec0*180./M_PI); */
    /* printf("mjd : %lf\n", mjd); */

    /*  Convert Python objects to array of specified built-in data-type.*/
    typenum = NPY_DOUBLE;
    /* requirements = NPY_IN_ARRAY; FIXME Python version check macro? */
    requirements = NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_ALIGNED;
    x_ecef_ = PyArray_FROM_OTF(x_ecef_o, typenum, requirements);
    if (!x_ecef_) return NULL;
    y_ecef_ = PyArray_FROM_OTF(y_ecef_o, typenum, requirements);
    if (!y_ecef_) return NULL;
    z_ecef_ = PyArray_FROM_OTF(z_ecef_o, typenum, requirements);
    if (!z_ecef_) return NULL;

    /* Extract dimensions and pointers. */
    /* TODO Require input arrays be 1D, and check dimension consistency.
     * int nd = PyArray_NDIM(x_ecef_); */
    dims = PyArray_DIMS((PyArrayObject*)x_ecef_);
    x_ecef = (double*)PyArray_DATA((PyArrayObject*)x_ecef_);
    y_ecef = (double*)PyArray_DATA((PyArrayObject*)y_ecef_);
    z_ecef = (double*)PyArray_DATA((PyArrayObject*)z_ecef_);

    /* Create New arrays for baseline coordinates. */
    n = dims[0];
    nb = (n * (n-1)) / 2;
    uu_ = PyArray_SimpleNew(1, &nb, NPY_DOUBLE);
    vv_ = PyArray_SimpleNew(1, &nb, NPY_DOUBLE);
    ww_ = PyArray_SimpleNew(1, &nb, NPY_DOUBLE);
    uu = (double*)PyArray_DATA((PyArrayObject*)uu_);
    vv = (double*)PyArray_DATA((PyArrayObject*)vv_);
    ww = (double*)PyArray_DATA((PyArrayObject*)ww_);

    /* Call function to evaluate baseline uvw */
    uvwsim_evaluate_baseline_uvw(uu, vv, ww, n, x_ecef, y_ecef, z_ecef,
        ra0, dec0, mjd);

    /* Decrement references to local array objects. */
    Py_DECREF(x_ecef_);
    Py_DECREF(y_ecef_);
    Py_DECREF(z_ecef_);

    /* Return baseline coordinates. */
    return Py_BuildValue("OOO", uu_, vv_, ww_);

fail: /* On fail, ...? */
    Py_XDECREF(x_ecef_);
    Py_XDECREF(y_ecef_);
    Py_XDECREF(z_ecef_);
    return NULL;
}

static PyObject* datetime_to_mjd(PyObject* self, PyObject* args)
{
    /* Read input arguments */
    int year, month, day, hour, minute;
    double seconds, mjd;

    if (!PyArg_ParseTuple(args, "iiiiid", &year, &month, &day, &hour, &minute,
        &seconds)) return NULL;

    /* Call conversion function. */
    mjd = uvwsim_datetime_to_mjd(year, month, day, hour, minute, seconds);

    /* Return mjd. */
    return Py_BuildValue("d", mjd);
}



/* Method table. */
static PyMethodDef pyuvwsim_funcs[] =
{
    {
        "load_station_coords",
        (PyCFunction)load_station_coords, METH_VARARGS,
        "(x,y,z) = load_station_coords(filename)\n"
        "Loads station coordinates from an ASCII layout file. The layout file\n"
        "should be 2 or 3 columns of coordinates, which are either space, \n"
        "comma, or tab separated."
    },
    {
        "convert_enu_to_ecef",
        (PyCFunction)convert_enu_to_ecef, METH_VARARGS,
        "(x_ecef, y_ecef, z_ecef) = convert_enu_to_ecef(x_enu, y_enu, z_enu, lon, lat, alt)\n"
        "Converts ENU coordinates to ECEF.\n"
    },
    {
        "evaluate_baseline_uvw",
        (PyCFunction)evaluate_baseline_uvw, METH_VARARGS,
        "(uu, vv, ww) = evaluate_baseline_uvw(x_ecef, y_ecef, z_ecef, ra0, dec0, mjd)\n"
        "Generate baselines coordinates.\n"
    },
    {
        "datetime_to_mjd",
        (PyCFunction)datetime_to_mjd, METH_VARARGS,
        "mjd = datetime_to_mjd(year, month, day, hour, minute, seconds)\n"
        "Convert datetime to Modified Julian date.\n"
    },
    {NULL, NULL, 0, NULL}
};

/* Initialisation function (called init[filename] where filename = name of *.so) */
/* http://docs.scipy.org/doc/numpy/user/c-info.how-to-extend.html */
PyMODINIT_FUNC PYUVWSIM_API initpyuvwsim()
{
    PyObject* m = Py_InitModule3("pyuvwsim", pyuvwsim_funcs, "docstring...");
    /* (void) Py_InitModule3("pyuvwsim", pyuvwsim_funcs, "docstring..."); */

    /* Import the use of numpy array objects. */
    import_array();

    /* Create error objects and add them to the module. */
    pyuvwsimError = PyErr_NewException("pyuvwsim.error", NULL, NULL);
    Py_INCREF(pyuvwsimError);
    PyModule_AddObject(m, "error", pyuvwsimError);

    /*
    PyObject* d = PyModule_GetDict(m);
    PyObject* tuple = Py_BuildValue("(ii)",
        UVWSIM_VERSION_MAJOR,
        UVWSIM_VERSION_MINOR);
    PyDict_SetItemString(d, "uvwsim_version", tuple);
    Py_DECREF(tuple);
    */
}
