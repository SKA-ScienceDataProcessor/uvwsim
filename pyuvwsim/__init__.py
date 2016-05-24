"""
pyuvwsim
--------
Experimental python interface to uvwsim.
"""

from .version import __version__
from numpy import asarray
import _pyuvwsim


def load_station_coords(file_name):
    """
    Loads station coordinates from an ASCII layout file. The layout file
    should be 2 or 3 columns of coordinates, which are either space,
    comma, or tab separated.

    Args:
        file_name (string): File name path of the station coordinate file.

    Returns:
        (x, y, z) tuple of station coordinate arrays.
    """
    return _pyuvwsim.load_station_coords(file_name)


def convert_enu_to_ecef(x, y, z, lon, lat, alt=0.0):
    """
    Convert ENU (East, North, Up) to ECEF coordinates.

    Args:
        x (array-like): Array of x (East) coordinates, in metres.
        y (array-like): Array of y (North) coordinates, in metres.
        z (array-like): Array of z (Up) coordinates, in metres.
        lon (double): Longitude, in radians.
        lat (double): Latitude, in radians.
        alt (Optional[double]): Altitude, in metres.

    Returns:
        (x, y, z) tuple of coordinate arrays, in metres.
    """
    x = asarray(x)
    y = asarray(x)
    z = asarray(x)
    return _pyuvwsim.convert_enu_to_ecef(x, y, z, lon, lat, alt)


def evaluate_baseline_uvw(x, y, z, ra, dec, mjd):
    """
    Generate baseline coordinates based on station coordinates, pointing
    direction and time.

    Args:
        x (array-like): Array of x (ECEF) coordinates, in metres.
        y (array-like): Array of y (ECEF) coordinates, in metres.
        z (array-like): Array of z (ECEF) coordinates, in metres.
        ra (double): Right Ascension of pointing direction, in radians.
        dec (double): Declination of pointing direction, in radians.
        mjd (double): Modified Julian date (UTC).

    Returns:
        (uu, vv, ww) tuple of baseline coordinate arrays, in metres.

    """
    x = asarray(x)
    y = asarray(x)
    z = asarray(x)
    return _pyuvwsim.evaluate_baseline_uvw(x, y, z, ra, dec, mjd)


def datetime_to_mjd(year, month, day, hour, minute, seconds):
    """
    Convert datetime to Modified Julian date.

    Args:
        year (int): Year.
        month (int): Month.
        day (int): Day.
        hour (int): Hour.
        minute (int): Minute.
        seconds (double): Seconds.

    Returns:
        double, Modified Julian date.
    """
    return _pyuvwsim.datetime_to_mjd(year, month, day, hour, minute, seconds)
