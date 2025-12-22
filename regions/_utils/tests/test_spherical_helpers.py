# Licensed under a 3-clause BSD style license - see LICENSE.rst

from astropy import units as u
from astropy.coordinates import SkyCoord
from numpy.testing import assert_allclose

from regions._utils.spherical_helpers import (
    _get_foci_direct_calc, discretize_spherical_ellipse_boundary)


class TestSphericalEllipseHelpers:
    center = SkyCoord(3, 4, unit='deg')
    width = 4 * u.deg
    height = 3 * u.deg
    angle = 5 * u.deg

    def test_boundary_discretize(self):
        n_points = 100

        # Get foci:
        p1, p2 = _get_foci_direct_calc(
            self.center, self.angle, self.width / 2., self.height / 2.
        )

        # Get boundary:
        bound_verts = discretize_spherical_ellipse_boundary(
            self.center, self.width, self.height, self.angle,
            n_points
        )

        # Get separations of coordinates and foci:
        sep_p1 = p1.separation(bound_verts)
        sep_p2 = p2.separation(bound_verts)

        # Boundary: sep_p1 + sep_p2 = 2a = width
        assert_allclose(sep_p1 + sep_p2, self.width)
