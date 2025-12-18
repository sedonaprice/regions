# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines line regions in both pixel and sky coordinates.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord

from regions._utils.spherical_helpers import (
    discretize_line_boundary, get_line_edge_raw_lonlat_bounds_circ_edges)
from regions.core.attributes import (RegionMetaDescr, RegionVisualDescr,
                                     ScalarPixCoord, ScalarSkyCoord)
from regions.core.bounding_box import RegionBoundingBox
from regions.core.core import PixelRegion, SkyRegion, SphericalSkyRegion
from regions.core.metadata import RegionMeta, RegionVisual
from regions.core.pixcoord import PixCoord

from .circle import CircleSphericalSkyRegion

__all__ = ['LinePixelRegion', 'LineSkyRegion', 'LineSphericalSkyRegion']


class LinePixelRegion(PixelRegion):
    """
    A line in pixel coordinates.

    Parameters
    ----------
    start : `~regions.PixCoord`
        The start position.
    end : `~regions.PixCoord`
        The end position.
    meta : `~regions.RegionMeta` or `dict`, optional
        A dictionary that stores the meta attributes of the region.
    visual : `~regions.RegionVisual` or `dict`, optional
        A dictionary that stores the visual meta attributes of the
        region.

    Examples
    --------
    .. plot::
        :include-source:

        from regions import PixCoord, LinePixelRegion
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)

        start = PixCoord(x=15, y=10)
        end = PixCoord(x=20, y=25)
        reg = LinePixelRegion(start=start, end=end)
        patch = reg.plot(ax=ax, edgecolor='red', lw=2, label='Line')

        ax.legend(handles=(patch,), loc='upper center')
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 30)
        ax.set_aspect('equal')
    """

    _params = ('start', 'end')
    _mpl_artist = 'Patch'
    start = ScalarPixCoord('The start pixel position as a |PixCoord|.')
    end = ScalarPixCoord('The end pixel position as a |PixCoord|.')
    meta = RegionMetaDescr('The meta attributes as a |RegionMeta|')
    visual = RegionVisualDescr('The visual attributes as a |RegionVisual|.')

    def __init__(self, start, end, meta=None, visual=None):
        self.start = start
        self.end = end
        self.meta = meta or RegionMeta()
        self.visual = visual or RegionVisual()

    @property
    def area(self):
        return 0

    def contains(self, pixcoord):
        in_reg = (False if pixcoord.isscalar
                  else np.zeros(pixcoord.x.shape, dtype=bool))

        if self.meta.get('include', True):
            return in_reg
        else:
            return np.logical_not(in_reg)

    def discretize_boundary(self, n_points=100):
        """
        Discretize the boundary into a CompoundPixelRegion, containing
        multiple individual LinePixelRegion instances as the line
        segments.

        Parameters
        ----------
        n_points : int, optional
            Number of points along the line's boundary.

        Returns
        -------
        line: `~regions.CompoundPixelRegion`
            Planar CompoundPixelRegion object,
            consisting of the union of multiple LinePixelRegion segments.
        """
        # Parametric equation to sample line:
        t = np.linspace(0, 1, num=n_points, endpoint=True)
        xs = self.start.x + t * (self.end.x - self.start.x)
        ys = self.start.y + t * (self.end.y - self.start.y)

        # Create a CompoundPixelRegion out of these segments:
        line = None
        for i in range(n_points - 1):
            lseg = LinePixelRegion(
                PixCoord(xs[i], ys[i]), PixCoord(xs[i + 1], ys[i + 1]),
                meta=self.meta.copy(),
                visual=self.visual.copy(),
            )
            line = lseg if line is None else line | lseg

        return line

    def to_sky(self, wcs):
        start = wcs.pixel_to_world(self.start.x, self.start.y)
        end = wcs.pixel_to_world(self.end.x, self.end.y)
        return LineSkyRegion(start, end, meta=self.meta.copy(),
                             visual=self.visual.copy())

    def to_spherical_sky(self, wcs=None, include_boundary_distortions=False,
                         discretize_kwargs=None):

        if discretize_kwargs is None:
            discretize_kwargs = {}

        if include_boundary_distortions:
            if wcs is None:
                raise ValueError(
                    "'wcs' must be set if 'include_boundary_distortions'=True"
                )
            # Requires planar to spherical projection (using WCS) and discretization
            # Will require implementing discretization in pixel space
            # to get correct handling of distortions.
            raise NotImplementedError

        start = wcs.pixel_to_world(self.start.x, self.start.y)
        end = wcs.pixel_to_world(self.end.x, self.end.y)

        return LineSphericalSkyRegion(
            start, end,
            meta=self.meta.copy(),
            visual=self.visual.copy()
        )

    @property
    def bounding_box(self):
        xmin = min(self.start.x, self.end.x)
        xmax = max(self.start.x, self.end.x)
        ymin = min(self.start.y, self.end.y)
        ymax = max(self.start.y, self.end.y)

        return RegionBoundingBox.from_float(xmin, xmax, ymin, ymax)

    def to_mask(self, mode='center', subpixels=5):
        # TODO: needs to be implemented
        raise NotImplementedError

    def as_artist(self, origin=(0, 0), **kwargs):
        """
        Return a matplotlib patch object for this region
        (`matplotlib.patches.Arrow`).

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` pixel position of the origin of the displayed
            image.

        **kwargs : dict
            Any keyword arguments accepted by
            `~matplotlib.patches.Arrow`. These keywords will override
            any visual meta attributes of this region.

        Returns
        -------
        artist : `~matplotlib.patches.Arrow`
            A matplotlib line patch.
        """
        # Note: Long term we want to support DS9 lines with arrow heads.
        # We may want to use Line2D instead of arrow for lines because the
        # width of the arrow is non-scalable in patches.
        from matplotlib.patches import Arrow

        x = self.start.x - origin[0]
        y = self.start.y - origin[1]
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        kwargs.setdefault('width', 0.1)

        mpl_kwargs = self.visual.define_mpl_kwargs(self._mpl_artist)
        mpl_kwargs.update(kwargs)

        return Arrow(x, y, dx, dy, **mpl_kwargs)

    def rotate(self, center, angle):
        """
        Rotate the region.

        Positive ``angle`` corresponds to counter-clockwise rotation.

        Parameters
        ----------
        center : `~regions.PixCoord`
            The rotation center point.
        angle : `~astropy.coordinates.Angle`
            The rotation angle.

        Returns
        -------
        region : `LinePixelRegion`
            The rotated region (which is an independent copy).
        """
        start = self.start.rotate(center, angle)
        end = self.end.rotate(center, angle)
        return self.copy(start=start, end=end)


class LineSkyRegion(SkyRegion):
    """
    A line in sky coordinates.

    Parameters
    ----------
    start : `~astropy.coordinates.SkyCoord`
        The start position.
    end : `~astropy.coordinates.SkyCoord`
        The end position.
    meta : `~regions.RegionMeta` or `dict`, optional
        A dictionary that stores the meta attributes of the region.
    visual : `~regions.RegionVisual` or `dict`, optional
        A dictionary that stores the visual meta attributes of the
        region.
    """

    _params = ('start', 'end')
    start = ScalarSkyCoord('The start position as a |SkyCoord|.')
    end = ScalarSkyCoord('The end position as a |SkyCoord|.')
    meta = RegionMetaDescr('The meta attributes as a |RegionMeta|')
    visual = RegionVisualDescr('The visual attributes as a |RegionVisual|.')

    def __init__(self, start, end, meta=None, visual=None):
        self.start = start
        self.end = end
        self.meta = meta or RegionMeta()
        self.visual = visual or RegionVisual()

    def contains(self, skycoord, wcs):  # pylint: disable=unused-argument
        # lines never contain anything
        return not self.meta.get('include', True)

    def discretize_boundary(self, wcs, n_points=100):
        """
        Discretize the boundary into a CompoundSkyRegion, containing
        multiple individual LineSkyRegion instances as the line
        segments.

        As LineSkyRegion is planar, this requires a WCS instance
        to map to a specified plane projection.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The world coordinate system transformation to use to convert
            between sky and pixel coordinates.

        n_points : int, optional
            Number of points along the line's boundary.

        Returns
        -------
        line: `~regions.CompoundSkyRegion`
            Planar CompoundSkyRegion object,
            consisting of the union of multiple LineSkyRegion segments.
        """
        # Transform to LinePixelRegion, discretize, and then
        # convert back to a SkyRegion

        pixline = self.to_pixel(wcs)
        disc_pixline = pixline.discretize_boundary(n_points=n_points)

        return disc_pixline.to_sky(wcs)

    def to_pixel(self, wcs):
        start_x, start_y = wcs.world_to_pixel(self.start)
        start = PixCoord(start_x, start_y)
        end_x, end_y = wcs.world_to_pixel(self.end)
        end = PixCoord(end_x, end_y)
        return LinePixelRegion(start, end, meta=self.meta.copy(),
                               visual=self.visual.copy())

    def to_spherical_sky(self, wcs=None, include_boundary_distortions=False,
                         discretize_kwargs=None):
        if discretize_kwargs is None:
            discretize_kwargs = {}

        if include_boundary_distortions:
            if wcs is None:
                raise ValueError(
                    "'wcs' must be set if 'include_boundary_distortions'=True"
                )
            # Requires planar to spherical projection (using WCS) and discretization
            # Will require implementing discretization in pixel space
            # to get correct handling of distortions.
            raise NotImplementedError

        return LineSphericalSkyRegion(
            self.start, self.end,
            meta=self.meta.copy(),
            visual=self.visual.copy()
        )


class LineSphericalSkyRegion(SphericalSkyRegion):
    """
    A line in spherical sky coordinates.

    Parameters
    ----------
    start : `~astropy.coordinates.SkyCoord`
        The start position.
    end : `~astropy.coordinates.SkyCoord`
        The end position.
    meta : `~regions.RegionMeta` or `dict`, optional
        A dictionary that stores the meta attributes of the region.
    visual : `~regions.RegionVisual` or `dict`, optional
        A dictionary that stores the visual meta attributes of the
        region.
    """

    _params = ('start', 'end')
    start = ScalarSkyCoord('The start position as a |SkyCoord|.')
    end = ScalarSkyCoord('The end position as a |SkyCoord|.')
    meta = RegionMetaDescr('The meta attributes as a |RegionMeta|')
    visual = RegionVisualDescr('The visual attributes as a |RegionVisual|.')

    def __init__(self, start, end, meta=None, visual=None):
        self.start = start
        self.end = end
        self.meta = meta or RegionMeta()
        self.visual = visual or RegionVisual()

    def contains(self, skycoord):
        # lines never contain anything
        in_reg = (False if skycoord.isscalar
                  else np.zeros(skycoord.shape, dtype=bool))

        if self.meta.get('include', True):
            # in_reg = False, always.  Lines do not include anything.
            return in_reg
        else:
            return np.logical_not(in_reg)

    @property
    def _line_center_sep(self):
        # Find center of line:
        sep = self.start.separation(self.end)
        pa = self.start.position_angle(self.end)
        cent = self.start.directional_offset_by(pa, sep / 2.)

        # Convert sep to a quantity for return:
        sep = sep.degree * u.deg

        return cent, sep

    @property
    def bounding_circle(self):
        cent, sep = self._line_center_sep

        return CircleSphericalSkyRegion(
            cent, sep / 2.
        )

    @property
    def bounding_lonlat(self):
        cent, _ = self._line_center_sep
        lons_arr, lats_arr = get_line_edge_raw_lonlat_bounds_circ_edges(
            SkyCoord([self.start, self.end]),
            cent
        )

        lons_arr, lats_arr = self._validate_lonlat_bounds(lons_arr, lats_arr)

        return lons_arr, lats_arr

    def transform_to(self, frame, merge_attributes=True):
        frame = self._validate_frame(frame)

        start_transf = self.start.transform_to(
            frame, merge_attributes=merge_attributes
        )
        end_transf = self.start.transform_to(
            frame, merge_attributes=merge_attributes
        )

        return LineSphericalSkyRegion(
            start_transf,
            end_transf,
            self.meta.copy(),
            self.visual.copy()
        )

    def discretize_boundary(self, n_points=100):  # pylint: disable=unused-argument
        """
        Discretize the boundary into a CompoundSphericalSkyRegion,
        containing multiple individual LineSphericalSkyRegion instances
        as the line segments.

        Parameters
        ----------
        n_points : int, optional
            Number of points along the line's boundary.

        Returns
        -------
        line: `~regions.CompoundSphericalSkyRegion`
            Spherical sky CompoundSphericalSkyRegion object,
            consisting of the union of multiple LineSphericalSkyRegion segments.
        """
        cent, _ = self._line_center_sep
        bound_verts = discretize_line_boundary(
            SkyCoord([self.start, self.end]),
            n_points
        )
        # Create a CompoundSphericalSkyRegion out of these segments:
        line = None
        for i in range(n_points - 1):
            lseg = LineSphericalSkyRegion(
                bound_verts[i], bound_verts[i + 1],
                meta=self.meta.copy(),
                visual=self.visual.copy(),
            )
            line = lseg if line is None else line | lseg

        return line

    def to_pixel(
        self,
        wcs=None,
        include_boundary_distortions=False,
        discretize_kwargs=None,
    ):
        if include_boundary_distortions:
            if discretize_kwargs is None:
                discretize_kwargs = {}

            if wcs is None:
                raise ValueError(
                    "'wcs' must be set if 'include_boundary_distortions'=True"
                )
            # Already implemented as part of CompoundSphericalSkyRegion
            # Don't pass boundary distortion / discretization kwargs to
            # CompoundSphericalSkyRegion.to_pixel, as these are already
            # considered in the explicit discretize_boundary.
            return self.discretize_boundary(**discretize_kwargs).to_pixel(
                wcs=wcs,
                include_boundary_distortions=False,
                discretize_kwargs=None,
            )

        return self.to_sky().to_pixel(wcs)

    def to_sky(
        self, wcs=None, include_boundary_distortions=False, discretize_kwargs=None
    ):
        if discretize_kwargs is None:
            discretize_kwargs = {}

        if include_boundary_distortions:
            if wcs is None:
                raise ValueError(
                    "'wcs' must be set if 'include_boundary_distortions'=True"
                )
            # Requires spherical to planar projection (from WCS) and discretization
            # Use to_pixel(), then apply "small angle approx" to get planar sky.
            return self.to_pixel(
                include_boundary_distortions=include_boundary_distortions,
                wcs=wcs,
                discretize_kwargs=discretize_kwargs,
            ).to_sky(wcs)

        return LineSkyRegion(
            self.start, self.end, meta=self.meta.copy(), visual=self.visual.copy()
        )
