# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines point regions in both pixel and sky coordinates.
"""

import numpy as np

from regions.core.attributes import (RegionMetaDescr, RegionVisualDescr,
                                     ScalarPixCoord, ScalarSkyCoord)
from regions.core.bounding_box import RegionBoundingBox
from regions.core.core import PixelRegion, SkyRegion, SphericalSkyRegion
from regions.core.metadata import RegionMeta, RegionVisual
from regions.core.pixcoord import PixCoord

__all__ = ['PointPixelRegion', 'PointSkyRegion', 'PointSphericalSkyRegion']


class PointPixelRegion(PixelRegion):
    """
    A point position in pixel coordinates.

    Parameters
    ----------
    center : `~regions.PixCoord`
        The position of the point.
    meta : `~regions.RegionMeta` or `dict`, optional
        A dictionary that stores the meta attributes of the region.
    visual : `~regions.RegionVisual` or `dict`, optional
        A dictionary that stores the visual meta attributes of the
        region.

    Examples
    --------
    .. plot::
        :include-source:

        from regions import PixCoord, PointPixelRegion, RegionVisual
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)

        regs = []
        regs.append(PointPixelRegion(PixCoord(2, 2),
                    visual=RegionVisual(marker='D')))
        regs.append(PointPixelRegion(PixCoord(2, 3),
                    visual=RegionVisual(marker='+')))
        regs.append(PointPixelRegion(PixCoord(3, 3),
                    visual=RegionVisual(marker='^')))
        regs.append(PointPixelRegion(PixCoord(3, 2),
                    visual=RegionVisual(marker='*')))
        regs.append(PointPixelRegion(PixCoord(2, 4),
                    visual=RegionVisual(marker='x')))
        regs.append(PointPixelRegion(PixCoord(4, 2)))
        for reg in regs:
            reg.plot(ax=ax)

        ax.set_xlim(0, 6)
        ax.set_ylim(0, 6)
        ax.set_aspect('equal')
    """

    _params = ('center',)
    _mpl_artist = 'Line2D'
    center = ScalarPixCoord('The point pixel position as a |PixCoord|.')
    meta = RegionMetaDescr('The meta attributes as a |RegionMeta|')
    visual = RegionVisualDescr('The visual attributes as a |RegionVisual|.')

    def __init__(self, center, meta=None, visual=None):
        self.center = center
        self.meta = meta or RegionMeta()
        self.visual = visual or RegionVisual()

    @property
    def area(self):
        return 0.0

    def contains(self, pixcoord):
        in_reg = (False if pixcoord.isscalar
                  else np.zeros(pixcoord.x.shape, dtype=bool))

        if self.meta.get('include', True):
            # in_reg = False, always.  Points do not include anything.
            return in_reg
        else:
            return np.logical_not(in_reg)

    def to_sky(self, wcs):
        center = wcs.pixel_to_world(self.center.x, self.center.y)
        return PointSkyRegion(center, meta=self.meta.copy(),
                              visual=self.visual.copy())

    def to_spherical_sky(self, wcs=None, include_boundary_distortions=False,
                         discretize_kwargs=None):
        if wcs is None:
            raise ValueError(
                "'wcs' must be set for transformations from pixel to spherical sky coordinates"
            )
        center = wcs.pixel_to_world(self.center.x, self.center.y)

        # Technically this portion is not required for a point,
        # but retaining it for consistency of behavior (requiring WCS
        # for boundary distortion-accounting transforms) compared to
        # other classes:
        if include_boundary_distortions:
            if discretize_kwargs is None:
                discretize_kwargs = {}

            return PointSphericalSkyRegion(
                center, meta=self.meta.copy(),
                visual=self.visual.copy()
            )
        return PointSphericalSkyRegion(
            center, meta=self.meta.copy(),
            visual=self.visual.copy()
        )

    @property
    def bounding_box(self):
        return RegionBoundingBox.from_float(self.center.x, self.center.x,
                                            self.center.y, self.center.y)

    def to_mask(self, mode='center', subpixels=5):
        # TODO: needs to be implemented
        raise NotImplementedError

    def as_artist(self, origin=(0, 0), **kwargs):
        """
        Return a matplotlib Line2D object for this region
        (`matplotlib.lines.Line2D`).

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` pixel position of the origin of the displayed
            image.

        **kwargs : `dict`
            Any keyword arguments accepted by
            `~matplotlib.lines.Line2D`. These keywords will override any
            visual meta attributes of this region.

        Returns
        -------
        artist : `~matplotlib.lines.Line2D`
            A matplotlib Line2D object.
        """
        from matplotlib.lines import Line2D

        mpl_kwargs = self.visual.define_mpl_kwargs(self._mpl_artist)
        mpl_kwargs.update(kwargs)

        return Line2D([self.center.x - origin[0]],
                      [self.center.y - origin[1]], **mpl_kwargs)

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
        region : `PointPixelRegion`
            The rotated region (which is an independent copy).
        """
        center = self.center.rotate(center, angle)
        return self.copy(center=center)


class PointSkyRegion(SkyRegion):
    """
    A pixel region in sky coordinates.

    Parameters
    ----------
    center : `~astropy.coordinates.SkyCoord`
        The position of the point.
    meta : `~regions.RegionMeta` or `dict`, optional
        A dictionary that stores the meta attributes of the region.
    visual : `~regions.RegionVisual` or `dict`, optional
        A dictionary that stores the visual meta attributes of the
        region.
    """

    _params = ('center',)
    center = ScalarSkyCoord('The point position as a |SkyCoord|.')
    meta = RegionMetaDescr('The meta attributes as a |RegionMeta|')
    visual = RegionVisualDescr('The visual attributes as a |RegionVisual|.')

    def __init__(self, center, meta=None, visual=None):
        self.center = center
        self.meta = meta or RegionMeta()
        self.visual = visual or RegionVisual()

    def contains(self, skycoord, wcs):  # pylint: disable=unused-argument
        # points never include anything
        in_reg = (False if skycoord.isscalar
                  else np.zeros(skycoord.shape, dtype=bool))

        if self.meta.get('include', True):
            # in_reg = False, always.  Points do not include anything.
            return in_reg
        else:
            return np.logical_not(in_reg)

    def to_pixel(self, wcs):
        center_x, center_y = wcs.world_to_pixel(self.center)
        center = PixCoord(center_x, center_y)
        return PointPixelRegion(center, meta=self.meta.copy(),
                                visual=self.visual.copy())

    def to_spherical_sky(self, wcs=None, include_boundary_distortions=False,
                         discretize_kwargs=None):
        # Technically this portion is not required for a point,
        # but retaining it for consistency of behavior (requiring WCS
        # for boundary distortion-accounting transforms) compared to
        # other classes:
        if include_boundary_distortions:
            if discretize_kwargs is None:
                discretize_kwargs = {}

            if wcs is None:
                raise ValueError(
                    "'wcs' must be set if 'include_boundary_distortions'=True"
                )
            return PointSphericalSkyRegion(
                self.center, meta=self.meta.copy(), visual=self.visual.copy()
            )

        return PointSphericalSkyRegion(
            self.center, meta=self.meta.copy(), visual=self.visual.copy()
        )


class PointSphericalSkyRegion(SphericalSkyRegion):
    """
    A point region in spherical sky coordinates.

    Parameters
    ----------
    center : `~astropy.coordinates.SkyCoord`
        The position of the point.
    meta : `~regions.RegionMeta` or `dict`, optional
        A dictionary that stores the meta attributes of the region.
    visual : `~regions.RegionVisual` or `dict`, optional
        A dictionary that stores the visual meta attributes of the
        region.
    """

    _params = ('center',)
    center = ScalarSkyCoord('The point position as a |SkyCoord|.')
    meta = RegionMetaDescr('The meta attributes as a |RegionMeta|')
    visual = RegionVisualDescr('The visual attributes as a |RegionVisual|.')

    def __init__(self, center, meta=None, visual=None):
        self.center = center
        self.meta = meta or RegionMeta()
        self.visual = visual or RegionVisual()

    def contains(self, skycoord):  # pylint: disable=unused-argument
        # points never include anything
        in_reg = (False if skycoord.isscalar
                  else np.zeros(skycoord.shape, dtype=bool))

        if self.meta.get('include', True):
            # in_reg = False, always.  Points do not include anything.
            return in_reg
        else:
            return np.logical_not(in_reg)

    @property
    def bounding_circle(self):
        # Two options:
        # Either return an error when calling this,
        # Or for consistency with other shapes,
        # return a radius 0 circle:
        # But circles can't have radius = 0.
        # So raise error
        raise ValueError(
            'Invalid property: Points cannot have a bounding circle,\n'
            'as spherical circles must have radii > 0'
        )

    @property
    def bounding_lonlat(self):
        # Two options:
        # Either return an error when calling this,
        # Or for consistency with other shapes,
        # return a "meaningless" span.
        # But for consistency with bounding circle,
        # raise an error
        raise ValueError(
            'Invalid property: Points cannot have a bounding lon/lat,\n'
            'as they do not have non-zero size.'
        )

    def transform_to(self, frame, merge_attributes=True):
        frame = self._validate_frame(frame)

        center_transf = self.center.transform_to(
            frame, merge_attributes=merge_attributes
        )

        return PointSphericalSkyRegion(
            center_transf,
            self.meta.copy(),
            self.visual.copy()
        )

    def discretize_boundary(self, n_points=100):  # pylint: disable=unused-argument
        # Two options:
        # Either return an error when calling this,
        # But for consistency with other shapes,
        # return just itself as it's already discrete.
        return self.copy()

    def to_pixel(
        self,
        wcs=None,
        include_boundary_distortions=False,
        discretize_kwargs=None,
    ):
        if include_boundary_distortions:
            if discretize_kwargs is None:
                discretize_kwargs = {}

            # Technically this portion is not required for a point,
            # but retaining it for consistency of behavior (requiring WCS
            # for boundary distortion-accounting transforms) compared to
            # other classes:
            if wcs is None:
                raise ValueError(
                    "'wcs' must be set if 'include_boundary_distortions'=True"
                )
            # Convert in the same way as sky -> pixel
            center_x, center_y = wcs.world_to_pixel(self.center)
            center = PixCoord(center_x, center_y)
            return PointPixelRegion(center, meta=self.meta.copy(),
                                    visual=self.visual.copy())

        return self.to_sky().to_pixel(wcs)

    def to_sky(
        self, wcs=None, include_boundary_distortions=False, discretize_kwargs=None
    ):
        if discretize_kwargs is None:
            discretize_kwargs = {}

        # Technically this portion is not required for a point,
        # but retaining it for consistency of behavior (requiring WCS
        # for boundary distortion-accounting transforms) compared to
        # other classes:
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

        return PointSkyRegion(
            self.center, meta=self.meta.copy(), visual=self.visual.copy()
        )
