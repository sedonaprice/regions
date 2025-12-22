# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines elliptical regions in both pixel and sky
coordinates.
"""

import math

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle

from regions._geometry import elliptical_overlap_grid
from regions._utils.spherical_helpers import (
    discretize_spherical_ellipse_boundary, spherical_ellipse_contains)
from regions._utils.wcs_helpers import pixel_scale_angle_at_skycoord
from regions.core.attributes import (PositiveScalar, PositiveScalarAngle,
                                     RegionMetaDescr, RegionVisualDescr,
                                     ScalarAngle, ScalarPixCoord,
                                     ScalarSkyCoord)
from regions.core.bounding_box import RegionBoundingBox
from regions.core.core import PixelRegion, SkyRegion, SphericalSkyRegion
from regions.core.mask import RegionMask
from regions.core.metadata import RegionMeta, RegionVisual
from regions.core.pixcoord import PixCoord

from .circle import CircleSphericalSkyRegion
from .polygon import PolygonPixelRegion, PolygonSphericalSkyRegion

__all__ = ['EllipsePixelRegion', 'EllipseSkyRegion', 'EllipseSphericalSkyRegion']


class EllipsePixelRegion(PixelRegion):
    """
    An ellipse in pixel coordinates.

    Parameters
    ----------
    center : `~regions.PixCoord`
        The position of the center of the ellipse.
    width : float
        The width of the ellipse (before rotation) in pixels.
    height : float
        The height of the ellipse (before rotation) in pixels.
    angle : `~astropy.units.Quantity`, optional
        The rotation angle of the ellipse, measured anti-clockwise. If
        set to zero (the default), the width axis is lined up with the x
        axis.
    meta : `~regions.RegionMeta` or `dict`, optional
        A dictionary that stores the meta attributes of the region.
    visual : `~regions.RegionVisual` or `dict`, optional
        A dictionary that stores the visual meta attributes of the
        region.

    Examples
    --------
    .. plot::
        :include-source:

        from astropy.coordinates import Angle
        from regions import PixCoord, EllipsePixelRegion
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)

        reg = EllipsePixelRegion(PixCoord(15, 10), width=16, height=10,
                                 angle=Angle(30, 'deg'))
        patch = reg.plot(ax=ax, facecolor='none', edgecolor='red', lw=2,
                         label='Ellipse')

        ax.legend(handles=(patch,), loc='upper center')
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 20)
        ax.set_aspect('equal')
    """

    _params = ('center', 'width', 'height', 'angle')
    _mpl_artist = 'Patch'
    center = ScalarPixCoord('The center pixel position as a |PixCoord|.')
    width = PositiveScalar('The width of the ellipse (before rotation) in '
                           'pixels as a float.')
    height = PositiveScalar('The height of the ellipse (before rotation) in '
                            'pixels as a float.')
    angle = ScalarAngle('The rotation angle measured anti-clockwise as a '
                        '|Quantity| angle.')
    meta = RegionMetaDescr('The meta attributes as a |RegionMeta|')
    visual = RegionVisualDescr('The visual attributes as a |RegionVisual|.')

    def __init__(self, center, width, height, angle=0. * u.deg, meta=None,
                 visual=None):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
        self.meta = meta or RegionMeta()
        self.visual = visual or RegionVisual()

    @property
    def area(self):
        return math.pi / 4 * self.width * self.height

    def contains(self, pixcoord):
        pixcoord = PixCoord._validate(pixcoord, name='pixcoord')
        cos_angle = np.cos(self.angle)
        sin_angle = np.sin(self.angle)
        dx = pixcoord.x - self.center.x
        dy = pixcoord.y - self.center.y
        in_ell = ((2 * (cos_angle * dx + sin_angle * dy) / self.width) ** 2
                  + (2 * (sin_angle * dx - cos_angle * dy)
                     / self.height) ** 2 <= 1.)
        if self.meta.get('include', True):
            return in_ell
        else:
            return np.logical_not(in_ell)

    def to_sky(self, wcs):
        center = wcs.pixel_to_world(self.center.x, self.center.y)
        _, pixscale, north_angle = pixel_scale_angle_at_skycoord(center, wcs)
        height = Angle(self.height * u.pix * pixscale, 'arcsec')
        width = Angle(self.width * u.pix * pixscale, 'arcsec')
        # region sky angles are defined relative to the WCS longitude axis;
        # photutils aperture sky angles are defined as the PA of the
        # semimajor axis (i.e., relative to the WCS latitude axis)
        angle = self.angle - (north_angle - 90 * u.deg)
        return EllipseSkyRegion(center, width, height, angle=angle,
                                meta=self.meta.copy(),
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

            # ### Potential solution:
            # # Leverage polygon class to_spherical_sky() functionality without
            # # distortions, as the distortions were already computed in creating
            # # that polygon approximation
            # return self.discretize_boundary(**discretize_kwargs).to_spherical_sky(
            #     wcs=wcs, include_boundary_distortions=False
            # )

        return self.to_sky(wcs).to_spherical_sky()

    @property
    def bounding_box(self):
        """
        The minimal bounding box (`~regions.RegionBoundingBox`)
        enclosing the exact elliptical region.
        """
        # We use the solution described in
        # https://stackoverflow.com/a/14163413
        cos_theta = np.cos(self.angle)
        sin_theta = np.sin(self.angle)
        width_x = 0.5 * self.width * cos_theta
        width_y = 0.5 * self.width * sin_theta
        height_x = 0.5 * self.height * -sin_theta
        height_y = 0.5 * self.height * cos_theta
        dx = np.sqrt(width_x**2 + height_x**2)
        dy = np.sqrt(width_y**2 + height_y**2)

        xmin = self.center.x - dx
        xmax = self.center.x + dx
        ymin = self.center.y - dy
        ymax = self.center.y + dy

        return RegionBoundingBox.from_float(xmin, xmax, ymin, ymax)

    def to_mask(self, mode='center', subpixels=5):
        # NOTE: assumes this class represents a single circle

        self._validate_mode(mode, subpixels)

        if mode == 'center':
            mode = 'subpixels'
            subpixels = 1

        # Find bounding box and mask size
        bbox = self.bounding_box
        ny, nx = bbox.shape

        # Find position of pixel edges and recenter so that ellipse is
        # at origin
        xmin = float(bbox.ixmin) - 0.5 - self.center.x
        xmax = float(bbox.ixmax) - 0.5 - self.center.x
        ymin = float(bbox.iymin) - 0.5 - self.center.y
        ymax = float(bbox.iymax) - 0.5 - self.center.y

        use_exact = 0 if mode == 'subpixels' else 1

        fraction = elliptical_overlap_grid(xmin, xmax, ymin, ymax, nx, ny,
                                           0.5 * self.width, 0.5 * self.height,
                                           self.angle.to(u.rad).value,
                                           use_exact, subpixels)

        return RegionMask(fraction, bbox=bbox)

    def as_artist(self, origin=(0, 0), **kwargs):
        """
        Return a matplotlib patch object for the region
        (`matplotlib.patches.Ellipse`).

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` pixel position of the origin of the displayed
            image.

        **kwargs : dict
            Any keyword arguments accepted by
            `~matplotlib.patches.Ellipse`. These keywords will override
            any visual meta attributes of this region.

        Returns
        -------
        artist : `~matplotlib.patches.Ellipse`
            A matplotlib ellipse patch.
        """
        from matplotlib.patches import Ellipse

        xy = self.center.x - origin[0], self.center.y - origin[1]
        width = self.width
        height = self.height
        # matplotlib expects rotation in degrees (anti-clockwise)
        angle = self.angle.to('deg').value

        mpl_kwargs = self.visual.define_mpl_kwargs(self._mpl_artist)
        mpl_kwargs.update(kwargs)

        return Ellipse(xy=xy, width=width, height=height, angle=angle,
                       **mpl_kwargs)

    def _update_from_mpl_selector(self, *args, **kwargs):
        xmin, xmax, ymin, ymax = self._mpl_selector.extents
        self.center = PixCoord(x=0.5 * (xmin + xmax),
                               y=0.5 * (ymin + ymax))
        self.width = (xmax - xmin)
        self.height = (ymax - ymin)
        self.angle = 0. * u.deg
        if self._mpl_selector_callback is not None:
            self._mpl_selector_callback(self)

    def as_mpl_selector(self, ax, active=True, sync=True, callback=None,
                        drag_from_anywhere=False, **kwargs):
        """
        Return a matplotlib editable widget for this region
        (`matplotlib.widgets.EllipseSelector`).

        Parameters
        ----------
        ax : `~matplotlib.axes.Axes`
            The matplotlib axes to add the selector to.
        active : bool, optional
            Whether the selector should be active by default.
        sync : bool, optional
            If `True` (the default), the region will be kept in
            sync with the selector. Otherwise, the selector will be
            initialized with the values from the region but the two will
            then be disconnected.
        callback : callable, optional
            If specified, this function will be called every time the
            region is updated. This only has an effect if ``sync`` is
            `True`. If a callback is set, it is called for the first
            time once the selector has been created.
        drag_from_anywhere : bool, optional
            If `True`, the selector can be moved by clicking anywhere within
            its bounds, else only at the central anchor
            (only available with matplotlib 3.5 upwards; default: `False`).
        **kwargs : dict
            Additional keyword arguments that are passed to
            `matplotlib.widgets.EllipseSelector`.

        Returns
        -------
        selector : `matplotlib.widgets.EllipseSelector`
            The matplotlib selector.

        Notes
        -----
        Once a selector has been created, you will need to keep a
        reference to it until you no longer need it. In addition,
        you can enable/disable the selector at any point by calling
        ``selector.set_active(True)`` or ``selector.set_active(False)``.
        """
        from matplotlib.widgets import EllipseSelector

        if hasattr(self, '_mpl_selector'):
            raise AttributeError('Cannot attach more than one selector to a region.')

        if self.angle.value != 0:
            raise NotImplementedError('Cannot create matplotlib selector for rotated ellipse.')

        if sync:
            sync_callback = self._update_from_mpl_selector
        else:
            def sync_callback(*args, **kwargs):
                pass

        rectprops = {'edgecolor': self.visual.get('color', 'black'),
                     'facecolor': 'none',
                     'linewidth': self.visual.get('linewidth', 1),
                     'linestyle': self.visual.get('linestyle', 'solid')}
        rectprops.update(kwargs.pop('props', dict()))
        kwargs.update({'props': rectprops})

        self._mpl_selector = EllipseSelector(
            ax, sync_callback, interactive=True,
            drag_from_anywhere=drag_from_anywhere, **kwargs)

        self._mpl_selector.extents = (self.center.x - self.width / 2,
                                      self.center.x + self.width / 2,
                                      self.center.y - self.height / 2,
                                      self.center.y + self.height / 2)
        self._mpl_selector.set_active(active)
        self._mpl_selector_callback = callback

        if sync and self._mpl_selector_callback is not None:
            self._mpl_selector_callback(self)

        return self._mpl_selector

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
        region : `EllipsePixelRegion`
            The rotated region (which is an independent copy).
        """
        center = self.center.rotate(center, angle)
        angle = self.angle + angle
        return self.copy(center=center, angle=angle)


class EllipseSkyRegion(SkyRegion):
    """
    An ellipse defined using sky coordinates.

    Parameters
    ----------
    center : `~astropy.coordinates.SkyCoord`
        The position of the center of the ellipse.
    width : `~astropy.units.Quantity`
        The width of the ellipse (before rotation) as an angle.
    height : `~astropy.units.Quantity`
        The height of the ellipse (before rotation) as an angle.
    angle : `~astropy.units.Quantity`, optional
        The rotation angle of the ellipse, measured anti-clockwise. If
        set to zero (the default), the width axis is lined up with the
        longitude axis of the celestial coordinates.
    meta : `~regions.RegionMeta` or `dict`, optional
        A dictionary that stores the meta attributes of the region.
    visual : `~regions.RegionVisual` or `dict`, optional
        A dictionary that stores the visual meta attributes of the
        region.
    """

    _params = ('center', 'width', 'height', 'angle')
    center = ScalarSkyCoord('The center position as a |SkyCoord|.')
    width = PositiveScalarAngle('The width of the ellipse (before rotation) '
                                'as a |Quantity| angle.')
    height = PositiveScalarAngle('The height of the ellipse (before rotation) '
                                 'as a |Quantity| angle.')
    angle = ScalarAngle('The rotation angle measured anti-clockwise as a '
                        '|Quantity| angle.')
    meta = RegionMetaDescr('The meta attributes as a |RegionMeta|')
    visual = RegionVisualDescr('The visual attributes as a |RegionVisual|.')

    def __init__(self, center, width, height, angle=0. * u.deg, meta=None,
                 visual=None):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
        self.meta = meta or RegionMeta()
        self.visual = visual or RegionVisual()

    def to_pixel(self, wcs):
        center, pixscale, north_angle = pixel_scale_angle_at_skycoord(
            self.center, wcs)
        height = (self.height / pixscale).to(u.pixel).value
        width = (self.width / pixscale).to(u.pixel).value
        # region sky angles are defined relative to the WCS longitude axis;
        # photutils aperture sky angles are defined as the PA of the
        # semimajor axis (i.e., relative to the WCS latitude axis)
        angle = self.angle + (north_angle - 90 * u.deg)
        return EllipsePixelRegion(center, width, height, angle=angle,
                                  meta=self.meta.copy(),
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

            # ### Potential solution:
            # # Leverage polygon class to_spherical_sky() functionality without
            # # distortions, as the distortions were already computed in creating
            # # that polygon approximation
            # return self.to_pixel(wcs).discretize_boundary(**discretize_kwargs).to_spherical_sky(
            #     wcs=wcs, include_boundary_distortions=False
            # )

        return EllipseSphericalSkyRegion(
            self.center.copy(), self.width.copy(), self.height.copy(),
            meta=self.meta.copy(), visual=self.visual.copy()
        )


class EllipseSphericalSkyRegion(SphericalSkyRegion):
    """
    An ellipse defined using spherical sky coordinates.

    Parameters
    ----------
    center : `~astropy.coordinates.SkyCoord`
        The position of the center of the ellipse.
    width : `~astropy.units.Quantity`
        The width of the ellipse (before rotation) as an angle.
    height : `~astropy.units.Quantity`
        The height of the ellipse (before rotation) as an angle.
    angle : `~astropy.units.Quantity`, optional
        The rotation angle of the ellipse, measured anti-clockwise. If
        set to zero (the default), the width axis is lined up with the
        longitude axis of the celestial coordinates.
    meta : `~regions.RegionMeta` or `dict`, optional
        A dictionary that stores the meta attributes of the region.
    visual : `~regions.RegionVisual` or `dict`, optional
        A dictionary that stores the visual meta attributes of the
        region.
    """

    _params = ('center', 'width', 'height', 'angle')
    center = ScalarSkyCoord('The center position as a |SkyCoord|.')
    width = PositiveScalarAngle('The width of the ellipse (before rotation) '
                                'as a |Quantity| angle.')
    height = PositiveScalarAngle('The height of the ellipse (before rotation) '
                                 'as a |Quantity| angle.')
    angle = ScalarAngle('The rotation angle measured anti-clockwise as a '
                        '|Quantity| angle.')
    meta = RegionMetaDescr('The meta attributes as a |RegionMeta|')
    visual = RegionVisualDescr('The visual attributes as a |RegionVisual|.')

    def __init__(self, center, width, height, angle=0. * u.deg, meta=None,
                 visual=None):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
        self.meta = meta or RegionMeta()
        self.visual = visual or RegionVisual()

        self._is_inner_annulus = False

    def _get_sph_angle_transf(
        self, center_transf, frame, merge_attributes=True
    ):
        # To determine angle transformation:
        # Put a test point on the ellipse at angle, width/2
        # (eg, on the "width" axis of the ellipse),
        # transform that point, and then determine the angle
        # in the new coord frame

        # SkyCoord.directional_offset_by uses PA defined
        # as angle E of N, so offset ellipse angle by -90 deg
        width_point = self.center.directional_offset_by(
            self.angle - 90 * u.deg, self.width / 2.
        )
        width_point_transf = width_point.transform_to(
            frame, merge_attributes=merge_attributes
        )

        # SkyCoord.position_angle returns angle E of N,
        # so offset by +90 deg to get ellipse angle definition
        angle_transf = center_transf.position_angle(
            width_point_transf
        ) + 90 * u.deg

        return angle_transf

    def contains(self, coord):
        in_ell = spherical_ellipse_contains(
            coord, self.center, self.width, self.height, self.angle
        )
        if self.meta.get('include', True):
            return in_ell
        else:
            return np.logical_not(in_ell)

    @property
    def bounding_circle(self):
        radius = np.max(
            [self.width.value, self.height.to(self.width.unit).value]
        ) * self.width.unit
        return CircleSphericalSkyRegion(
            center=self.center.copy(),
            radius=radius
        )

    def _get_padded_polygon_approximation(self, fac=1):
        # Enable dilation or contraction with fac
        padfac = 0.01
        padmax = 4 * u.arcsec
        maxdim = np.max(
            [self.width.value, self.height.to(self.width.unit).value]
        ) * self.width.unit
        if maxdim > padmax / padfac:
            pad_width = pad_height = padmax
            n_points = 1000
        else:
            pad_width = self.width * padfac
            pad_height = self.height * padfac
            n_points = 100

        ell_pad = EllipseSphericalSkyRegion(
            self.center.copy(),
            self.width.copy() + fac * pad_width,
            self.height.copy() + fac * pad_height,
            angle=self.angle.copy()
        )
        return ell_pad.discretize_boundary(n_points=n_points)

    @property
    def bounding_lonlat(self):
        # Do an approximation, as fully general computation
        # would be very complex to work out the spherical geometry
        # and tangent points for:
        # Increase width, height by a few arcsec / 1% if small?
        # Discretize that padded ellipse boundary
        # Get bounding lonlat of that polygon and return
        fac = -1 if self._is_inner_annulus else 1

        poly_pad = self._get_padded_polygon_approximation(fac=fac)

        return poly_pad.bounding_lonlat

    def transform_to(self, frame, merge_attributes=True):
        # Assuming a spherical -> spherical frame transformation:
        frame = self._validate_frame(frame)

        center_transf = self.center.transform_to(
            frame, merge_attributes=merge_attributes
        )

        angle_transf = self._get_sph_angle_transf(
            center_transf, frame, merge_attributes=merge_attributes
        )

        return EllipseSphericalSkyRegion(
            center_transf,
            self.width.copy(),
            self.height.copy(),
            angle_transf,
            self.meta.copy(),
            self.visual.copy()
        )

    def discretize_boundary(self, n_points=100):
        bound_verts = discretize_spherical_ellipse_boundary(
            self.center, self.width, self.height, self.angle,
            n_points
        )
        return PolygonSphericalSkyRegion(bound_verts)

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
            # Requires spherical to planar projection (from WCS) and discretization
            verts = wcs.world_to_pixel(
                self.discretize_boundary(**discretize_kwargs).vertices
            )
            return PolygonPixelRegion(
                PixCoord(*verts), meta=self.meta.copy(), visual=self.visual.copy()
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

        return EllipseSkyRegion(
            self.center, self.width, self.height, self.angle,
            meta=self.meta.copy(), visual=self.visual.copy()
        )
