# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines rectangular regions in both pixel and sky
coordinates.
"""
import operator

import astropy.units as u
import numpy as np
from astropy.coordinates import Angle, SkyCoord

from regions._geometry import rectangular_overlap_grid
from regions._utils.spherical_helpers import (
    cross_product_skycoord2skycoord, discretize_all_edge_boundaries,
    get_edge_raw_lonlat_bounds_circ_edges)
from regions._utils.wcs_helpers import pixel_scale_angle_at_skycoord
from regions.core.attributes import (PositiveScalar, PositiveScalarAngle,
                                     RegionMetaDescr, RegionVisualDescr,
                                     ScalarAngle, ScalarPixCoord,
                                     ScalarSkyCoord)
from regions.core.bounding_box import RegionBoundingBox
from regions.core.compound import CompoundSphericalSkyRegion
from regions.core.core import PixelRegion, SkyRegion, SphericalSkyRegion
from regions.core.mask import RegionMask
from regions.core.metadata import RegionMeta, RegionVisual
from regions.core.pixcoord import PixCoord
from regions.shapes.circle import CircleSphericalSkyRegion
from regions.shapes.polygon import (PolygonPixelRegion,
                                    PolygonSphericalSkyRegion)

__all__ = ['RectanglePixelRegion', 'RectangleSkyRegion', 'RectangleSphericalSkyRegion']


class RectanglePixelRegion(PixelRegion):
    """
    A rectangle in pixel coordinates.

    Parameters
    ----------
    center : `~regions.PixCoord`
        The position of the center of the rectangle.
    width : float
        The width of the rectangle (before rotation) in pixels.
    height : float
        The height of the rectangle (before rotation) in pixels.
    angle : `~astropy.units.Quantity`, optional
        The rotation angle of the rectangle, measured anti-clockwise. If
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
        from regions import PixCoord, RectanglePixelRegion
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1)

        reg = RectanglePixelRegion(PixCoord(x=15, y=10), width=8,
                                   height=5, angle=Angle(30, 'deg'))
        patch = reg.plot(ax=ax, facecolor='none', edgecolor='red', lw=2,
                         label='Rectangle')

        ax.legend(handles=(patch,), loc='upper center')
        ax.set_xlim(0, 30)
        ax.set_ylim(0, 20)
        ax.set_aspect('equal')
    """

    _params = ('center', 'width', 'height', 'angle')
    _mpl_artist = 'Patch'
    center = ScalarPixCoord('The center pixel position as a |PixCoord|.')
    width = PositiveScalar('The width of the rectangle (before rotation) in '
                           'pixels as a float.')
    height = PositiveScalar('The height of the rectangle (before rotation) '
                            'in pixels as a float.')
    angle = ScalarAngle('The rotation angle measured anti-clockwise as a '
                        '|Quantity| angle.')
    meta = RegionMetaDescr('The meta attributes as a |RegionMeta|')
    visual = RegionVisualDescr('The visual attributes as a |RegionVisual|.')

    def __init__(self, center, width, height, angle=0 * u.deg, meta=None,
                 visual=None):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
        self.meta = meta or RegionMeta()
        self.visual = visual or RegionVisual()

    @property
    def area(self):
        return self.width * self.height

    def contains(self, pixcoord):
        cos_angle = np.cos(self.angle)
        sin_angle = np.sin(self.angle)
        dx = pixcoord.x - self.center.x
        dy = pixcoord.y - self.center.y
        dx_rot = cos_angle * dx + sin_angle * dy
        dy_rot = sin_angle * dx - cos_angle * dy
        in_rect = ((np.abs(dx_rot) < self.width * 0.5)
                   & (np.abs(dy_rot) < self.height * 0.5))
        if self.meta.get('include', True):
            return in_rect
        else:
            return np.logical_not(in_rect)

    def discretize_boundary(self, n_points=10):
        """
        Discretize the boundary into a PolygonPixelRegion.

        Parameters
        ----------
        n_points : int, optional
            Number of points along the each edge's boundary.

        Returns
        -------
        poly_pix_region: `~regions.PolygonPixelRegion`
            Planar PolygonPixelRegion object, with vertices in clockwise order.
        """
        t = np.linspace(0, 1, num=n_points, endpoint=False)

        # The four rectangle vertices or corners
        x, y = self.corners.T
        # Need to invert order because of CW convention:
        vertices = PixCoord(x=x[::-1], y=y[::-1])

        all_edge_bound_verts = None
        for i in range(len(vertices)):
            # Endpoints of one edge: vertices[i-1], vertices[i]

            xs = vertices[i - 1].x + t * (
                vertices[i].x - vertices[i - 1].x
            )
            ys = vertices[i - 1].y + t * (
                vertices[i].y - vertices[i - 1].y
            )
            bound_verts = PixCoord(xs, ys)

            if all_edge_bound_verts is None:
                all_edge_bound_verts = bound_verts
            else:
                all_edge_bound_verts = PixCoord(
                    np.concatenate(
                        [all_edge_bound_verts.x, bound_verts.x]
                    ),
                    np.concatenate(
                        [all_edge_bound_verts.y, bound_verts.y]
                    ),
                )

        return PolygonPixelRegion(
            all_edge_bound_verts,
            meta=self.meta.copy(),
            visual=self.visual.copy()
        )

    def to_sky(self, wcs):
        center = wcs.pixel_to_world(self.center.x, self.center.y)
        _, pixscale, north_angle = pixel_scale_angle_at_skycoord(center, wcs)
        width = Angle(self.width * u.pix * pixscale, 'arcsec')
        height = Angle(self.height * u.pix * pixscale, 'arcsec')
        # region sky angles are defined relative to the WCS longitude axis;
        # photutils aperture sky angles are defined as the PA of the
        # semimajor axis (i.e., relative to the WCS latitude axis)
        angle = self.angle - (north_angle - 90 * u.deg)
        return RectangleSkyRegion(center, width, height, angle=angle,
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

        return self.to_sky(wcs).to_spherical_sky()

    @property
    def bounding_box(self):
        w2 = self.width / 2.
        h2 = self.height / 2.
        cos_angle = np.cos(self.angle)  # self.angle is a Quantity
        sin_angle = np.sin(self.angle)  # self.angle is a Quantity
        dx1 = abs(w2 * cos_angle - h2 * sin_angle)
        dy1 = abs(w2 * sin_angle + h2 * cos_angle)
        dx2 = abs(w2 * cos_angle + h2 * sin_angle)
        dy2 = abs(w2 * sin_angle - h2 * cos_angle)
        dx = max(dx1, dx2)
        dy = max(dy1, dy2)

        xmin = self.center.x - dx
        xmax = self.center.x + dx
        ymin = self.center.y - dy
        ymax = self.center.y + dy

        return RegionBoundingBox.from_float(xmin, xmax, ymin, ymax)

    def to_mask(self, mode='center', subpixels=5):
        # NOTE: assumes this class represents a single rectangle

        self._validate_mode(mode, subpixels)

        if mode == 'center':
            mode = 'subpixels'
            subpixels = 1

        # Find bounding box and mask size
        bbox = self.bounding_box
        ny, nx = bbox.shape

        # Find position of pixel edges and recenter so that circle is at
        # origin
        xmin = float(bbox.ixmin) - 0.5 - self.center.x
        xmax = float(bbox.ixmax) - 0.5 - self.center.x
        ymin = float(bbox.iymin) - 0.5 - self.center.y
        ymax = float(bbox.iymax) - 0.5 - self.center.y

        use_exact = 0 if mode == 'subpixels' else 1

        fraction = rectangular_overlap_grid(xmin, xmax, ymin, ymax, nx, ny,
                                            self.width, self.height,
                                            self.angle.to(u.rad).value,
                                            use_exact, subpixels)

        return RegionMask(fraction, bbox=bbox)

    def as_artist(self, origin=(0, 0), **kwargs):
        """
        Return a matplotlib patch object for this region
        (`matplotlib.patches.Rectangle`).

        Parameters
        ----------
        origin : array_like, optional
            The ``(x, y)`` pixel position of the origin of the displayed
            image.

        **kwargs : dict
            Any keyword arguments accepted by
            `~matplotlib.patches.Rectangle`. These keywords will
            override any visual meta attributes of this region.

        Returns
        -------
        artist : `~matplotlib.patches.Rectangle`
            A matplotlib rectangle patch.
        """
        from matplotlib.patches import Rectangle

        xy = self._lower_left_xy()
        xy = xy[0] - origin[0], xy[1] - origin[1]
        width = self.width
        height = self.height
        # matplotlib expects rotation in degrees (anti-clockwise)
        angle = self.angle.to('deg').value

        mpl_kwargs = self.visual.define_mpl_kwargs(self._mpl_artist)
        mpl_kwargs.update(kwargs)

        return Rectangle(xy=xy, width=width, height=height,
                         angle=angle, **mpl_kwargs)

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
        Return a matplotlib editable widget for the region
        (`matplotlib.widgets.RectangleSelector`).

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
            Additional keyword arguments are passed to
            `matplotlib.widgets.RectangleSelector`.

        Returns
        -------
        selector : `matplotlib.widgets.RectangleSelector`
            The matplotlib selector.

        Notes
        -----
        Once a selector has been created, you will need to keep a
        reference to it until you no longer need it. In addition,
        you can enable/disable the selector at any point by calling
        ``selector.set_active(True)`` or ``selector.set_active(False)``.
        """
        from matplotlib.widgets import RectangleSelector

        if hasattr(self, '_mpl_selector'):
            raise AttributeError('Cannot attach more than one selector to a region.')

        if self.angle.value != 0:
            raise NotImplementedError('Cannot create matplotlib selector for '
                                      'rotated rectangle.')

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

        self._mpl_selector = RectangleSelector(
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

    @property
    def corners(self):
        """
        Return the x, y coordinate pairs that define the corners.
        """
        corners = [(-self.width / 2, -self.height / 2),
                   (self.width / 2, -self.height / 2),
                   (self.width / 2, self.height / 2),
                   (-self.width / 2, self.height / 2),
                   ]
        rotmat = [[np.cos(self.angle), np.sin(self.angle)],
                  [-np.sin(self.angle), np.cos(self.angle)]]

        return np.dot(corners, rotmat) + np.array([self.center.x,
                                                   self.center.y])

    def to_polygon(self):
        """
        Return a 4-sided polygon equivalent to this rectangle.
        """
        x, y = self.corners.T
        vertices = PixCoord(x=x, y=y)
        return PolygonPixelRegion(vertices=vertices, meta=self.meta.copy(),
                                  visual=self.visual.copy())

    def _lower_left_xy(self):
        """
        Compute lower left ``xy`` pixel position.

        This is used for the conversion to matplotlib in ``as_artist``.

        Taken from
        http://photutils.readthedocs.io/en/latest/_modules/photutils/ape
        rture/rectangle.html#RectangularAperture.plot.
        """
        hw = self.width / 2.
        hh = self.height / 2.
        sint = np.sin(self.angle)
        cost = np.cos(self.angle)
        dx = (hh * sint) - (hw * cost)
        dy = -(hh * cost) - (hw * sint)
        x = self.center.x + dx
        y = self.center.y + dy
        return x, y

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
        region : `RectanglePixelRegion`
            The rotated region (which is an independent copy).
        """
        center = self.center.rotate(center, angle)
        angle = self.angle + angle
        return self.copy(center=center, angle=angle)


class RectangleSkyRegion(SkyRegion):
    """
    A rectangle in sky coordinates.

    Parameters
    ----------
    center : `~astropy.coordinates.SkyCoord`
        The position of the center of the rectangle.
    width : `~astropy.units.Quantity`
        The width of the rectangle (before rotation) as an angle.
    height : `~astropy.units.Quantity`
        The height of the rectangle (before rotation) as an angle.
    angle : `~astropy.units.Quantity`, optional
        The rotation angle of the rectangle, measured anti-clockwise. If
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
    width = PositiveScalarAngle('The width of the rectangle (before rotation) '
                                'as a |Quantity| angle.')
    height = PositiveScalarAngle('The height of the rectangle (before '
                                 'rotation) as a |Quantity| angle.')
    angle = ScalarAngle('The rotation angle measured anti-clockwise as a '
                        '|Quantity| angle.')
    meta = RegionMetaDescr('The meta attributes as a |RegionMeta|')
    visual = RegionVisualDescr('The visual attributes as a |RegionVisual|.')

    def __init__(self, center, width, height, angle=0 * u.deg, meta=None,
                 visual=None):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
        self.meta = meta or RegionMeta()
        self.visual = visual or RegionVisual()

    def discretize_boundary(self, wcs, n_points=10):
        """
        Discretize the boundary into a PolygonSkyRegion.

        As SkyRegions are planar, this requires a WCS instance
        to map to a specified plane projection.

        Parameters
        ----------
        wcs : `~astropy.wcs.WCS`
            The world coordinate system transformation to use to convert
            between sky and pixel coordinates.

        n_points : int, optional
            Number of points along the each edge's boundary.

        Returns
        -------
        poly_sky_region: `~regions.PolygonSkyRegion`
            Planar PolygonSkyRegion object, with vertices in clockwise order.
        """
        # Transform to a PixelRegion, discretize, and then
        # convert back to a SkyRegion

        pixreg = self.to_pixel(wcs)
        disc_pixreg = pixreg.discretize_boundary(n_points=n_points)

        return disc_pixreg.to_sky(wcs)

    def to_pixel(self, wcs):
        center, pixscale, north_angle = pixel_scale_angle_at_skycoord(
            self.center, wcs)
        width = (self.width / pixscale).to(u.pix).value
        height = (self.height / pixscale).to(u.pix).value
        # region sky angles are defined relative to the WCS longitude axis;
        # photutils aperture sky angles are defined as the PA of the
        # semimajor axis (i.e., relative to the WCS latitude axis)
        angle = self.angle + (north_angle - 90 * u.deg)
        return RectanglePixelRegion(center, width, height, angle=angle,
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

        return RectangleSphericalSkyRegion(
            self.center.copy(),
            self.width.copy(),
            self.height.copy(),
            self.angle.copy(),
            meta=self.meta.copy(),
            visual=self.visual.copy()
        )


class RectangleSphericalSkyRegion(SphericalSkyRegion):
    """
    A rectangle in spherical sky coordinates.

    Parameters
    ----------
    center : `~astropy.coordinates.SkyCoord`
        The position of the center of the rectangle.
    width : `~astropy.units.Quantity`
        The width of the rectangle (before rotation) as an angle.
    height : `~astropy.units.Quantity`
        The height of the rectangle (before rotation) as an angle.
    angle : `~astropy.units.Quantity`, optional
        The rotation angle of the rectangle, measured anti-clockwise. If
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
    width = PositiveScalarAngle('The width of the rectangle (before rotation) '
                                'as a |Quantity| angle.')
    height = PositiveScalarAngle('The height of the rectangle (before '
                                 'rotation) as a |Quantity| angle.')
    angle = ScalarAngle('The rotation angle measured anti-clockwise as a '
                        '|Quantity| angle.')
    meta = RegionMetaDescr('The meta attributes as a |RegionMeta|')
    visual = RegionVisualDescr('The visual attributes as a |RegionVisual|.')

    def __init__(self, center, width, height, angle=0 * u.deg, meta=None,
                 visual=None):
        self.center = center
        self.width = width
        self.height = height
        self.angle = angle
        self.meta = meta or RegionMeta()
        self.visual = visual or RegionVisual()

    def _get_sph_rect_ref_points(self):
        # Ref points in CW order: w2, h2, w1, h1
        #   -------- h1 --------
        #  |                    |
        #  w1        c          w2
        #  |                    |
        #   -------- h2 --------
        # with the entire rectangle rotated by self.angle
        # Account for definition differences:
        # SkyCoord.directional_offset_by takes PA defined E of N
        # Rect angle is angle N of W.

        ref_pts = []
        ref_pts.append(self.center.directional_offset_by(self.angle.to(u.deg) - 90 * u.deg,
                                                         0.5 * self.width))
        ref_pts.append(self.center.directional_offset_by(self.angle.to(u.deg) + 180 * u.deg,
                                                         0.5 * self.height))
        ref_pts.append(self.center.directional_offset_by(self.angle.to(u.deg) + 90 * u.deg,
                                                         0.5 * self.width))
        ref_pts.append(self.center.directional_offset_by(self.angle.to(u.deg),
                                                         0.5 * self.height))

        return ref_pts

    def _get_sph_angle_transf(
        self, center_transf, frame, merge_attributes=True
    ):

        ref_pts = self._get_sph_rect_ref_points()
        # Ref points in CW order: w2, h2, w1, h1
        w2_ref_pt_transf = ref_pts[0].transform_to(
            frame, merge_attributes=merge_attributes
        )

        # Account for definition differences:
        # SkyCoord.directional_offset_by takes PA defined E of N
        # Rect angle is angle N of W.
        angle_transf = center_transf.position_angle(w2_ref_pt_transf).to(u.deg) + 90 * u.deg

        return angle_transf

    @property
    def _edge_circs(self):
        """
        Get list of the great circles defining the rectangle boundaries.
        """
        ref_pts = self._get_sph_rect_ref_points()
        # Ref points in CW order: w2, h2, w1, h1

        ref_c_gcs = [cross_product_skycoord2skycoord(ref_pt, self.center) for ref_pt in ref_pts]

        gcs = []
        for i in range(len(ref_c_gcs)):
            c_gc = cross_product_skycoord2skycoord(
                ref_c_gcs[i], ref_pts[i],
            )
            gcs.append(CircleSphericalSkyRegion(c_gc, 90 * u.deg))

        return gcs

    @property
    def _compound_region(self):
        # Need N great circles to define boundaries for an N-sided polygon -- here, a rectangle
        # verts are in CW order: Cross product to get bounding great circle centers
        # Compute GCs and stack into a compound set:
        compreg = None
        gcs = self._edge_circs
        for gc in gcs:
            if compreg is None:
                compreg = gc
            else:
                compreg = CompoundSphericalSkyRegion(
                    compreg, gc, operator.and_, self.meta, self.visual,
                )

        return compreg

    @property
    def vertices(self):
        """
        Spherical rectangle vertices, in clockwise order, starting from
        the SW vertex as ween on the sky (if angle=0).
        """
        verts = []
        gcs = self._edge_circs
        for i in range(len(gcs)):
            # Wrap around at 0:
            # ind_other = (i + 2) % len(gcs)
            # ind = (i + 1) % len(gcs)
            ind_other = (i + 1) % len(gcs)
            ind = (i) % len(gcs)
            verts.append(cross_product_skycoord2skycoord(
                gcs[ind].center, gcs[ind_other].center
            ))

        return SkyCoord(verts)

    def contains(self, coord):
        return self._compound_region.contains(coord)

    @property
    def bounding_circle(self):
        # Same bounding circle as for a polygon:
        cent = self.center
        seps = cent.separation(self.vertices)
        return CircleSphericalSkyRegion(center=cent, radius=np.max(seps))

    @property
    def bounding_lonlat(self):
        # Same bounding lon/lat as for a polygon:
        lons_arr, lats_arr = get_edge_raw_lonlat_bounds_circ_edges(
            self.vertices, self.center, self._edge_circs
        )

        lons_arr, lats_arr = self._validate_lonlat_bounds(lons_arr, lats_arr)

        return lons_arr, lats_arr

    def transform_to(self, frame, merge_attributes=True):
        frame = self._validate_frame(frame)

        center_transf = self.center.transform_to(
            frame, merge_attributes=merge_attributes
        )

        angle_transf = self._get_sph_angle_transf(
            center_transf, frame, merge_attributes=merge_attributes
        )

        return RectangleSphericalSkyRegion(
            center_transf,
            self.width.copy(),
            self.height.copy(),
            angle_transf,
            meta=self.meta.copy(),
            visual=self.visual.copy()
        )

    def discretize_boundary(self, n_points=10):
        bound_verts = discretize_all_edge_boundaries(
            self.vertices, self._edge_circs, n_points
        )
        return PolygonSphericalSkyRegion(bound_verts)

    def to_sky(
            self,
            wcs=None,
            include_boundary_distortions=False,
            discretize_kwargs=None
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

        return RectangleSkyRegion(
            self.center.copy(),
            self.width.copy(),
            self.height.copy(),
            self.angle.copy(),
            meta=self.meta.copy(),
            visual=self.visual.copy()
        )

    def to_pixel(
            self,
            wcs=None,
            include_boundary_distortions=False,
            discretize_kwargs=None,
    ):

        if discretize_kwargs is None:
            discretize_kwargs = {}
        if include_boundary_distortions:
            if wcs is None:
                raise ValueError(
                    "'wcs' must be set if 'include_boundary_distortions'=True"
                )
            # Requires spherical to planar projection (from WCS) and discretization
            verts = wcs.world_to_pixel(
                self.discretize_boundary(**discretize_kwargs).vertices
            )

            return PolygonPixelRegion(
                PixCoord(*verts), meta=self.meta.copy(),
                visual=self.visual.copy()
            )

        return self.to_sky().to_pixel(wcs)
