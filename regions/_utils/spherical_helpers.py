# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module provides spherical sky region calculation helper tools.
"""

import astropy.units as u
import numpy as np
from astropy.coordinates import (Latitude, Longitude, SkyCoord,
                                 SphericalRepresentation,
                                 UnitSphericalRepresentation,
                                 cartesian_to_spherical)

from regions._utils.optional_deps import HAS_SPHERICAL_GEOMETRY

__all__ = []

# ------------------------------------------------------------------
# Spherical polygon functions

# ------------------------------------------------------------------
# vvvvvvv
# Handling non-convex spherical polygons


def _do_arc_segments_intersect(a, b, c, d):
    # Return an array (or scalar if a & b are scalars)
    # of booleans indicating if the arcs
    # defined by a-b and c-d intersect.
    # c and d are assumed to be single points.
    # a and b can be scalars or arrays.

    # Following
    # https://web.archive.org/web/20160505193152/https://www.mathworks.com/matlabcentral/newsreader/view_thread/276271
    # Definitions:
    # p = cross(a,b); % p is along the normal to the plane of arc a-to-b
    # q = cross(c,d); % Similarly for q and arc c-to-d
    # t = cross(p,q); % t is along the line of intersection of the planes
    # s1 = dot(cross(p,a),t);
    # s2 = dot(cross(b,p),t);
    # s3 = dot(cross(q,c),t);
    # s4 = dot(cross(d,q),t);

    cart_a = a.frame.represent_as('cartesian')
    cart_b = b.frame.represent_as('cartesian')
    cart_c = c.frame.represent_as('cartesian')
    cart_d = d.frame.represent_as('cartesian')

    # Check of length zero arc for either -> no intersection:
    if np.all(
        np.isclose(cart_a.x, cart_b.x)
        & np.isclose(cart_a.y, cart_b.y)
        & np.isclose(cart_a.z, cart_b.z)
    ) | np.all(
        np.isclose(cart_c.x, cart_d.x)
        & np.isclose(cart_c.y, cart_d.y)
        & np.isclose(cart_c.z, cart_d.z)
    ):
        if a.isscalar:
            return False
        else:
            return np.zeros(len(a), dtype=bool)

    cross_ab = cart_a.cross(cart_b)
    cross_cd = cart_c.cross(cart_d)

    cart_t = cross_ab.cross(cross_cd)

    s_vals = [
        np.atleast_1d(cross_ab.cross(cart_a).dot(cart_t)),
        np.atleast_1d(cart_b.cross(cross_ab).dot(cart_t)),
        np.atleast_1d(cross_cd.cross(cart_c).dot(cart_t)),
        np.atleast_1d(cart_d.cross(cross_cd).dot(cart_t))
    ]
    # If all same sign, abs value of sum of signs should be 4
    all_same_sign = (np.abs(np.sum(np.sign(s_vals).T, axis=1)) == 4)

    # Scalar:
    if a.isscalar:
        return all_same_sign[0]

    # Array:
    return all_same_sign


def _is_even_num_intersections(coord, verts, point):
    # Polygon contains following even-odd rule:
    # https://en.wikipedia.org/wiki/Even%E2%80%93odd_rule

    # Determine if number of intersections between the arcs
    # defined by the point and coords and the polygon segments
    # is even or odd.

    intersections = [
        np.atleast_1d(
            _do_arc_segments_intersect(
                coord,
                point,
                verts[i - 1],
                verts[i]
            )
        )
        for i in range(len(verts))
    ]

    # Numpy will sum booleans as True = 1, False = 0,
    # so no need to pre-convert:
    num_intersections = np.sum(intersections, axis=0)

    if coord.isscalar:
        return num_intersections[0] % 2 == 0

    return num_intersections % 2 == 0


def is_centroid_is_contained_in_polygon(verts, centroid):
    """
    Check if a spherical polygon region contains the centroid.

    Parameters
    ----------
    verts : `~astropy.coordinates.SkyCoord`
        The vertices of the spherical polygon.

    centroid : `~astropy.coordinates.SkyCoord`
        The centroid of the polygon.

    Returns
    -------
    contains : boolean
        Whether the centroid is contained in the polygon or not.
    """
    # Sanity check:
    # Using centroid and and a point close to the anti-centroid,
    # the number of intersections
    # along centroid-to-anti-centroid should be ODD.
    # If even, the centroid isn't actually contained, and the
    # opposite even/odd rule should be applied.

    cart_anti_centroid = - centroid.directional_offset_by(
        0, 0.01 * u.deg
    ).frame.represent_as('cartesian')

    _, lat, lon = cartesian_to_spherical(
        cart_anti_centroid.x,
        cart_anti_centroid.y,
        cart_anti_centroid.z
    )

    # Ensure internal data representation format is consistent
    # with input coordinate convention:
    unit = centroid.represent_as('spherical').lon.unit
    anti_centroid = SkyCoord(lon.to(unit), lat.to(unit), frame=centroid.frame)

    # Even number of intersections: BOTH outside.
    return not _is_even_num_intersections(anti_centroid, verts, centroid)


def spherical_polygon_contains(coord, verts, centroid):
    """
    Check if a spherical polygon region contains points.

    Parameters
    ----------
    coord : `~astropy.coordinates.SkyCoord`
        Coordinates of point(s) to check.

    verts : `~astropy.coordinates.SkyCoord`
        The vertices of the spherical polygon.

    centroid : `~astropy.coordinates.SkyCoord`
        The centroid of the polygon.

    Returns
    -------
    contains : list-like or boolean
        Boolean (scalar or list, corresponding to coords)
        of whether or not each point is contained within the polygon.
    """
    # Note: polygon edges must be < 180 deg of length!

    is_centroid_contained = is_centroid_is_contained_in_polygon(
        verts,
        centroid,
    )
    is_even_intersections = _is_even_num_intersections(coord, verts, centroid)

    if not is_centroid_contained:
        return np.logical_not(is_even_intersections)

    return is_even_intersections

# Handling non-convex spherical polygons
# ^^^^^^^
# ------------------------------------------------------------------


def _do_sph_polygon_contains_array(poly_sph, c_sph):
    lons = c_sph.lon.degree
    lats = c_sph.lat.degree
    # List comprehension loop over full set of coordinates
    cont_list = [
        poly_sph.contains_lonlat(lon, lat, degrees=True)
        for lon, lat in zip(lons, lats)
    ]
    return np.array(cont_list)


def do_sph_polygon_contains(
    coord, poly,
    bounding_circle_precut=125
):
    """
    Determine whether points are contained within a polygon or not,
    using functionality in spherical_geometry.

    Parameters
    ----------
    coord : `~astropy.coordinates.SkyCoord`
        The points to check as a SkyCoord.

    poly : `~regions.PolygonSphericalSkyRegion`
        The spherical polygon region instance.

    bounding_circle_precut : int or None, optional
        If set to an integer, perform a pre-cut check using
        a padded bounding circle.
    """
    # TODO: option for speeding up:
    # May be worth the added overhead of doing a check against
    # a padded bounding circle if len(coord) > ~125,
    # though this depends on distribution of coords
    # with respect to the polygon --- which is difficult to know
    # in general.
    # With len(coord) > ~125, the overhead is fairly small
    # so might be reasonable to include.

    poly_sph = poly._sph_geom_poly

    c_sph = coord.spherical
    if coord.isscalar:
        return poly_sph.contains_lonlat(
            c_sph.lon.degree,
            c_sph.lat.degree,
            degrees=True
        )
    else:
        if (bounding_circle_precut is not None) and (len(coord) > bounding_circle_precut):
            # Prefilter coordinate search:
            # Pad the bounding slightly and check only points
            # that pass the first cut.
            bc = poly.bounding_circle
            pad1 = bc.radius + 1 * u.arcsec
            pad2 = bc.radius * 1.01
            bc.radius = pad1 if pad1 < pad2 else pad2
            result = bc.contains(coord)
            resultpol = _do_sph_polygon_contains_array(poly_sph, c_sph[result])
            result[result] = resultpol
            return result

        # Otherwise, just apply sph_polygon contains on full coordinate set
        _do_sph_polygon_contains_array(poly_sph, c_sph)


def _optimize_polygon_contains_loop_by_sides(poly, coord, nverts):
    if HAS_SPHERICAL_GEOMETRY:
        from spherical_geometry import great_circle_arc
    else:
        raise ValueError(
            'Cannot use `do_optimized_polygon_contains()` '
            'unless `spherical_geometry` is installed!'
        )
    # Relies on sph_geom polygon:
    # Vertices are shoestringed, with the first and last points the same.
    poly_sph = poly._sph_geom_poly
    verts = poly_sph._polygons[0]._points
    inside = poly_sph._polygons[0]._inside

    # # Independent of sph_geom polygon creation:
    # v_cart = poly.vertices.cartesian
    # verts = np.array([
    #     v_cart._x.value,
    #     v_cart._y.value,
    #     v_cart._z.value
    # ]).T
    # inside = poly.centroid_avg.cartesian.xyz.value
    # # QUESTION: Will this lead to issues with centroid for edge cases?
    # # Yes it could, it seems, so skip this method even though
    # # faster for Nvert = 3 relative to the alternative above.

    # Necessary for shoestrining in loop below:
    # num_internal_verts = verts.shape[0]

    # # spherical_geometry method:
    # points = vector.lonlat_to_vector(
    #     c_sph.lon.degree,
    #     c_sph.lat.degree,
    #     degrees=True
    # )
    # # Restructure to Nx3 array:
    # points = np.asanyarray(points).T

    # Use astropy.coordinate.SkyCoord methods instead:
    c_cart = coord.cartesian
    points = np.array([
        c_cart._x.value,
        c_cart._y.value,
        c_cart._z.value
    ]).T

    # # Step does not appear to be necessary
    # insides_array = np.repeat([inside], ncoos, axis=0)

    # intersects_points = []
    # for i in range(nverts):
    #     intersects = great_circle_arc.intersects(
    #         verts[i], verts[i + 1], insides_array, points
    #         # verts[i], verts[(i + 1) % num_internal_verts], insides_array, points
    #     )
    #     intersects_points.append(intersects)

    # List comprehension
    intersects_points = [
        great_circle_arc.intersects(
            verts[i], verts[i + 1], inside, points
            # verts[i], verts[(i + 1) % num_internal_verts], insides_array, points
        )
        for i in range(nverts)
    ]

    return (np.sum(intersects_points, axis=0) % 2) == 0


def do_optimized_polygon_contains(
    coord, poly,
    gt_fallback=None,
    bounding_circle_precut=125,
):
    """
    Determine whether points are contained within a polygon or not,
    using functionality in spherical_geometry.

    Parameters
    ----------
    coord : `~astropy.coordinates.SkyCoord`
        The points to check as a SkyCoord.

    poly : `~regions.PolygonSphericalSkyRegion`
        The spherical polygon region instance.

    gt_fallback : int or None, optional
        If set to an integer, fallback to the polygon contains implemented in
        `spherical_polygon_contains()` from this package.

    bounding_circle_precut : int or None, optional
        If set to an integer, perform a pre-cut check using
        a padded bounding circle.
    """
    # Scalar point: just use built-in implementation:
    if coord.isscalar:
        return do_sph_polygon_contains(
            coord, poly,
            bounding_circle_precut=bounding_circle_precut
        )

    nverts = len(poly.vertices)
    ncoos = len(coord)
    # c_sph = coord.spherical

    if nverts > ncoos:
        # Use the built-in implementation:
        return do_sph_polygon_contains(
            coord, poly,
            bounding_circle_precut=bounding_circle_precut
        )

    # ----------------------------------------------
    # Otherwise, FLIP the implementation:
    # Explicitly loop over the edges, and use coordinates as array:

    if (gt_fallback is not None) and (ncoos > gt_fallback):
        # If roughly ncoos >~ 125, the fallback implementation is actually
        # FASTER.  This appears to be due to the use of loops, not vectors,
        # in the quad double calculations in spherical_gometry for
        # great_circle_arc.intersects().
        # The fallback will have some loss of accuracy because it uses doubles.
        return spherical_polygon_contains(
            coord,
            poly.vertices,
            poly.centroid
        )

    if (bounding_circle_precut is not None) and (len(coord) > bounding_circle_precut):
        # Prefilter coordinate search:
        # Pad the bounding slightly and check only points
        # that pass the first cut.
        bc = poly.bounding_circle
        pad1 = bc.radius + 1 * u.arcsec
        pad2 = bc.radius * 1.01
        bc.radius = pad1 if pad1 < pad2 else pad2
        result = bc.contains(coord)
        resultpol = _optimize_polygon_contains_loop_by_sides(poly, coord[result], nverts)
        result[result] = resultpol
        return result

    return _optimize_polygon_contains_loop_by_sides(poly, coord, nverts)

# ------------------------------------------------------------------
# Bounding lon/lat and edge discretization functions


def cross_product_skycoord2skycoord(c1, c2):
    """
    Compute cross product of two sky coordinates (from a spherical
    representation), returning a third sky coordinate (on a spherical
    representation).

    Parameters
    ----------
    c1 : `~astropy.coordinates.SkyCoord`
        The first SkyCoord.

    c2 : `~astropy.coordinates.SkyCoord`
        The second SkyCoord.

    Returns
    -------
    `~astropy.coordinates.SkyCoord`
        The cross product as a SkyCoord, with a spherical representation.
    """
    cross = c1.frame.represent_as('cartesian').cross(c2.frame.represent_as('cartesian'))
    c_cart = cross / cross.norm()
    _, lat, lon = cartesian_to_spherical(c_cart.x, c_cart.y, c_cart.z)
    return SkyCoord(lon, lat, frame=c1.frame)


def cross_product_sum_skycoord2skycoord(coos):
    """
    Cross product sum of vertices, assuming vertices are in CW order.

    Use to determine minimum distance between all vertices, as a
    centroid definition.

    Parameters
    ----------
    coos : `~astropy.coordinates.SkyCoord`
        The vertices as a SkyCoord.

    Returns
    -------
    `~astropy.coordinates.SkyCoord`
        The minimum distance centroid cross product as a SkyCoord,
        with a spherical representation.
    """
    # verts are in CW order:
    verts_cart = coos.frame.represent_as('cartesian')

    # Faster than loop:
    crosssum = np.append(
        np.array([verts_cart[-1].cross(verts_cart[0]).xyz.value]).T,
        verts_cart[:-1].cross(verts_cart[1:]).xyz.value,
        axis=1
    ).sum(axis=1)

    # Normalize sum of cross products to get cartesian representation of
    # centroid === minimum distance to other points location
    c_cart = crosssum / np.sqrt(np.sum(crosssum**2))
    _, lat, lon = cartesian_to_spherical(c_cart[0], c_cart[1], c_cart[2])

    # Ensure internal data representation format is consistent
    # with input coordinate convention:
    unit = coos[0].represent_as('spherical').lon.unit
    return SkyCoord(lon.to(unit), lat.to(unit), frame=coos.frame)


def bounding_lonlat_poles_processing(region, lons_arr, lats_arr, inner_region=None):
    """
    Check if region covers either pole & modify latitude bounds
    accordingly.

    Parameters
    ----------
    region : `~regions.SphericalSkyRegion`
        The SphericalSkyRegion region.

    lons_arr : list-like [`~astropy.coordinates.Longitude`]
        Initial estimate of longitude bounds.

    lats_arr : list-like [`~astropy.coordinates.Latitude`]
        Initial estimate of latitude bounds.

    inner_region : `~regions.SphericalSkyRegion`, optional
        The inner region, for an annulus (which modifies the pole logic).
        Default is None (for a simply-connected region).

    Returns
    -------
    lons_arr : list-like [`~astropy.coordinates.Longitude`] or None
        Corrected longitude bounds. Is None if the region goes over
        the pole (as the region contains all longitudes).

    lats_arr : list-like [`~astropy.coordinates.Latitude`]
        Corrected latitude bounds.
    """
    # Check if shape covers either pole & modify lats arr accordingly:
    poles = SkyCoord([0, 0], [-90, 90], unit=u.deg, frame=region.frame)

    # ------------------------------------
    # Simply connected region
    if inner_region is None:
        pole_contains = region.contains(poles)
        if np.any(pole_contains):
            lons_arr = None
            # S pole:
            if pole_contains[0]:
                lats_arr[0] = -90 * u.deg
            # N pole
            if pole_contains[1]:
                lats_arr[1] = 90 * u.deg

        if lons_arr is not None:
            return Longitude(lons_arr), Latitude(lats_arr)

        return None, Latitude(lats_arr)

    # ------------------------------------
    # Inner region set: annulus logic:
    pole_contains = inner_region.contains(poles)
    if np.any(pole_contains):
        lats_raw_inner = get_circle_latitude_tangent_limits(inner_region.center,
                                                            inner_region.radius)

        # S pole:
        if pole_contains[0]:
            # Change lower lats bound to be the minimum of the
            # inner boundary itself
            lats_arr[0] = lats_raw_inner[0]
        # N pole
        if pole_contains[1]:
            # Change upper lats bound to be the maximum of the
            # inner boundary itself
            lats_arr[1] = lats_raw_inner[1]

    if lons_arr is not None:
        return Longitude(lons_arr), Latitude(lats_arr)

    return None, Latitude(lats_arr)


def _get_circle_latitude_tangent_points(center, radius):
    # Get the points on an arbitrary circle that
    # intersect the tangent latitude circle limits.

    # This includes lon values, with +180deg folding
    # for over the pole cases

    crepr = center.represent_as('spherical')

    lons = [crepr.lon, crepr.lon]
    lats = [crepr.lat - radius, crepr.lat + radius]

    # Fold if over poles:
    if lats[0] < -90 * u.deg:
        lats[0] = -180 * u.deg - lats[0]
        lons[0] = lons[0] + 180 * u.deg

    if lats[1] > 90 * u.deg:
        lats[1] = 180 * u.deg - lats[1]
        lons[1] = lons[1] + 180 * u.deg

    return SkyCoord(lons, lats, frame=center.frame)


def _get_circle_longitude_tangent_points(center, radius):
    # Get the points on an arbitrary circle that
    # intersect the tangent longitude circle limits.

    # This includes lat values = 0deg
    # (all longitude circles have centers on equator)

    crepr = center.represent_as('spherical')

    lats = [0 * u.deg, 0 * u.deg]

    lats_ref = np.array(
        [(crepr.lat - radius).to(u.deg).value, (crepr.lat + radius).to(u.deg).value]
    )
    lons = []

    if np.any(np.abs(lats_ref) > 90):
        return None

    if np.any(np.abs(lats_ref) == 90):
        return SkyCoord(
            [crepr.lon - 90 * u.deg, crepr.lon + 90 * u.deg], lats, frame=center.frame
        )

    for sgn in [-1, 1]:
        # Do 1 then -1, because of lon increasing to east
        lon_gc = crepr.lon - sgn * (
            np.arccos(
                np.sin(radius.to(u.radian).value) / np.cos(crepr.lat.to(u.radian).value)
            )
            * u.radian
        ).to(u.deg)

        lons.append(lon_gc + sgn * 90 * u.deg)

    return SkyCoord(lons, lats, frame=center.frame)


def get_circle_latitude_tangent_limits(center, radius):
    """
    For an arbitrary spherical circle with center (lon0, lat0) and
    radius R0, get the tangent latitude circle limits (encoded at
    latitude values, not radii of the latitude circles).

    Use these to determine the latitude bounding limits from those tangents.
    (the points where the latitude circles intersect this circle)
    (Tangent circles have Rlat = 90 - (lat0 +- R0),
    equivalent to latitude limits lat0 +- R0)

    Ignores any issues with "over the pole" bounds --
    this only computes the values of the tangent circles.
    Other processing will handle over the pole bounds logic.

    Parameters
    ----------
    center : `~astropy.coordinates.SkyCoord`
        The circle center as a SkyCoord.

    radius : `~astropy.coordinates.Angle` or `~astropy.Quantity`
        The circle radius (as an |Angle| or |Quantity| with angular units).

    Returns
    -------
    latitude_limits : `~astropy.coordinates.Latitude`
        Length two |Latitude| with the latitudes of the tangent circles.
    """
    tan_lat_pts = _get_circle_latitude_tangent_points(center, radius)
    tan_lat_pts = tan_lat_pts.represent_as('spherical')

    return Latitude(tan_lat_pts.lat).to(u.deg)


def get_circle_longitude_tangent_limits(center, radius):
    """
    For an arbitrary spherical circle with center (lon0, lat0) and
    radius R0, get the longitudes of the centers of the tangent
    "longitude great circles" (as all longitude circles have
    latitude=0deg).

    Use these to determine the longitude bounds for this circle (the
    points where the longitude GCs intersect the circle).

    If | lat0 +- R0 | > 90deg: lon_bounds = None (Crosses pole, so no
    tangent longitude great circle exists -- all longitude lines [great
    circles] cross this circle.)

    If | lat0 +- R0 | = 90deg: lon_bounds = [lon0-90,lon0+90] (Touches
    either pole, so spans full 180 of longitudes centered on lon0)

    If | lat0 +- R0 | < 90deg:     lon0 +- arccos[ sin(R0) / cos(lat0) ]

    Parameters
    ----------
    center : `~astropy.coordinates.SkyCoord`
        The circle center as a SkyCoord.

    radius : `~astropy.coordinates.Angle` or `~astropy.Quantity`
        The circle radius (as an |Angle| or |Quantity| with angular units).

    Returns
    -------
    longitude_limits : `~astropy.coordinates.Longitude`
        Length two |Longitude| with the longitudes of the centers
        of the tangent longitude great circles.
    """
    tan_lon_pts = _get_circle_longitude_tangent_points(center, radius)

    if tan_lon_pts is None:
        return None

    tan_lon_pts = tan_lon_pts.represent_as('spherical')
    return Longitude(tan_lon_pts.lon).to(u.deg)


def _add_tan_pts_if_in_pa_range(
    coord_list,
    tan_pts,
    gc,
    wrap_ang,
    pas_verts_wrap,
    coord=None,
):
    pas_tan_pts = gc.center.position_angle(tan_pts).to(u.deg)

    # CHECK RANGES:
    # To handle possible cases of lon values "wrapping around" across
    # the standard 360->0 wrap, wrap lon values + coord values
    # around the first entry angle
    pas_tan_pts_wrap = pas_tan_pts.wrap_at(wrap_ang)

    in_range = (pas_tan_pts_wrap >= pas_verts_wrap[0]) & (
        pas_tan_pts_wrap <= pas_verts_wrap[1]
    )

    if np.any(in_range):
        tptrepr = tan_pts.represent_as('spherical')
        coord_list = np.append(coord_list, getattr(tptrepr, coord)[in_range])

    return coord_list


def _check_edge_lt_pi(pas_verts, wrap_ang):
    pas_verts_wrap = pas_verts.wrap_at(wrap_ang)

    is_valid_arc_length = ((pas_verts_wrap[1] - pas_verts_wrap[0]).to(u.deg) <= 180 * u.deg)

    return pas_verts_wrap, is_valid_arc_length


def _validate_vertices_ordering(verts, gc, gc_center=None):
    if gc is not None:
        gc_center = gc.center
    pas_verts = gc_center.position_angle(verts).to(u.deg)

    wrap_ang = pas_verts[0]

    # Check ordering of vertices pas:

    # Principle: ALL EDGES must be <= 180 deg of length
    # So difference of pa[1] - pa[0] <= 180 deg
    # If not, swap the order.

    pas_verts_wrap, is_valid_arc_length = _check_edge_lt_pi(pas_verts, wrap_ang)

    if not is_valid_arc_length:
        wrap_ang_opp = pas_verts[-1]
        pas_verts_wrap_opp, is_valid_arc_length_opp = _check_edge_lt_pi(
            pas_verts[::-1], wrap_ang_opp
        )
        if is_valid_arc_length_opp:
            return pas_verts_wrap_opp, wrap_ang_opp

        raise ValueError('Invalid arc')  # should never occur

    return pas_verts_wrap, wrap_ang


def _validate_lon_bounds_ordering(lons_arr, centroid):
    # Invert longitude order if centroid is outside of range:
    wrap_ang = lons_arr[0]
    centroid_lon_wrap = centroid.represent_as('spherical').lon.wrap_at(wrap_ang)
    in_range_lon = (centroid_lon_wrap >= lons_arr[0].wrap_at(wrap_ang)) & (
        centroid_lon_wrap <= lons_arr[1].wrap_at(wrap_ang)
    )
    if not in_range_lon:
        lons_arr = lons_arr[::-1]

    return lons_arr


def get_edge_raw_lonlat_bounds_circ_edges(vertices, centroid, gcs):
    """
    Get the raw longitude / latitude bounds from the circle edges of
    spherical sky region.

    Parameters
    ----------
    vertices : `~astropy.coordinates.SkyCoord`
        The vertices as a SkyCoord.

    centroid : `~astropy.coordinates.SkyCoord`
        The polygon centroid as a SkyCoord.

    gcs : `~regions.CircleSphericalSkyRegion`
        The circle boundaries as CircleSphericalSkyRegion instances.

    Returns
    -------
    longitude_limits, latitude_limits: `~astropy.coordinates.Longitude`
        Length two |Longitude| and |Latitude| with the computed
        longitude/latitude bounds from the polygon edges.
    """
    # Consider lon/lat of vertices: may produce min/max bounds:
    vrepr = vertices.represent_as('spherical')

    # lons_list = vrepr.lon
    # lats_list = vrepr.lat

    # Special handling:
    # Exclude vertices from longitude bounds if any is on a pole
    lons_list = []
    lats_list = []
    for v in vrepr:
        if np.abs(v.lat.to(u.deg).deg) < 90:
            lons_list.append(v.lon)
            lats_list.append(v.lat)
    lons_list = Longitude(lons_list, unit=u.radian)
    lats_list = Latitude(lats_list, unit=u.radian)

    # Need to also check for "out bulging" from edges,
    # as far as latitude/lon bounds:
    # eg, 2 vertices at ~60deg: the gc arc goes ~closer to the pole;
    # a circle centered close to the pole but not extending over it:
    # WIDE lon bounds

    # Requires checking if the "bounding" points are *ON* the polygons.

    for i, gc in enumerate(gcs):
        # PAs from gc center to vertices:
        verts = SkyCoord(np.concatenate([[vertices[i - 1]], [vertices[i]]]))

        pas_verts_wrap, wrap_ang = _validate_vertices_ordering(verts, gc)

        # --------------------------------------------------------
        # Latitude tangent points from bound circle as len 2 SkyCoord:
        # Only add to the list if the tangent point is located along this edge

        tan_lat_pts = _get_circle_latitude_tangent_points(gc.center, gc.radius)

        lats_list = _add_tan_pts_if_in_pa_range(
            lats_list, tan_lat_pts, gc, wrap_ang, pas_verts_wrap,
            coord='lat'
        )

        # --------------------------------------------------------
        # Longitude tangent points from bound circle as len 2 SkyCoord:
        # Only add to the list if the tangent point is located along this edge

        tan_lon_pts = _get_circle_longitude_tangent_points(gc.center, gc.radius)
        if tan_lon_pts is not None:
            lons_list = _add_tan_pts_if_in_pa_range(
                lons_list, tan_lon_pts, gc, wrap_ang, pas_verts_wrap,
                coord='lon'
            )

    lons_arr = [lons_list.min(), lons_list.max()]
    lats_arr = [lats_list.min(), lats_list.max()]

    # --------------------------------------------------------
    # Invert longitude order if centroid is outside of range:
    lons_arr = _validate_lon_bounds_ordering(lons_arr, centroid)

    return Longitude(lons_arr).to(u.deg), Latitude(lats_arr).to(u.deg)


def get_line_edge_raw_lonlat_bounds_circ_edges(coords, center):
    """
    Get the raw longitude / latitude bounds from the circle edges of
    spherical sky region.

    Parameters
    ----------
    coords : `~astropy.coordinates.SkyCoord`
        The start, end of the line as a SkyCoord.

    center : `~astropy.coordinates.SkyCoord`
        The line center as a SkyCoord.

    Returns
    -------
    longitude_limits, latitude_limits: `~astropy.coordinates.Longitude`
        Length two |Longitude| and |Latitude| with the computed
        longitude/latitude bounds from the polygon edges.
    """
    # Consider lon/lat of vertices: may produce min/max bounds:
    vrepr = coords.represent_as('spherical')

    # Special handling:
    # Exclude vertices from longitude bounds if any is on a pole
    lons_list = []
    lats_list = []
    for v in vrepr:
        if np.abs(v.lat.to(u.deg).deg) < 90:
            lons_list.append(v.lon)
            lats_list.append(v.lat)
    lons_list = Longitude(lons_list, unit=u.radian)
    lats_list = Latitude(lats_list, unit=u.radian)

    # Need to also check for "out bulging" from edges,
    # as far as latitude/lon bounds:
    # eg, 2 vertices at ~60deg: the gc arc goes ~closer to the pole;
    # a circle centered close to the pole but not extending over it:
    # WIDE lon bounds

    gc_center = cross_product_skycoord2skycoord(coords[0], coords[1])
    gc_radius = 90 * u.deg

    pas_verts_wrap, wrap_ang = _validate_vertices_ordering(
        coords, None, gc_center=gc_center
    )

    # --------------------------------------------------------
    # Latitude tangent points from bound circle as len 2 SkyCoord:
    # Only add to the list if the tangent point is located along this edge
    tan_lat_pts = _get_circle_latitude_tangent_points(gc_center, gc_radius)

    lats_list = _add_tan_pts_if_in_pa_range(
        lats_list, tan_lat_pts, None, wrap_ang, pas_verts_wrap, coord='lat',
        gc_center=gc_center
    )

    # --------------------------------------------------------
    # Longitude tangent points from bound circle as len 2 SkyCoord:
    # Only add to the list if the tangent point is located along this edge

    tan_lon_pts = _get_circle_longitude_tangent_points(gc_center, gc_radius)
    if tan_lon_pts is not None:
        lons_list = _add_tan_pts_if_in_pa_range(
            lons_list, tan_lon_pts, None, wrap_ang, pas_verts_wrap, coord='lon',
            gc_center=gc_center
        )

    lons_arr = [lons_list.min(), lons_list.max()]
    lats_arr = [lats_list.min(), lats_list.max()]

    # --------------------------------------------------------
    # Invert longitude order if centroid is outside of range:
    lons_arr = _validate_lon_bounds_ordering(lons_arr, center)

    return Longitude(lons_arr).to(u.deg), Latitude(lats_arr).to(u.deg)


def _discretize_edge_boundary(vertices, circ, n_points,
                              circ_center=None, circ_radius=None):

    if circ is not None:
        circ_center = circ.center
        circ_radius = circ.radius

    # Discretize an edge boundary defined by a circle, geodetic or not:
    # either great circle arc, or a span of a non-great circle
    # (e.g., constant lat edges of RangeSphericalSkyRegion)

    # For every edge boundary: determine range of PAs spanned by lines
    # connecting circle center to the two vertices bounding that edge:
    pas_verts = circ.center.position_angle(vertices).to(u.deg)

    pas_verts_wrap, wrap_ang = _validate_vertices_ordering(
        vertices, circ, gc_center=circ_center,
    )

    # Need to wrap angles to calculate span: wrap at lower value:
    pas_verts_wrap = pas_verts.wrap_at(wrap_ang)
    pa_span = pas_verts_wrap[1] - pas_verts_wrap[0]

    # Sample angle range over pa_span with Npoints, and offset
    # to start at pas_verts[0]
    theta = np.linspace(0, 1, num=n_points, endpoint=False) * pa_span + pas_verts[0]

    # Calculate directional offsets to get boundary discretization,
    # with vertices as SkyCoords
    bound_verts = circ.center.directional_offset_by(theta, circ_radius)

    return bound_verts


def discretize_line_boundary(coords, n_points):
    """
    Discretize all edge boundaries for spherical sky regions.

    Parameters
    ----------
    coords : `~astropy.coordinates.SkyCoord`
        The start, end of the line as a SkyCoord.

    n_points : int
        The number of points for discretization along each edge.

    Returns
    -------
    edge_bound_verts : `~astropy.coordinates.SkyCoord`
        The discretized boundary edge vertices.
    """
    # Iterate over full set of vertices & boundary circles for
    # a region (polygon or range)

    gc_center = cross_product_skycoord2skycoord(coords[0], coords[1])
    gc_radius = 90 * u.deg

    bound_verts = _discretize_edge_boundary(coords, None, n_points,
                                            circ_center=gc_center, circ_radius=gc_radius)

    return SkyCoord(
        SkyCoord(
            bound_verts,
            representation_type=UnitSphericalRepresentation,
        ),
        representation_type=SphericalRepresentation,
    )


def discretize_all_edge_boundaries(vertices, circs, n_points):
    """
    Discretize all edge boundaries for spherical sky regions.

    Parameters
    ----------
    vertices : `~astropy.coordinates.SkyCoord`
        The vertices of the spherical sky region.

    circs : `~regions.CircleSphericalSkyRegion`
        The circle boundaries as CircleSphericalSkyRegion instances.

    n_points : int
        The number of points for discretization along each edge.

    Returns
    -------
    all_edge_bound_verts : `~astropy.coordinates.SkyCoord`
        The discretized boundary edge vertices.
    """
    # Iterate over full set of vertices & boundary circles for
    # a region (polygon or range)

    all_edge_bound_verts = None
    for i, circ in enumerate(circs):
        # PAs from gc center to vertices:
        verts = SkyCoord(np.concatenate([[vertices[i - 1]], [vertices[i]]]))

        bound_verts = _discretize_edge_boundary(verts, circ, n_points)

        if all_edge_bound_verts is None:
            all_edge_bound_verts = bound_verts
        else:
            all_edge_bound_verts = SkyCoord(np.concatenate(
                [all_edge_bound_verts.copy(), bound_verts]
            ))
            # For some reason concatenate is adding distances,
            # so use remove those by running from
            # UnitSpherical->SphericalRepresentation...
            all_edge_bound_verts = SkyCoord(
                SkyCoord(
                    all_edge_bound_verts,
                    representation_type=UnitSphericalRepresentation,
                ),
                representation_type=SphericalRepresentation,
            )

    return all_edge_bound_verts
