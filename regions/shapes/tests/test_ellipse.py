# Licensed under a 3-clause BSD style license - see LICENSE.rst

import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Latitude, Longitude, SkyCoord
from astropy.io import fits
from astropy.tests.helper import assert_quantity_allclose
from astropy.utils.data import get_pkg_data_filename
from astropy.wcs import WCS
from numpy.testing import assert_allclose, assert_equal

from regions._utils.optional_deps import HAS_MATPLOTLIB
from regions.core import PixCoord, RegionMeta, RegionVisual
from regions.shapes.circle import CircleSphericalSkyRegion
from regions.shapes.ellipse import (EllipsePixelRegion, EllipseSkyRegion,
                                    EllipseSphericalSkyRegion)
from regions.shapes.polygon import (PolygonPixelRegion, PolygonSkyRegion,
                                    PolygonSphericalSkyRegion)
from regions.shapes.tests.test_common import (BaseTestPixelRegion,
                                              BaseTestSkyRegion,
                                              BaseTestSphericalSkyRegion)
from regions.tests.helpers import make_simple_wcs


@pytest.fixture(scope='session', name='wcs')
def wcs_fixture():
    filename = get_pkg_data_filename('data/example_header.fits')
    header = fits.getheader(filename)
    return WCS(header)


class TestEllipsePixelRegion(BaseTestPixelRegion):
    meta = RegionMeta({'text': 'test'})
    visual = RegionVisual({'color': 'blue'})
    reg = EllipsePixelRegion(center=PixCoord(3, 4), width=4, height=3,
                             angle=5 * u.deg, meta=meta, visual=visual)
    sample_box = [-2, 8, -1, 9]
    inside = [(4.5, 4)]
    outside = [(5, 4)]
    expected_area = 3 * np.pi
    expected_repr = ('<EllipsePixelRegion(center=PixCoord(x=3, y=4), '
                     'width=4, height=3, angle=5.0 deg)>')
    expected_str = ('Region: EllipsePixelRegion\ncenter: PixCoord(x=3, y=4)\n'
                    'width: 4\nheight: 3\nangle: 5.0 deg')

    def test_copy(self):
        reg = self.reg.copy()
        assert reg.center.xy == (3, 4)
        assert reg.width == 4
        assert reg.height == 3
        assert_allclose(reg.angle.to_value('deg'), 5)
        assert reg.meta == self.meta
        assert reg.visual == self.visual

    def test_pix_sky_roundtrip(self):
        wcs = make_simple_wcs(SkyCoord(2 * u.deg, 3 * u.deg), 0.1 * u.deg, 20)
        reg_new = self.reg.to_sky(wcs).to_pixel(wcs)
        assert_allclose(reg_new.center.x, self.reg.center.x)
        assert_allclose(reg_new.center.y, self.reg.center.y)
        assert_allclose(reg_new.width, self.reg.width)
        assert_allclose(reg_new.height, self.reg.height)
        assert_quantity_allclose(reg_new.angle, self.reg.angle)
        assert reg_new.meta == self.reg.meta
        assert reg_new.visual == self.reg.visual

        # test that converted region meta and visual are copies and not views
        reg_new.meta['text'] = 'new'
        reg_new.visual['color'] = 'green'
        assert reg_new.meta['text'] != self.reg.meta['text']
        assert reg_new.visual['color'] != self.reg.visual['color']

    def test_to_spherical_sky(self, wcs):
        sphskycircle = self.reg.to_spherical_sky(wcs,
                                                 include_boundary_distortions=False)
        assert isinstance(sphskycircle, EllipseSphericalSkyRegion)

        sphskypoly = self.reg.to_spherical_sky(wcs,
                                               include_boundary_distortions=True)
        assert isinstance(sphskypoly, PolygonSphericalSkyRegion)
        assert sphskypoly.contains(wcs.pixel_to_world(self.reg.center.x, self.reg.center.y))

    def test_to_spherical_sky_no_wcs(self):
        with pytest.raises(ValueError) as excinfo:
            _ = self.reg.to_spherical_sky(include_boundary_distortions=True)
        estr = "'wcs' must be set if 'include_boundary_distortions'=True"
        assert estr in str(excinfo.value)

    @pytest.mark.skipif(not HAS_MATPLOTLIB, reason='matplotlib is required')
    def test_as_artist(self):
        patch = self.reg.as_artist()
        assert_allclose(patch.center, (3, 4))
        assert_allclose(patch.width, 4)
        assert_allclose(patch.height, 3)
        assert_allclose(patch.angle, 5)

    def test_rotate(self):
        reg = self.reg.rotate(PixCoord(2, 3), 90 * u.deg)
        assert_allclose(reg.center.xy, (1, 4))
        assert_allclose(reg.angle.to_value('deg'), 95)

    def test_eq(self):
        reg = self.reg.copy()
        assert reg == self.reg
        reg.width = 3
        assert reg != self.reg

    def test_discretize(self):
        regpixdiscr = self.reg.discretize_boundary(n_points=10)
        assert isinstance(regpixdiscr, PolygonPixelRegion)
        assert len(regpixdiscr.vertices) == 10

        # Validate ordering of vertices:
        assert regpixdiscr.contains(self.reg.center)

        # Test smaller width/height all contained:
        reg2 = self.reg.copy()
        reg2.width = self.reg.width * 0.75
        reg2.height = self.reg.height * 0.75
        reg2disc = reg2.discretize_boundary(n_points=10)
        assert regpixdiscr.contains(reg2disc.vertices).all()

    def test_region_bbox(self):
        a = 7
        b = 3
        reg = EllipsePixelRegion(PixCoord(50, 50), width=a, height=b,
                                 angle=0. * u.deg)
        assert reg.bounding_box.shape == (b, a)

        reg = EllipsePixelRegion(PixCoord(50.5, 50.5), width=a, height=b,
                                 angle=0. * u.deg)
        assert reg.bounding_box.shape == (b + 1, a + 1)

        reg = EllipsePixelRegion(PixCoord(50, 50), width=a, height=b,
                                 angle=90. * u.deg)
        assert reg.bounding_box.shape == (a, b)

    def test_region_bbox_zero_size(self):
        with pytest.raises(ValueError):
            EllipsePixelRegion(PixCoord(50, 50), width=0, height=0,
                               angle=0. * u.deg)

        with pytest.raises(ValueError):
            EllipsePixelRegion(PixCoord(50, 50), width=10, height=0,
                               angle=0. * u.deg)

    # temporarily disable sync=True test due to random failures
    # @pytest.mark.parametrize('sync', (False, True))
    @pytest.mark.parametrize('sync', (False,))
    def test_as_mpl_selector(self, sync):

        plt = pytest.importorskip('matplotlib.pyplot')

        rng = np.random.default_rng(0)
        data = rng.random((16, 16))
        mask = np.zeros_like(data)

        ax = plt.subplot(1, 1, 1)
        ax.imshow(data)

        def update_mask(reg):
            mask[:] = reg.to_mask(mode='subpixels', subpixels=10).to_image(data.shape)

        # For now this will only work with unrotated ellipses. Once this
        # works with rotated ellipses, the following exception check can
        # be removed as well as the ``angle=0 * u.deg`` in the call to
        # copy() below.
        with pytest.raises(NotImplementedError,
                           match=('Cannot create matplotlib selector for rotated ellipse.')):
            self.reg.as_mpl_selector(ax)

        region = self.reg.copy(angle=0 * u.deg)

        selector = region.as_mpl_selector(ax, callback=update_mask, sync=sync)  # noqa: F841

        from matplotlib.backend_bases import MouseEvent
        canvas = ax.figure.canvas

        evt = MouseEvent('button_press_event', canvas,
                         *ax.transData.transform((7.3, 4.4)), button=1)
        canvas.callbacks.process(evt.name, evt)
        evt = MouseEvent('motion_notify_event', canvas,
                         *ax.transData.transform((9.3, 5.4)), button=1)
        canvas.callbacks.process(evt.name, evt)
        evt = MouseEvent('button_release_event', canvas,
                         *ax.transData.transform((9.3, 5.4)), button=1)
        canvas.callbacks.process(evt.name, evt)
        ax.figure.canvas.draw()

        if sync:

            assert_allclose(region.center.x, 8.3)
            assert_allclose(region.center.y, 4.9)
            assert_allclose(region.width, 2)
            assert_allclose(region.height, 1)
            assert_quantity_allclose(region.angle, 0 * u.deg)

            assert_equal(mask, region.to_mask(mode='subpixels', subpixels=10).to_image(data.shape))

        else:

            assert_allclose(region.center.x, 3)
            assert_allclose(region.center.y, 4)
            assert_allclose(region.width, 4)
            assert_allclose(region.height, 3)
            assert_quantity_allclose(region.angle, 0 * u.deg)

            assert_equal(mask, 0)

        with pytest.raises(AttributeError, match=('Cannot attach more than one selector to a reg')):
            region.as_mpl_selector(ax)

    @pytest.mark.parametrize('anywhere', (False, True))
    def test_mpl_selector_drag(self, anywhere):
        """
        Test dragging of entire region from central handle and anywhere.
        """
        plt = pytest.importorskip('matplotlib.pyplot')

        rng = np.random.default_rng(0)
        data = rng.random((16, 16))
        mask = np.zeros_like(data)

        ax = plt.subplot(1, 1, 1)
        ax.imshow(data)

        def update_mask(reg):
            mask[:] = reg.to_mask(mode='subpixels', subpixels=10).to_image(data.shape)

        region = self.reg.copy(angle=0 * u.deg)

        selector = region.as_mpl_selector(ax, callback=update_mask,
                                          drag_from_anywhere=anywhere)
        assert selector.drag_from_anywhere is anywhere
        assert region._mpl_selector.drag_from_anywhere is anywhere

        # click_and_drag(selector, start=(3, 4), end=(3.5, 4.5))

        from matplotlib.backend_bases import MouseEvent
        canvas = ax.figure.canvas

        evt = MouseEvent('button_press_event', canvas,
                         *ax.transData.transform((3, 4)), button=1)
        canvas.callbacks.process(evt.name, evt)
        evt = MouseEvent('motion_notify_event', canvas,
                         *ax.transData.transform((3.5, 4.5)), button=1)
        canvas.callbacks.process(evt.name, evt)
        evt = MouseEvent('button_release_event', canvas,
                         *ax.transData.transform((3.5, 4.5)), button=1)
        canvas.callbacks.process(evt.name, evt)
        ax.figure.canvas.draw()

        assert_allclose(region.center.x, 3.5)
        assert_allclose(region.center.y, 4.5)
        assert_allclose(region.width, 4)
        assert_allclose(region.height, 3)

        evt = MouseEvent('button_press_event', canvas,
                         *ax.transData.transform((3.25, 4.25)), button=1)
        canvas.callbacks.process(evt.name, evt)
        evt = MouseEvent('motion_notify_event', canvas,
                         *ax.transData.transform((4.25, 5.25)), button=1)
        canvas.callbacks.process(evt.name, evt)
        evt = MouseEvent('button_release_event', canvas,
                         *ax.transData.transform((4.25, 5.25)), button=1)
        canvas.callbacks.process(evt.name, evt)
        ax.figure.canvas.draw()

        # For drag_from_anywhere=False this will have created a new 1x1 rectangle.
        if anywhere:
            assert_allclose(region.center.x, 4.5)
            assert_allclose(region.center.y, 5.5)
            assert_allclose(region.width, 4)
            assert_allclose(region.height, 3)
        else:
            assert_allclose(region.center.x, 4.5)
            assert_allclose(region.center.y, 5.5)

        assert_equal(mask, region.to_mask(mode='subpixels', subpixels=10).to_image(data.shape))

    @pytest.mark.parametrize('userargs',
                             ({'useblit': True},
                              {'grab_range': 20, 'minspanx': 5, 'minspany': 4},
                              {'props': {'facecolor': 'blue', 'linewidth': 2}},
                              {'twit': 'gumby'}))
    def test_mpl_selector_kwargs(self, userargs):
        """
        Test that additional kwargs are passed to selector.
        """
        plt = pytest.importorskip('matplotlib.pyplot')

        rng = np.random.default_rng(0)
        data = rng.random((16, 16))
        mask = np.zeros_like(data)

        ax = plt.subplot(1, 1, 1)
        ax.imshow(data)

        def update_mask(reg):
            mask[:] = reg.to_mask(mode='subpixels', subpixels=10).to_image(data.shape)

        region = self.reg.copy(angle=0 * u.deg)
        region.visual = {'color': 'red'}

        if 'twit' in userargs:
            with pytest.raises(TypeError, match=(r'__init__.. got an unexpected keyword argument')):
                selector = region.as_mpl_selector(ax, callback=update_mask, **userargs)
        else:
            selector = region.as_mpl_selector(ax, callback=update_mask, **userargs)
            assert region._mpl_selector.artists[0].get_edgecolor() == (1, 0, 0, 1)

            if 'props' in userargs:
                assert region._mpl_selector.artists[0].get_facecolor() == (0, 0, 1, 1)
                assert region._mpl_selector.artists[0].get_linewidth() == 2
            else:
                assert region._mpl_selector.artists[0].get_facecolor() == (0, 0, 0, 0)
                assert region._mpl_selector.artists[0].get_linewidth() == 1

                for key, val in userargs.items():
                    assert getattr(region._mpl_selector, key) == val
                    assert getattr(selector, key) == val


class TestEllipseSkyRegion(BaseTestSkyRegion):
    meta = RegionMeta({'text': 'test'})
    visual = RegionVisual({'color': 'blue'})
    reg = EllipseSkyRegion(center=SkyCoord(3, 4, unit='deg'), width=4 * u.deg,
                           height=3 * u.deg, angle=5 * u.deg, meta=meta,
                           visual=visual)

    expected_repr = ('<EllipseSkyRegion(center=<SkyCoord (ICRS): (ra, dec) '
                     'in deg\n    (3., 4.)>, width=4.0 deg, height=3.0 deg,'
                     ' angle=5.0 deg)>')
    expected_str = ('Region: EllipseSkyRegion\ncenter: <SkyCoord (ICRS): '
                    '(ra, dec) in deg\n    (3., 4.)>\nwidth: 4.0 deg\n'
                    'height: 3.0 deg\nangle: 5.0 deg')

    def test_copy(self):
        reg = self.reg.copy()
        assert_allclose(reg.center.ra.deg, 3)
        assert_allclose(reg.width.to_value('deg'), 4)
        assert_allclose(reg.height.to_value('deg'), 3)
        assert_allclose(reg.angle.to_value('deg'), 5)
        assert reg.meta == self.meta
        assert reg.visual == self.visual

    def test_transformation(self, wcs):
        skycoord = SkyCoord(3 * u.deg, 4 * u.deg, frame='galactic')
        skyellipse = EllipseSkyRegion(skycoord, 4 * u.arcsec,
                                      2 * u.arcsec, angle=30 * u.deg)

        pixellipse = skyellipse.to_pixel(wcs)

        assert_allclose(pixellipse.center.x, -50.5)
        assert_allclose(pixellipse.center.y, 299.5)
        assert_allclose(pixellipse.height, 0.027777777777828305)
        assert_allclose(pixellipse.width, 0.05555555555565661)

        skyellipse2 = pixellipse.to_sky(wcs)

        assert_quantity_allclose(skyellipse.center.data.lon,
                                 skyellipse2.center.data.lon)
        assert_quantity_allclose(skyellipse.center.data.lat,
                                 skyellipse2.center.data.lat)
        assert_quantity_allclose(skyellipse.width, skyellipse2.width)
        assert_quantity_allclose(skyellipse.height, skyellipse2.height)

        sphskyellipse = self.reg.to_spherical_sky(wcs,
                                                  include_boundary_distortions=False)
        assert isinstance(sphskyellipse, EllipseSphericalSkyRegion)

        sphskypoly = self.reg.to_spherical_sky(wcs,
                                               include_boundary_distortions=True)
        assert isinstance(sphskypoly, PolygonSphericalSkyRegion)
        assert sphskypoly.contains(self.reg.center)

    def test_to_spherical_sky_no_wcs(self):
        with pytest.raises(ValueError) as excinfo:
            _ = self.reg.to_spherical_sky(include_boundary_distortions=True)
        estr = "'wcs' must be set if 'include_boundary_distortions'=True"
        assert estr in str(excinfo.value)

    def test_dimension_center(self):
        center = SkyCoord([1, 2] * u.deg, [3, 4] * u.deg)
        width = 2 * u.arcsec
        height = 3 * u.arcsec
        with pytest.raises(ValueError) as excinfo:
            EllipseSkyRegion(center, width, height)
        estr = "'center' must be a scalar SkyCoord"
        assert estr in str(excinfo.value)

    def test_contains(self, wcs):
        position = SkyCoord([1, 3] * u.deg, [2, 4] * u.deg)
        # 1,2 is outside, 3,4 is the center and is inside
        assert all(self.reg.contains(position, wcs)
                   == np.array([False, True], dtype='bool'))

    def test_eq(self):
        reg = self.reg.copy()
        assert reg == self.reg
        reg.width = 3 * u.deg
        assert reg != self.reg

    def test_discretize(self, wcs):
        regskydiscr = self.reg.discretize_boundary(wcs, n_points=10)
        assert isinstance(regskydiscr, PolygonSkyRegion)
        assert len(regskydiscr.vertices) == 10

        # Validate ordering of vertices:
        assert regskydiscr.contains(self.reg.center, wcs)

        # Test smaller width/height all contained:
        reg2 = self.reg.copy()
        reg2.width = self.reg.width * 0.75
        reg2.height = self.reg.height * 0.75
        reg2disc = reg2.discretize_boundary(wcs, n_points=10)
        assert regskydiscr.contains(reg2disc.vertices, wcs).all()


class TestEllipseSphericalSkyRegion(BaseTestSphericalSkyRegion):
    inside = [(3 * u.deg, 4 * u.deg)]
    outside = [(3 * u.deg, 0 * u.deg)]
    meta = RegionMeta({'text': 'test'})
    visual = RegionVisual({'color': 'blue'})
    reg = EllipseSphericalSkyRegion(SkyCoord(3 * u.deg, 4 * u.deg),
                                    4 * u.arcsec, 2 * u.arcsec,
                                    angle=30 * u.deg,
                                    meta=meta, visual=visual)

    expected_repr = ('<EllipseSphericalSkyRegion(center=<SkyCoord (ICRS): (ra, dec) in '
                     'deg\n    (3., 4.)>, width=4.0 arcsec, height=2.0 arcsec, '
                     'angle=30.0 deg)>')
    expected_str = ('Region: EllipseSphericalSkyRegion\ncenter: <SkyCoord (ICRS): '
                    '(ra, dec) in deg\n    (3., 4.)>\nwidth: 4.0 arcsec'
                    '\nheight: 2.0 arcsec\nangle: 30.0 deg')

    def test_copy(self):
        reg = self.reg.copy()
        assert_allclose(reg.center.ra.deg, 3)
        assert_allclose(reg.width.to_value('arcsec'), 4)
        assert_allclose(reg.height.to_value('arcsec'), 2)
        assert_allclose(reg.angle.to_value('deg'), 30)
        assert reg.meta == self.meta
        assert reg.visual == self.visual

    def test_transformation(self, wcs):
        skycoord = SkyCoord(3 * u.deg, 4 * u.deg, frame='galactic')
        sphskyellipse = EllipseSphericalSkyRegion(skycoord, 4 * u.arcsec,
                                                  2 * u.arcsec, angle=30 * u.deg)

        pixellipse = sphskyellipse.to_pixel(wcs)

        assert_allclose(pixellipse.center.x, -50.5)
        assert_allclose(pixellipse.center.y, 299.5)
        assert_allclose(pixellipse.height, 0.027777777777828305)
        assert_allclose(pixellipse.width, 0.05555555555565661)

        skyellipse = self.reg.to_sky(wcs)
        assert isinstance(skyellipse, EllipseSkyRegion)

        sphskyellipse2 = pixellipse.to_spherical_sky(wcs)

        assert_quantity_allclose(sphskyellipse.center.data.lon,
                                 sphskyellipse2.center.data.lon)
        assert_quantity_allclose(sphskyellipse.center.data.lat,
                                 sphskyellipse2.center.data.lat)
        assert_quantity_allclose(sphskyellipse.width, sphskyellipse2.width)
        assert_quantity_allclose(sphskyellipse.height, sphskyellipse2.height)

        polysky = sphskyellipse.to_sky(wcs, include_boundary_distortions=True)
        assert isinstance(polysky, PolygonSkyRegion)

        polypix = sphskyellipse.to_pixel(wcs, include_boundary_distortions=True)
        assert isinstance(polypix, PolygonPixelRegion)

    def test_transformation_no_wcs(self):
        with pytest.raises(ValueError) as excinfo:
            _ = self.reg.to_sky(include_boundary_distortions=True)
        estr = "'wcs' must be set if 'include_boundary_distortions'=True"
        assert estr in str(excinfo.value)

        with pytest.raises(ValueError) as excinfo:
            _ = self.reg.to_pixel(include_boundary_distortions=True)
        estr = "'wcs' must be set if 'include_boundary_distortions'=True"
        assert estr in str(excinfo.value)

    def test_frame_transformation(self):
        skycoord = SkyCoord(3 * u.deg, 4 * u.deg, frame='galactic')
        reg = EllipseSphericalSkyRegion(skycoord, 4 * u.arcsec,
                                        2 * u.arcsec, angle=30 * u.deg)

        reg2 = reg.transform_to('icrs')
        assert reg2.center == skycoord.transform_to('icrs')
        assert_allclose(reg2.width.to_value('arcsec'), 4)
        assert isinstance(reg2, EllipseSphericalSkyRegion)
        assert reg2.frame.name == 'icrs'

    def test_dimension_center(self):
        center = SkyCoord([1, 2] * u.deg, [3, 4] * u.deg)
        width = height = 2 * u.arcsec
        angle = 30 * u.deg
        with pytest.raises(ValueError) as excinfo:
            EllipseSphericalSkyRegion(center, width, height, angle)
        estr = "'center' must be a scalar SkyCoord"
        assert estr in str(excinfo.value)

    def test_contains(self):
        # Add a test confirming rotation is correct:
        position = SkyCoord([3 * u.deg - 1.5 * u.arcsec, 3 * u.deg - 1.5 * u.arcsec],
                            [4 * u.deg + 1 * u.arcsec, 4 * u.deg - 1 * u.arcsec])
        # first is inside, second is outside
        assert all(self.reg.contains(position)
                   == np.array([True, False], dtype='bool'))

    def test_eq(self):
        reg = self.reg.copy()
        assert reg == self.reg
        reg.width = 3 * u.arcsec
        assert reg != self.reg

    def test_zero_size(self):
        with pytest.raises(ValueError):
            EllipseSphericalSkyRegion(SkyCoord(3 * u.deg, 4 * u.deg), 0. * u.arcsec,
                                      4 * u.arcsec, angle=30 * u.deg)

    def test_bounding_circle(self):
        skycoord = SkyCoord(3 * u.deg, 4 * u.deg, frame='icrs')
        reg = CircleSphericalSkyRegion(skycoord, 4 * u.arcsec)

        bounding_circle = self.reg.bounding_circle
        assert bounding_circle == reg

    def test_bounding_lonlat(self):
        bounding_lonlat = self.reg.bounding_lonlat

        assert_quantity_allclose(bounding_lonlat[0],
                                 Longitude([2.9994932098008498 * u.deg,
                                            3.000506789937744 * u.deg]))

        assert_quantity_allclose(bounding_lonlat[1],
                                 Latitude([3.999628954395713 * u.deg,
                                           4.000371045505152 * u.deg]))
