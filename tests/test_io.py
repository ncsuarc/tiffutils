#!/usr/bin/env python

import tiffutils
import fractions
from pyexiv2.metadata import ImageMetadata
import numpy as np
import os
import tempfile
import unittest

test_dir = os.path.dirname(__file__)

# GRBG DNG image of field
field_dng = os.path.join(test_dir, 'images/field.dng')
field_data = os.path.join(test_dir, 'images/field.npy')

class TestLoadDNG(unittest.TestCase):

    def test_cfa(self):
        data, cfa = tiffutils.load_dng(field_dng)
        self.assertEqual(cfa, tiffutils.CFA_GRBG)

    def test_data(self):
        reference = np.load(field_data)
        data, cfa = tiffutils.load_dng(field_dng)
        self.assertTrue((data==reference).all())

def str_to_array(s, shape):
    """
    Convert flat string list of floats to np.array with shape

    In Exif metadata, various matrices are stored as a string flattened
    list of fractions, with spaces between elements.  Convert these to
    float np.arrays of the appropriate shape.

    Arguments:
        s: String list to convert
        shape: Numpy shape of returned array

    Returns:
        np.array with matrix contents and shape
    """
    data = np.array([float(d) for d in s])
    return data.reshape(shape)

class TestSaveDNG(unittest.TestCase):

    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        self.name = os.path.join(self.tempdir, 'save_dng.dng')
        self.reference = np.load(field_data)

    def tearDown(self):
        if os.path.exists(self.name):
            os.remove(self.name)
        os.rmdir(self.tempdir)

    def test_data(self):
        tiffutils.save_dng(self.reference, self.name)
        data, cfa = tiffutils.load_dng(self.name)
        self.assertTrue((data==self.reference).all())

    def test_data_compressed(self):
        tiffutils.save_dng(self.reference, self.name, compression=True)
        data, cfa = tiffutils.load_dng(self.name)
        self.assertTrue((data==self.reference).all())

    def test_data_none(self):
        with self.assertRaises(TypeError):
            tiffutils.save_dng(None, self.name)

    def test_data_noncontiguous(self):
        data = np.zeros((10, 10))
        view = data[:,1]
        with self.assertRaises(ValueError):
            tiffutils.save_dng(view, self.name)

    def test_cfa(self):
        tiffutils.save_dng(self.reference, self.name,
                           cfa_pattern=tiffutils.CFA_BGGR)
        data, cfa = tiffutils.load_dng(self.name)
        self.assertEquals(cfa, tiffutils.CFA_BGGR)

    def test_color_matrix1(self):
        matrix = np.array([
           [1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]])

        tiffutils.save_dng(self.reference, self.name,
                           color_matrix1=matrix)

        meta = ImageMetadata(self.name)
        meta.read()
        color_matrix1 = str_to_array(meta['Exif.Image.ColorMatrix1'].value,
                                     matrix.shape)

        self.assertTrue((color_matrix1==matrix).all())

    def test_color_matrix1_bad(self):
        matrix = np.array([True, False, True])

        with self.assertRaises(ValueError):
            tiffutils.save_dng(self.reference, self.name,
                               color_matrix1=matrix)

    def test_color_matrix2_omitted(self):
        tiffutils.save_dng(self.reference, self.name)

        meta = ImageMetadata(self.name)
        meta.read()
        self.assertTrue('Exif.Image.ColorMatrix2' not in meta)

    def test_color_matrix2(self):
        matrix = np.array([
           [1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]])

        tiffutils.save_dng(self.reference, self.name,
                           color_matrix2=matrix)

        meta = ImageMetadata(self.name)
        meta.read()
        color_matrix2 = str_to_array(meta['Exif.Image.ColorMatrix2'].value,
                                     matrix.shape)

        self.assertTrue((color_matrix2==matrix).all())

    def test_color_matrix2_bad(self):
        matrix = np.array([True, False, True])

        with self.assertRaises(ValueError):
            tiffutils.save_dng(self.reference, self.name,
                               color_matrix2=matrix)

    def test_calibration_illuminant1_omitted(self):
        tiffutils.save_dng(self.reference, self.name)

        meta = ImageMetadata(self.name)
        meta.read()
        self.assertTrue('Exif.Image.CalibrationIlluminant1' not in meta)

    def test_calibration_illuminant1(self):
        tiffutils.save_dng(self.reference, self.name,
                           calibration_illuminant1=tiffutils.ILLUMINANT_D65)

        meta = ImageMetadata(self.name)
        meta.read()
        illuminant1 = float(meta['Exif.Image.CalibrationIlluminant1'].value)
        self.assertEquals(illuminant1, tiffutils.ILLUMINANT_D65)

    def test_calibration_illumimant2_omitted(self):
        tiffutils.save_dng(self.reference, self.name)

        meta = ImageMetadata(self.name)
        meta.read()
        self.assertTrue('Exif.Image.CalibrationIllumimant2' not in meta)

    def test_calibration_illuminant2(self):
        tiffutils.save_dng(self.reference, self.name,
                           calibration_illuminant2=tiffutils.ILLUMINANT_D65)

        meta = ImageMetadata(self.name)
        meta.read()
        illuminant2 = float(meta['Exif.Image.CalibrationIlluminant2'].value)
        self.assertEquals(illuminant2, tiffutils.ILLUMINANT_D65)
