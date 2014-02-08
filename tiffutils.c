/*
 * Copyright (c) 2013, North Carolina State University Aerial Robotics Club
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the North Carolina State University Aerial Robotics Club
 *       nor the names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <Python.h>
#include <tiffio.h>

#ifndef TIFFTAG_CFAREPEATPATTERNDIM
#error libtiff with CFA pattern support required
#endif

#define PY_ARRAY_UNIQUE_SYMBOL  tiffutils_core_ARRAY_API
#include <numpy/arrayobject.h>

enum tiff_cfa_color {
    CFA_RED = 0,
    CFA_GREEN = 1,
    CFA_BLUE = 2,
};

enum cfa_pattern {
    CFA_BGGR = 0,
    CFA_GBRG,
    CFA_GRBG,
    CFA_RGGB,
    CFA_NUM_PATTERNS,
};

static const char cfa_patterns[4][CFA_NUM_PATTERNS] = {
    [CFA_BGGR] = {CFA_BLUE, CFA_GREEN, CFA_GREEN, CFA_RED},
    [CFA_GBRG] = {CFA_GREEN, CFA_BLUE, CFA_RED, CFA_GREEN},
    [CFA_GRBG] = {CFA_GREEN, CFA_RED, CFA_BLUE, CFA_GREEN},
    [CFA_RGGB] = {CFA_RED, CFA_GREEN, CFA_GREEN, CFA_BLUE},
};

/* Default ColorMatrix1, when none provided */
static const float default_color_matrix1[] = {
     2.005, -0.771, -0.269,
    -0.752,  1.688,  0.064,
    -0.149,  0.283,  0.745
};

/*
 * Create flat float array from PyArray
 *
 * Allocate and return a flatten float array from a 2D numpy array.
 *
 * @param array  2D numpy array to convert
 * @param dest  Pointer to destination array
 * @param len   Array size returned here
 * @returns 0 on success, negative on error, with exception set
 */
static int PyArray_to_float_array(PyObject *array, float **dest, int *len) {
    PyArray_Descr *float_descr;
    PyObject *float_array;
    npy_intp *dims;
    float **data;

    if (!PyArray_Check(array)) {
        PyErr_SetString(PyExc_TypeError, "Array must be a 2D ndarray");
        return -1;
    }

    if (PyArray_NDIM(array) != 2) {
        PyErr_SetString(PyExc_ValueError, "Array must be a 2D ndarray");
        return -1;
    }

    float_descr = PyArray_DescrFromType(NPY_FLOAT32);

    Py_INCREF(float_descr);
    float_array = PyArray_NewLikeArray((PyArrayObject*)array, NPY_CORDER,
                                       float_descr, 0);
    if (!float_array) {
        return -1;
    }

    /* Convert to float32 by copying into new array */
    if (PyArray_CopyInto((PyArrayObject*)float_array, (PyArrayObject*)array)) {
        goto err_decref_float_array;
    }

    dims = PyArray_DIMS(array);

    if (PyArray_AsCArray(&float_array, &data, dims, 2, float_descr)) {
        goto err_decref_float_array;
    }

    *len = dims[0]*dims[1];

    *dest = malloc(*len*sizeof(float));
    if (!*dest) {
        PyErr_SetString(PyExc_MemoryError, "Unable to allocate C array");
        goto err_free_c_array;
    }

    for (int i = 0; i < dims[0]; i++) {
        for (int j = 0; j < dims[1]; j++) {
            (*dest)[i*dims[1] + j] = data[i][j];
        }
    }

    PyArray_Free(float_array, data);
    Py_DECREF(float_array);
    return 0;

err_free_c_array:
    PyArray_Free(float_array, data);
err_decref_float_array:
    Py_DECREF(float_array);
    return -1;
}

/*
 * Create ColorMatrix1 array
 *
 * Create a color matrix array from list.  If no list provided, use the
 * default ColorMatrix1.
 *
 * @param array  2D numpy array containing color matrix
 * @param color_matrix1 Pointer to destination array for color matrix
 * @param len   Array size returned here
 * @returns 0 on success, negative on error, with exception set
 */
static int handle_color_matrix1(PyObject *array, float **color_matrix1, int *len) {
    /* No list provided, use default */
    if (array == Py_None) {
        *color_matrix1 = malloc(sizeof(default_color_matrix1));
        if (!*color_matrix1) {
            PyErr_SetString(PyExc_MemoryError, "Unable to allocate color matrix");
            return -1;
        }

        memcpy(*color_matrix1, default_color_matrix1,
               sizeof(default_color_matrix1));

        *len = sizeof(default_color_matrix1)/sizeof(float);

        return 0;
    }

    /* Convert provided array */
    return PyArray_to_float_array(array, color_matrix1, len);
}

static PyObject *tiffutils_save_dng(PyObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {
        "image", "filename", "camera", "cfa_pattern", "color_matrix1",
        "color_matrix2", "calibration_illuminant1", "calibration_illuminant2",
        NULL
    };

    PyArrayObject *array;
    PyObject *color_matrix1_ndarray = Py_None;
    PyObject *color_matrix2_ndarray = Py_None;
    unsigned short calibration_illuminant1 = 0;
    unsigned short calibration_illuminant2 = 0;
    unsigned int pattern = CFA_RGGB;
    float *color_matrix1, *color_matrix2 = NULL;
    int color_matrix1_len, color_matrix2_len;
    int ndims, width, height, type, bytes_per_pixel;
    npy_intp *dims;
    char *filename;
    char *camera = "Unknown";
    char *mem;
    TIFF *file = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Os|sIOOHH", kwlist, &array,
                                     &filename, &camera, &pattern,
                                     &color_matrix1_ndarray,
                                     &color_matrix2_ndarray,
                                     &calibration_illuminant1,
                                     &calibration_illuminant2)) {
        return NULL;
    }

    if (pattern >= CFA_NUM_PATTERNS) {
        PyErr_SetString(PyExc_ValueError, "Invalid CFA pattern");
        return NULL;
    }

    if (!PyArray_Check(array)) {
        PyErr_SetString(PyExc_TypeError, "ndarray required");
        return NULL;
    }

    if (!PyArray_ISCONTIGUOUS(array)) {
        PyErr_SetString(PyExc_TypeError, "ndarray must be contiguous");
        return NULL;
    }

    ndims = PyArray_NDIM(array);
    dims = PyArray_DIMS(array);
    type = PyArray_TYPE(array);
    mem = PyArray_BYTES(array);

    if (ndims != 2) {
        PyErr_SetString(PyExc_TypeError, "ndarray must be 2 dimensional");
        return NULL;
    }

    height = dims[0];
    width = dims[1];

    switch (type) {
    case NPY_UINT8:
        bytes_per_pixel = 1;
        break;
    case NPY_UINT16:
        bytes_per_pixel = 2;
        break;
    default:
        PyErr_SetString(PyExc_TypeError, "ndarray must be uint8 or uint16");
        return NULL;
    }

    if (handle_color_matrix1(color_matrix1_ndarray, &color_matrix1,
                             &color_matrix1_len)) {
        return NULL;
    }

    if ((color_matrix2_ndarray != Py_None) &&
        PyArray_to_float_array(color_matrix2_ndarray, &color_matrix2,
                               &color_matrix2_len)) {
        goto err;
    }

    file = TIFFOpen(filename, "w");
    if (file == NULL) {
        PyErr_SetString(PyExc_IOError, "libtiff failed to open file for writing.");
        goto err;
    }

    TIFFSetField(file, TIFFTAG_IMAGEWIDTH, width);
    TIFFSetField(file, TIFFTAG_IMAGELENGTH, height);
    TIFFSetField(file, TIFFTAG_UNIQUECAMERAMODEL, camera);

    TIFFSetField(file, TIFFTAG_ORIENTATION, ORIENTATION_TOPLEFT);
    TIFFSetField(file, TIFFTAG_PLANARCONFIG, PLANARCONFIG_CONTIG);
    TIFFSetField(file, TIFFTAG_SUBFILETYPE, 0);

    TIFFSetField(file, TIFFTAG_BITSPERSAMPLE, 8*bytes_per_pixel);
    TIFFSetField(file, TIFFTAG_SAMPLESPERPIXEL, 1);
    TIFFSetField(file, TIFFTAG_PHOTOMETRIC, PHOTOMETRIC_CFA);

    TIFFSetField(file, TIFFTAG_DNGVERSION, "\001\001\0\0");
    TIFFSetField(file, TIFFTAG_DNGBACKWARDVERSION, "\001\0\0\0");

    TIFFSetField(file, TIFFTAG_CFAREPEATPATTERNDIM, (short[]){2,2});
    TIFFSetField(file, TIFFTAG_CFAPATTERN, cfa_patterns[pattern]);
    TIFFSetField(file, TIFFTAG_COLORMATRIX1, color_matrix1_len, color_matrix1);

    if (color_matrix2) {
        TIFFSetField(file, TIFFTAG_COLORMATRIX2, color_matrix2_len, color_matrix2);
    }

    if (calibration_illuminant1) {
        TIFFSetField(file, TIFFTAG_CALIBRATIONILLUMINANT1, calibration_illuminant1);
    }

    if (calibration_illuminant2) {
        TIFFSetField(file, TIFFTAG_CALIBRATIONILLUMINANT2, calibration_illuminant2);
    }

    for (int row = 0; row < height; row++) {
        if (TIFFWriteScanline(file, mem, row, 0) < 0) {
            TIFFClose(file);
            PyErr_SetString(PyExc_IOError, "libtiff failed to write row.");
            goto err;
        }
        else {
            mem += width * bytes_per_pixel;
        }
    }

    TIFFWriteDirectory(file);
    TIFFClose(file);

    if (color_matrix2) {
        free(color_matrix2);
    }

    free(color_matrix1);

    Py_INCREF(Py_None);
    return Py_None;

err:
    if (color_matrix2) {
        free(color_matrix2);
    }
    free(color_matrix1);
    return NULL;
}

/*
 * Detect CFA pattern of tiff
 *
 * @param tiff  Image to detect pattern of
 * @returns PyObject of CFA type (one of the CFA constants),
 *          or None, if unknown.  NULL if exception raised.
 */
static PyObject *tiff_cfa(TIFF *tiff) {
    uint16_t *cfarepeatpatterndim[2];
    uint8_t *cfapattern[4];
    short x, y;

    if (!TIFFGetField(tiff, TIFFTAG_CFAREPEATPATTERNDIM, &cfarepeatpatterndim)) {
        goto none;
    }

    x = (*cfarepeatpatterndim)[0];
    y = (*cfarepeatpatterndim)[1];

    /* Only support 2x2 CFA patterns */
    if (x != 2 || y != 2) {
        goto none;
    }

    if (!TIFFGetField(tiff, TIFFTAG_CFAPATTERN, &cfapattern)) {
        goto none;
    }

    /* Look for matching known pattern */
    for (int i = 0; i < CFA_NUM_PATTERNS; i++) {
        for (int j = 0; j < 4; j++) {
            if ((*cfapattern)[j] != cfa_patterns[i][j]) {
                break;
            }
            /* Found a match */
            else if (j == 3) {
                return PyLong_FromLong(i);
            }
        }
    }

none:
    Py_INCREF(Py_None);
    return Py_None;
}

static PyObject *tiffutils_load_dng(PyObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {
        "filename", NULL
    };

    char *filename;
    TIFF *tiff = NULL;
    uint32_t imagelength;
    tsize_t scanlinesize;
    uint16_t planarconfig, samplesperpixel, bitspersample;
    PyObject *cfa = NULL;
    int type;
    npy_intp dims[2];
    PyObject *array;
    PyArray_Descr *descr;
    void *data;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s", kwlist, &filename)) {
        return NULL;
    }

    /* Surpress warnings */
    TIFFSetWarningHandler(NULL);

    tiff = TIFFOpen(filename, "r");
    if (!tiff) {
        PyErr_SetString(PyExc_IOError, "Failed to open file");
        return NULL;
    }

    scanlinesize = TIFFScanlineSize(tiff);

    if (!TIFFGetField(tiff, TIFFTAG_IMAGELENGTH, &imagelength)) {
        PyErr_SetString(PyExc_IOError, "Image length not found");
        goto err;
    }

    if (!TIFFGetField(tiff, TIFFTAG_PLANARCONFIG, &planarconfig)) {
        /* Contiguous is default */
        planarconfig = PLANARCONFIG_CONTIG;
    }

    if (!TIFFGetField(tiff, TIFFTAG_SAMPLESPERPIXEL, &samplesperpixel)) {
        /* 1 is default */
        samplesperpixel = 1;
    }

    if (!TIFFGetField(tiff, TIFFTAG_BITSPERSAMPLE, &bitspersample)) {
        /* 1 is default */
        bitspersample = 1;
    }

    if (planarconfig != PLANARCONFIG_CONTIG) {
        PyErr_SetString(PyExc_ValueError, "Only contiguous planar configuration supported");
        goto err;
    }

    if (samplesperpixel != 1) {
        PyErr_SetString(PyExc_ValueError, "Only 1 sample per pixel supported");
        goto err;
    }

    /* Detect CFA pattern */
    cfa = tiff_cfa(tiff);
    if (!cfa) {
        goto err;
    }

    /* Create array */

    switch (bitspersample) {
    case 8:
        type = NPY_UINT8;
        break;
    case 16:
        type = NPY_UINT16;
        break;
    default:
        PyErr_Format(PyExc_ValueError, "Unsupported bit depth %hu", bitspersample);
        goto err_decref_cfa;
    }

    descr = PyArray_DescrFromType(type);
    if (!descr) {
        goto err_decref_cfa;
    }

    dims[0] = imagelength;
    dims[1] = scanlinesize / (bitspersample/8);

    Py_INCREF(descr);
    array = PyArray_NewFromDescr(&PyArray_Type, descr, 2, dims,
                                 NULL, NULL, 0, NULL);
    if (!array) {
        goto err_decref_cfa;
    }

    data = PyArray_DATA(array);

    for (int row = 0; row < imagelength; row++) {
        if (TIFFReadScanline(tiff, data, row, 0) < 0) {
            PyErr_SetString(PyExc_IOError, "libtiff failed to read row");
            goto err_decref_array;
        }

        data += scanlinesize;
    }

    TIFFClose(tiff);

    return Py_BuildValue("(NN)", array, cfa);

err_decref_array:
    Py_DECREF(array);
err_decref_cfa:
    Py_DECREF(cfa);
err:
    TIFFClose(tiff);
    return NULL;
}

PyMethodDef tiffutilsMethods[] = {
    {"save_dng", (PyCFunction) tiffutils_save_dng, METH_VARARGS | METH_KEYWORDS,
        "save_dng(image, filename, [camera='Unknown', cfa_pattern=tiffutils.CFA_RGGB,\n"
        "   color_matrix1=None])\n\n"
        "Save an ndarray as a DNG.\n\n"
        "The image will be saved as a RAW DNG,a superset of TIFF.\n"
        "Arguments:\n"
        "    image: Image to save.  This should be a 2-dimensional, uint8 or\n"
        "        uint16 Numpy array.\n"
        "    filename: Destination file to save DNG to.\n"
        "    camera: Unique name of camera model\n"
        "    cfa_pattern: Bayer color filter array pattern.\n"
        "       One of tiffutils.CFA_*\n"
        "    color_matrix1: A 2D ndarray containing the desired ColorMatrix1.\n"
        "       If not specified, a default is used.\n"
        "    color_matrix2: A 2D ndarray containing the desired ColorMatrix2.\n"
        "       If not specified, the field is omitted.\n"
        "    calibration_illuminant1: The desired CalibrationIlluminant1 value.\n"
        "       If not specified or 0, the field is omitted.\n"
        "    calibration_illuminant2: The desired CalibrationIlluminant2 value.\n"
        "       If not specified or 0, the field is omitted.\n\n"
        "Raises:\n"
        "    TypeError: image is not the appropriate format.\n"
        "    IOError: file could not be written."
    },
    {"load_dng", (PyCFunction) tiffutils_load_dng, METH_VARARGS,
        "load_dng(filename) -> image ndarray\n\n"
        "Load DNG file as ndarray.\n"
        "Expects a CFA image, with 1 sample per pixel and 8- or\n"
        "16-bits per pixel.\n\n"
        "Arguments:\n"
        "   filename: Path to file to load\n\n"
        "Returns:\n"
        "   (image, cfa), where image is an ndarray containing the image\n"
        "   data, and cfa is one of the tiffutils.CFA_* constants describing\n"
        "   the CFA pattern of the image, or None, if unknown.\n\n"
        "Raises:\n"
        "   IOError: Unable to open or read file\n"
        "   ValueError: Unsupported DNG format\n"
    },
    {NULL, NULL, 0, NULL}
};

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef tiffutilsmodule = {
    PyModuleDef_HEAD_INIT,
    "tiffutils",    /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
    tiffutilsMethods
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_tiffutils(void) {
#else
PyMODINIT_FUNC inittiffutils(void) {
#endif
    PyObject* m;

    import_array();

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&tiffutilsmodule);
#else
    m = Py_InitModule("tiffutils", tiffutilsMethods);
#endif

    if (m == NULL) {
#if PY_MAJOR_VERSION >= 3
        return NULL;
#else
        return;
#endif
    }

    PyModule_AddIntConstant(m, "CFA_BGGR", CFA_BGGR);
    PyModule_AddIntConstant(m, "CFA_GBRG", CFA_GBRG);
    PyModule_AddIntConstant(m, "CFA_GRBG", CFA_GRBG);
    PyModule_AddIntConstant(m, "CFA_RGGB", CFA_RGGB);

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}

int main(int argc, char *argv[]) {
#if PY_MAJOR_VERSION >= 3
    wchar_t name[128];
    mbstowcs(name, argv[0], 128);
#else
    char name[128];
    strncpy(name, argv[0], 128);
#endif

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(name);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Add a static module */
#if PY_MAJOR_VERSION >= 3
    PyInit_tiffutils();
#else
    inittiffutils();
#endif

    return 0;
}
