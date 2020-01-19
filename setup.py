from setuptools import Extension, setup
# We import as _build_ext so that we can subclass it with our own
from setuptools.command.build_ext import build_ext as _build_ext


class build_ext(_build_ext):
    def finalize_options(self):
        _build_ext.finalize_options(self)
        # Prevent numpy from thinking it is still in its setup process
        __builtins__.__NUMPY_SETUP__ = False
        import numpy

        self.include_dirs.append(numpy.get_include())


setup(
    name="tiffutils",
    version="1.0.0",
    description="Utilities for TIFF files",
    author="NC State Aerial Robotics Club",
    author_email="aerialrobotics@ncsu.edu",
    license="BSD",
    # So that we can import numpy
    cmdclass={"build_ext": build_ext},
    setup_requires=["numpy"],
    ext_modules=[
        Extension(
            "tiffutils",
            extra_compile_args=["-std=gnu99", "-g3"],
            libraries=["tiff"],
            sources=["tiffutils.c"],
        )
    ],
)
