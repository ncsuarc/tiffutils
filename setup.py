from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class numpy_build_ext(build_ext):
    """
    Subclass of build_ext that dynamically imports numpy and adds
    numpy.get_include() to the include_dirs.
    """

    def finalize_options(self):
        build_ext.finalize_options(self)
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
    # Use our custom_build_ext that dynamically imports numpy and make sure
    # that numpy is installed before we run it
    cmdclass={"build_ext": numpy_build_ext},
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
