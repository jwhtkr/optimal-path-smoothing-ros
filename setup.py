# noqa: D100
# pylint: disable=missing-module-docstring

from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=['path_smoothing', 'optimal_control', 'tools', 'geometry'],
    package_dir={'': 'src'},
    scripts=['']
)

setup(**setup_args)
