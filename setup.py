from setuptools import setup

version = {}
with open("snappyko/version.py") as fp:
    exec(fp.read(), version)

setup(name='SnappyKO',
      version=str(version['__version__']),
      description='Fast Keplerian orbit computation using Taylor series expansions.',
      author='Hannu Parviainen',
      author_email='hpparvi@gmail.com',
      url='https://github.com/hpparvi/snappyko',
      package_dir={'snappyko':'snappyko'},
      packages=['snappyko'],
      install_requires=["numpy", "numba", "semantic_version", "deprecated"],
      include_package_data=True,
      license='GPLv3',
      classifiers=[
          "Topic :: Scientific/Engineering",
          "Intended Audience :: Science/Research",
          "Intended Audience :: Developers",
          "Development Status :: 4 - Beta",
          "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
          "Operating System :: OS Independent",
          "Programming Language :: Python",
      ],
      keywords='astronomy astrophysics exoplanets'
      )
