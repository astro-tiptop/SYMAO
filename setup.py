from setuptools import setup

NAME = 'symao'
DESCRIPTION = ''
URL = 'https://github.com/FabioRossiArcetri/SYMAO'
EMAIL = 'fabio.rossi@inaf.it'
AUTHOR = 'Fabio Rossi'
LICENSE = 'MIT'


setup(name='symao',
      description=DESCRIPTION,
      url=URL,
      author_email=EMAIL,
      author=AUTHOR,
      license=LICENSE,
      version='1.0',
      packages=['symao', ],
      install_requires=[
          "numpy",
          "scipy",
          "sympy",
          "seeing",
      ]
      )
