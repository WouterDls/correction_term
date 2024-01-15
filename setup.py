from setuptools import find_packages, setup
my_pckg = find_packages(include=["correction_term"])
with open("README.md") as f:
    LONG_DESCRIPTION = f.read()
setup(
    name="Correction term",
    version="0.0.0",
    packages=my_pckg,
    include_package_data=True,
    url="https://github.com/WouterDls/correction_term",
    license="BSD-3",
    author="Wouter Deleersnyder",
    author_email="Wouter.Deleersnyder@kuleuven.be",
    description="A multidimensional AI-trained correction to the 1D approximate model for Airborne TDEM sensing",
    long_description=LONG_DESCRIPTION,
    install_requires=["numpy", "scipy", "cython", "sklearn", "scikit-fda", "pandas", "matplotlib"],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires=">=3.7",
)