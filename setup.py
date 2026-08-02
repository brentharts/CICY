import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pyCICY",
    version="0.6.0",
    author="Robin Schneider",
    author_email="robin.schneider@physics.uu.se",
    description="A python CICY toolkit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/robin-schneider/CICY",
    packages=setuptools.find_packages(),
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    install_requires=[
        "numpy",
        "scipy",
        "sympy",
        "texttable",
    ],
    extras_require={
        # pyCICY.viz and CICY.plot_cohomologies need a plotting stack;
        # everything else works without one.
        "viz": ["matplotlib"],
    },
    entry_points={
        "console_scripts": [
            "pycicy-viz = pyCICY.viz:main",
        ],
    },
)
