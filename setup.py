# Copyright 2023 Ram Rachum and collaborators.
# This program is distributed under the MIT license.
import setuptools
import re


def read_file(filename):
    with open(filename) as file:
        return file.read()

version = re.search("__version__ = '([0-9.]*)'",
                    read_file('stubborn/__init__.py')).group(1)

setuptools.setup(
    name='stubborn',
    version=version,
    author='Ram Rachum',
    author_email='ram@rachum.com',
    description='An Environment for Evaluating Stubbornness between Agents with Aligned Incentives',
    long_description=read_file('readme.md'),
    long_description_content_type='text/markdown',
    url='https://github.com/cool-RR/stubborn',
    packages=setuptools.find_packages(exclude=['tests*']),
    install_requires=read_file('requirements.txt'),
    extras_require={
        'tests': {
            'pytest',
        },
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: Implementation :: CPython',
        'License :: OSI Approved :: MIT License',
    ],
)
