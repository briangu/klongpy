import os
from setuptools import setup

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()


setup(
    name='klongpy',
    packages=['klongpy'],
    version='0.3.73',
    description='Python implementation of Klong language.',
    author='Brian Guarraci',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    install_requires=['numpy~=1.23.0'],
    python_requires='>=3.8',
    extras_require={
        'cupy': ["cupy"],
        'cuda12x': ["cupy-cuda12x"],
        'cuda11x': ["cupy-cuda11x"],
        'cuda111': ["cupy-cuda111"],
        'cuda110': ["cupy-cuda110"],
        'cuda102': ["cupy-cuda102"],
        'rocm-5-0': ["cupy-rocm-5-0"],
        'rocm-4-3': ["cupy-rocm-4-3"],
        'repl': ["colorama"],
    },
    include_package_data=True,
    test_suite='tests',
    scripts=['scripts/kgpy']
)
