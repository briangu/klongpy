import os
from setuptools import setup, find_packages

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()


extra_requires = {
        'cupy': ["cupy"],
        'cuda12x': ["cupy-cuda12x"],
        'cuda11x': ["cupy-cuda11x"],
        'cuda111': ["cupy-cuda111"],
        'cuda110': ["cupy-cuda110"],
        'cuda102': ["cupy-cuda102"],
        'rocm-5-0': ["cupy-rocm-5-0"],
        'rocm-4-3': ["cupy-rocm-4-3"],
        'repl': ["colorama==0.4.6"],
        'web': ["aiohttp==3.8.5"],
        'db': ["pandas==2.0.3","duckdb==0.8.1"],
        'ws': ["websockets==11.0.3"],
    }

# full feature set extras
extra_requires['full'] = extra_requires['repl'] + extra_requires['web'] + extra_requires['db'] + extra_requires['ws']
extra_requires['all'] = extra_requires['full']

setup(
    name='klongpy',
    packages=find_packages(),
    version='0.5.13',
    description='High-Performance Klong array language with rich Python integration.',
    author='Brian Guarraci',
    license='MIT',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
    ],
    install_requires=['numpy~=1.24.4'],
    python_requires='<=3.12,>=3.8',
    extras_require=extra_requires,
    include_package_data=True,
    zip_safe=False,
    test_suite='tests',
    scripts=['scripts/kgpy']
)
