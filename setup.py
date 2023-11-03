'''
Copyright 2022 Airbus SAS
Modifications on 2023/06/06-2023/11/03 Copyright 2023 Capgemini

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
# coding: utf-8

from setuptools import setup, find_packages
from datetime import date
import os


def __path(filename):
    ''''Build a full absolute path using the given filename

        :params filename : filename to ass to the path of this module
        :returns: full builded path
    '''
    return os.path.join(os.path.dirname(__file__), filename)


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

# Read the requirements.in file and extract the required dependencies
with open('requirements.in', 'r') as requirements_file:
    reqs_list = [line.strip() for line in requirements_file if line.strip()]

# Manage module version using date
today = date.today()

# formating the date as yy.mm.dd
version = today.strftime('%y.%m.%d')

# check if the version.info file is existing (add a suffix to the version
# in case of multiple release in a day)
# it is intended that the version.info file contain only one line with the
# suffix information

suffix = ''
if os.path.exists(__path('version.info')):
    suffix = open(__path('version.info')).read().strip()

if len(suffix) > 0:
    version = f'{version}.{suffix}'

setup(
    name='climate-economics',
    version=version,
    description='Python library to evaluate climate change impact',
    long_description=readme,
    author='Airbus SAS, Capgemini',
    url='https://github.com/os-climate/witness-core',
    license=license,
    packages=find_packages(exclude=('tests', 'docs')),
    include_package_data=True,
    python_requires='>=3.9',
    install_requires=reqs_list,
)
