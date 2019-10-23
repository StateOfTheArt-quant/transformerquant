# -*- coding: utf-8 -*-
# Copyright StateOfTheArt.quant. 
#
# * Commercial Usage: please contact allen.across@gmail.com
# * Non-Commercial Usage:
#     Licensed under the Apache License, Version 2.0 (the "License");
#     you may not use this file except in compliance with the License.
#     You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

from os.path import dirname, join

from setuptools import find_packages, setup


def read_file(file):
    with open(file, "rt") as f:
        return f.read()


with open(join(dirname(__file__), 'transformerquant/VERSION.txt'), 'rb') as f:
    version = f.read().decode('ascii').strip()
    

setup(
    name='transformerquant',
    version=version,
    description='transformerquant',
    packages=find_packages(exclude=[]),
    author='Jiang Yu',
    author_email='allen.across@gmail.com',
    license='Apache License v2',
    package_data={'': ['*.*']},
    url='https://github.com/StateOfTheArt-quant/transformerquant',
    install_requires=read_file("requirements.txt").strip(),
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "portfolio_analyzer = transformerquant.__main__:entry_point"
        ]
    },
    classifiers=[
        'Programming Language :: Python',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)

