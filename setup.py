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
    author_email='yujiangallen@126.com',
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

