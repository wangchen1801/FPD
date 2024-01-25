#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name='fpd',
    version='1.0',
    author='wangzc',
    url="https://github.com/wangchen1801/FPD",
    description="Code for 'Fine-Grained Prototypes Distillation for Few-Shot Object Detection(FPD).'",
    packages=find_packages(exclude=('configs', 'data', 'work_dirs')),
    # install_requires=['clip@git+ssh://git@github.com/openai/CLIP.git'],
)


