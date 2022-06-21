from distutils.core import setup

import setuptools
from pip._internal.req import parse_requirements

requirements = [
    ir.requirement for ir in parse_requirements(
        'requirements.txt',
        session='test')]

setup(
    name='blip-vit',
    packages=['blip', 'blip.models'],
    version='0.0.2',  # Ideally should be same as your GitHub release tag varsion
    description='BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation',
    author='salesforce',
    author_email='junnan.li@salesforce.com',
    url='https://github.com/fernandoTB/BLIP',
    download_url='https://github.com/fernandoTB/BLIP/archive/refs/tags/0.0.1.tar.gz',
    keywords=['0.0.1'],
    install_requires=requirements,
    classifiers=[],
)
