from setuptools import setup, find_packages
import time

with open('README.md', 'rt', encoding='utf-8') as file:
    long_description = file.read()

with open('requirements.txt', 'rt') as file:
    install_requires = file.readlines()

setup(
    name='spline',
    version=time.strftime('%Y.%m.%d.%H.%M', time.localtime()),
    description='lib',
    long_description=long_description,
    long_description_content_type='text/markdown',  # если long_description = .md
    author='Daniil Andryushin',
    author_email='',
    url='https://github.com/ParkhomenkoDV/spline.git',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=install_requires,
    package_data={'spline':['1139.xlsx', '6033.xlsx', '100092.xlsx'], },
)