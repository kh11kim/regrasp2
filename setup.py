from setuptools import setup, find_packages

setup(
   name='regrasp',
   version='0.1',
   description='A useful module',
   author='Kanghyun Kim',
   author_email='kh11kim@kaist.ac.kr',
   packages=find_packages(exclude=[]),  #same as name
   #install_requires=['bar', 'greek'], #external packages as dependencies
)