from setuptools import setup, find_packages

setup(name='flaxcrf',
      version='0.1.0',
      description="Conditional random field in JAX",
      long_description='',
      url='https://github.com/alao/flaxcrf.git',
      author='jimmy',
      author_email='alao556@live.com',
      license='Apache-2.0',
      classifiers=["Programming Language :: Python :: 3.7",
                   "Programming Language :: Python :: 3.8",
                   "Programming Language :: Python :: 3.9",
                   "Programming Language :: Python :: 3.10"],
      keywords='jax',
      packages=['flaxcrf'],
      python_requires='>=3.7')
