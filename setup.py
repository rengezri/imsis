from setuptools import setup, find_packages

setup(name='imsis',
      version='1.0',
      description='imsis',
      classifiers=[
        'Development Status :: 0.1',
        'Programming Language :: Python :: 3.x',
      ],
      keywords='simple imaging and analysis in Python',
      author='AA',
      packages=find_packages(),
      install_requires=[
          'markdown',
      ],
      include_package_data=True,
      zip_safe=False)
