from distutils.core import setup
import setuptools

setup(
  name = 'imsis',
  packages = ['imsis'],
  version = '1.1.9',
  license='MIT',
  description = 'image analysis in Python',
  author = 'rengezri',
  author_email = 'rengezri@gmail.com',
  url = 'https://github.com/rengezri',
  download_url = 'https://github.com/rengezri/imsis/archive/refs/tags/v1.1.9.tar.gz',
  keywords = ['image analysis', 'dialogs', 'batch processing'],   # Keywords that define your package best
  install_requires=['markdown'],
  classifiers=[
    'Topic :: Scientific/Engineering :: Image Processing',
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Programming Language :: Python :: 3.9',
    'Programming Language :: Python :: 3.10',
  ],
)
