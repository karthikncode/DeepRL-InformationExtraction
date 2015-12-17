from distutils.core import setup
from distutils.sysconfig import get_python_lib
import os

setup(
	name='YahooBoss-Python',
	version='0.2.0',
	author='Constituent Voice',
	author_email='opensource@constituentvoice.com',
	packages=['yahooboss'],
	data_files=[],
	scripts=[],
	url='https://github.com/constituentvoice/YahooBoss-Python',
	license='BSD',
	description='Simple wrapper around the Yahoo Boss API (https://developer.yahoo.com/boss/search/)',
	long_description=open('README.rst').read(),
	install_requires=["oauth2","requests >= 1.0.0"]
)
