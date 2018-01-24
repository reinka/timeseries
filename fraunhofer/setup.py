from setuptools import setup


setup(
    name='fraunhofer',
    description='Custom scripts for time series analysis',
    author='Andrei Poehlmann',
    author_email='andrei.poehlmann90@gmail.com',
    license='AGPL-3.0',
    install_requires=['pandas',
                      'scikit-learn',
                      'matplotlib',
                      'seaborn',
                      'tensorflow',
                      'keras'
                      ],
    packages=['fraunhofer'],
    zip_safe=False,
)
