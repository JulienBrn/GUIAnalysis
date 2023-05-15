from distutils.core import setup


setup(
    name='analysisGUI',
    packages=['analysisGUI'],
    package_data={
        "analysisGUI.ui": ["*.gif"],
    },
    version='0.1',
    license='MIT',
    description = 'analysisGUI for python analysis',
    description_file = "README.md",
    author="Julien Braine",
    author_email='julienbraine@yahoo.fr',
    url='https://github.com/JulienBrn/analysisGUI',
    download_url = 'https://github.com/JulienBrn/analysisGUI.git',
    package_dir={'': 'src'},
    keywords=['python', 'dataframe'],
    install_requires=['pandas', 'matplotlib', 'PyQt5', "sklearn", "scikit-learn", "scipy", "numpy", "tqdm", "beautifullogger", "statsmodels"],
)
