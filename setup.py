from setuptools import setup, find_packages

setup(
    name="ga_error_sources",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "ipywidgets==8.0.4",
        "scipy==1.6.2",
        "seaborn==0.13.2",
        "numpy==1.21.5",
        "ipython==7.34.0",
        "pandas==1.5.2",
        "matplotlib==3.6.2",
        "jupyter_client==7.4.8",
        "prettytable-3.11.0"
    ],
    author="José Antonio Fiorote",
    author_email = "jafiorote@gmail.com",
    license = "LGPLv3+",
    description="Descrição do pacote",
    classifiers = [
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",

        "License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)",
        "Programming Language :: Python :: 3",

        "Natural Language :: English",
        "Operating System :: POSIX :: Linux",
    ],
    keywords = "mutual information, sequence alignment, genetic algorithm, protein modeling",
    url="https://github.com/seu_usuario/seu_repositorio",
)