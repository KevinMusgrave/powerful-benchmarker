import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="powerful-benchmarker",
    version="0.9.0",
    author="Kevin Musgrave",
    author_email="tkm45@cornell.edu",
    description="A highly-configurable tool that enables thorough evaluation of deep metric learning algorithms. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KevinMusgrave/powerful-benchmarker",
    packages=setuptools.find_packages(include=["powerful_benchmarker.*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.0',
    install_requires=[
          'numpy',
          'scikit-learn',
          'torch',
          'torchvision',
          'easy-module-attribute-getter',
          'record-keeper',
          'tensorboard',
          'matplotlib',
          'pretrainedmodels',
          'pytorch-metric-learning',
          'pandas',
          'ax-platform',
          'faiss-gpu'
    ],
)