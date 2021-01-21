import setuptools
import glob

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="powerful-benchmarker",
    version="0.9.33",
    author="Kevin Musgrave",
    description="A PyTorch library for benchmarking deep metric learning. It's powerful.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KevinMusgrave/powerful-benchmarker",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    package_data = {'src/powerful_benchmarker': glob.glob("configs/**/*.yaml", recursive=True)},
    include_package_data=True,
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
          'easy-module-attribute-getter == 0.9.41',
          'record-keeper == 0.9.29',
          'tensorboard',
          'matplotlib',
          'pretrainedmodels',
          'pytorch-metric-learning == 0.9.92',
          'pandas',
          'ax-platform',
          'faiss-gpu',
          'gdown >= 3.12.0',
    ],
)
