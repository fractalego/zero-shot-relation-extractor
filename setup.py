from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='zero-shot-re',
      version='0.0.4',
      url='http://github.com/fractalego/zero-shot-relation-extractor',
      author='Alberto Cetoli',
      author_email='alberto@nlulite.com',
      description="A zero-shot relation extractor",
      long_description=long_description,
      long_description_content_type="text/markdown",
      packages=['zero_shot_re',
                ],
      install_requires=[
          'numpy==1.19.1',
          'transformers==4.9.2',
          'torch==1.9.0',
          'jupyterlab==2.2.9',
      ],
      classifiers=[
          'License :: OSI Approved :: MIT License',
      ],
      include_package_data=True,
      zip_safe=False)
