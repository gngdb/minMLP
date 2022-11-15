from setuptools import setup

setup(name='minMLP',
      version='0.0.1',
      author='Gavia Gray',
      packages=['minmlp'],
      description='A PyTorch implementation of a causal MLP-Mixer',
      license='MIT',
      install_requires=[
            'torch',
            'einops'
      ],
)
