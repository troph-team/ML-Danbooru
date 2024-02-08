from setuptools import setup, find_packages

setup(
    name='ml_danbooru_tagger',
    version='0.2.2',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision>=0.5.0',
        'loguru',
        'inplace_abn',
        'einops',
        'timm',
        'huggingface_hub',
        'scikit-learn',
        'xformers==0.0.22.post7',  # higher version requires torch 2.2
    ],
    entry_points={
        'console_scripts': [
            'ml-danbooru-tagger=demo_ca:main',
        ],
    },
)
