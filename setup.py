from setuptools import setup, find_packages

setup(
    name='ml_danbooru_tagger',
    version='0.2.2',
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'loguru',
        'inplace_abn',
        'einops',
        'timm',
        'huggingface_hub',
        'scikit-learn',
        'xformers',
    ],
    entry_points={
        'console_scripts': [
            'ml-danbooru-tagger=demo_ca:main',
        ],
    },
)
