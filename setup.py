from setuptools import setup, find_packages

setup(
    name='ml_danbooru_tagger',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'torch>=1.12',
        'torchvision>=0.5.0',
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
