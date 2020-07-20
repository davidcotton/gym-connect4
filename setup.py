from setuptools import setup

setup(
    name='gym_connect4',
    version='0.1.0',
    description='2-player Connect4 OpenAI Gym environment',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=['gym>=0.9.6', 'numpy'],
    extras_require={
        'dev': ['pytest', 'pytest-cov', 'pytest-sugar']
    },
)
