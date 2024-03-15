from setuptools import setup, find_packages

setup(
    name='slp_generator',
    version='0.1.0',
    author='evgeny_genov',
    author_email='your-email@example.com',
    description='A package for generating synthetic load profiles',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-github-username/slp_generator',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.6',
    install_requires=[
        'pandas==1.4.3',
        'numpy==1.23.1',
        'scikit-learn==1.1.1',
        'kmodes==0.12.1',
    ],
)