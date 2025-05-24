from setuptools import setup, find_packages

setup(
    name='ai-model-zoo-pro',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'torch',
        'tensorflow',
        'scikit-learn',
        'Flask'
    ],
    entry_points={
        'console_scripts': [
            'run-api=api.app:main',
            'run-webapp=webapp.app:main',
        ],
    },
    python_requires='>=3.9',
    description='A collection of various AI models with a unified API and web interface.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/ai-model-zoo-pro',
    author='Your Name',
    author_email='your.email@example.com',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)