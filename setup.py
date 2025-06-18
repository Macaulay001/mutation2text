from setuptools import setup, find_packages

setup(
    name='mutation2text',
    version='0.1.0',
    description='A project to understand protein mutation impacts using LLMs and GCA/Resampler adapters.',
    author='Your Name/Team',
    author_email='your.email@example.com',
    # url='https://github.com/yourusername/mutation2text', # Optional: Project URL
    packages=find_packages(exclude=['tests*', 'notebooks*', 'output*']),
    install_requires=[
        # List core dependencies from requirements.txt, 
        # or read them dynamically from requirements.txt
        # Example:
        # 'torch>=2.0.0',
        # 'transformers>=4.35.0',
        # 'deepspeed>=0.10.0',
        # 'peft>=0.7.0',
        # 'scikit-learn>=1.1.0',
        # 'pyyaml>=6.0',
        # 'tensorboard>=2.10',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License', # Or your chosen license
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
    python_requires='>=3.8',
    # entry_points={
    #     'console_scripts': [
    #         'mutation2text_pretrain=scripts.train:main',
    #         'mutation2text_finetune=scripts.finetune:main',
    #     ],
    # },
) 