from setuptools import setup, find_packages

setup(
    name="eilof",  # Name of your package
    version="1.0.0",  # Version number
    description="Efficient implementation of Incremental Local Outlier Factor (EILOF).",
    author="Rui Hu, Luc (Zhilu) Chen, Yiwei Wang",  # Replace with the authors' names
    author_email="rui.hu@csusb.edu, luchen@g.harvard.edu, yiweiw@ucr.edu",  # Replace with authors' emails
    url="https://github.com/yourusername/eilof",  # Replace with your GitHub repository URL
    packages=find_packages(),  # Automatically finds your package in the directory
    install_requires=[
        "numpy>=2.0.0,<3.0.0",         
        "scipy>=1.13.0,<2.0.0",       
        "matplotlib>=3.9.0,<4.0.0",   
        "seaborn>=0.13.0,<1.0.0",     
        "scikit-learn>=1.6.0,<2.0.0", 
    ],
    python_requires=">=3.8",  # Specify the required Python version
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)