from setuptools import setup, find_packages

# Read the content of your README.md file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="eilof",  
    version="1.0.4",  
    description="Efficient implementation of Incremental Local Outlier Factor (EILOF).",
    long_description=long_description,  # Link README.md content
    long_description_content_type="text/markdown",  # Specify Markdown format for PyPI
    author="Rui Hu, Luc (Zhilu) Chen, Yiwei Wang",  
    author_email="rui.hu@csusb.edu, luchen@g.harvard.edu, yiweiw@ucr.edu",  
    url="https://github.com/Lucchh/eilof",  
    packages=find_packages(), 
    install_requires=[
        "numpy>=1.19.5,<2.0.0",         
        "scipy>=1.6.0,<2.0.0",       
        "matplotlib>=3.3.4,<4.0.0",   
        "seaborn>=0.9.0,<1.0.0",    
        "scikit-learn>=1.0.0,<2.0.0",
    ],
    python_requires=">=3.8",  
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)