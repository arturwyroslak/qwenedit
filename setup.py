#!/usr/bin/env python3
"""
Setup script for Qwen Image Edit Hairstyle Transfer.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="qwenedit",
    version="1.0.0",
    author="Artur Wyroslak",
    author_email="95839215+arturwyroslak@users.noreply.github.com",
    description="Qwen Image Edit - CPU-optimized hairstyle transfer with Gradio UI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arturwyroslak/qwenedit",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "qwenedit=cli:main",
        ],
    },
)