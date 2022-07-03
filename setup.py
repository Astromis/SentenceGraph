import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sentence_graph",
    version="0.1.0",
    author="Igor Buyanov",
    author_email="buyanov.igor.o@yandex.ru",
    description="A handy wrapper for UDPipe",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Astromis/SentenceGraph",
    project_urls={
        "Bug Tracker": "https://github.com/Astromis/SentenceGraph/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=[
        'conllu==1.3.1',
        'ufal.udpipe==1.2.0.3',
        'networkx==2.1',
        'matplotlib==2.2.3'
    ],
    include_package_data=True,
)