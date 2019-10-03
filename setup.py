from setuptools import setup, find_packages


setup(
    name='htx',
    version='0.0.1',
    description='Interface for using machine learning models within Heartex platform',
    author='Nikolai Liubimov',
    author_email='nik@heartex.net',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'Flask==1.0.2',
        'attrs==19.1.0',
        'xmljson==0.2.0',
        'lxml==4.3.3',
        'rq==1.0',
        'requests==2.22.0'
    ],
)
