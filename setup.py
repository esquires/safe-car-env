from setuptools import find_packages, setup

setup(
    name="safe_car_env",
    version="0.0.1",
    author="Eric Squires",
    long_description="",
    description="",
    zip_safe=False,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy",
        "gym",
        "types-pyyaml",
        "unitpy",
        "pygame",
        "moviepy==1.0.3",
    ],
)
