# Copyright (c) 2026 Jun Suk Kim (김준석)
# Maintainer: Jun Suk Kim (김준석) <junsuk55944587@gmail.com>
# SPDX-License-Identifier: Apache-2.0

from setuptools import find_packages, setup

package_name = "ontology_rgat_px4"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    author="Jun Suk Kim",
    author_email="junsuk55944587@gmail.com",
    maintainer="Jun Suk Kim",
    maintainer_email="junsuk55944587@gmail.com",
    description="Safe external gateway between MATLAB and PX4.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "ros2_gateway = ontology_rgat_px4.ros2_gateway:main",
            "mavlink_gateway = ontology_rgat_px4.mavlink_gateway:main",
        ],
    },
)

