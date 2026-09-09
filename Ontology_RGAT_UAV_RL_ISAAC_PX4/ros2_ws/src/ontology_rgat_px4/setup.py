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
    maintainer="CICS2026 research workspace",
    maintainer_email="research@example.invalid",
    description="Safe external gateway between MATLAB and PX4.",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            "ros2_gateway = ontology_rgat_px4.ros2_gateway:main",
            "mavlink_gateway = ontology_rgat_px4.mavlink_gateway:main",
        ],
    },
)

