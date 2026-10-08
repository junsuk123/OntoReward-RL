from glob import glob

from setuptools import find_packages, setup

package_name = "ontology_rgat_landing"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="CICS2026 research workspace",
    maintainer_email="research@example.invalid",
    description="Minimal-observation Ontology -> R-GAT landing pipeline nodes.",
    license="BSD-3-Clause",
    entry_points={
        "console_scripts": [
            f"{name} = {package_name}.{name}:main" for name in (
                "aruco_pad_detector", "px4_localization_bridge",
                "observation_assembler", "ontology_node", "rgat_policy_node",
                "safety_supervisor_node", "sim_clock_bridge")
        ],
    },
)
