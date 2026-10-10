from glob import glob

from setuptools import setup

package_name = "rb3_control"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/docs", glob("docs/*.md")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="domku",
    maintainer_email="domkut@seznam.cz",
    description="Betaflight MSP-over-USB bridge and controller for the RB3 Gen 2",
    license="GPL-2.0-only",
    entry_points={
        "console_scripts": [
            "flight_controller = rb3_control.node:main",
            "fc_check = rb3_control.check:main",
            "teleop = rb3_control.teleop:main",
        ],
    },
)
