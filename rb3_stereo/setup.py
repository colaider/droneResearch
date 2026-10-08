from glob import glob

from setuptools import setup

package_name = "rb3_stereo"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md", "QUICKSTART.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.py")),
        ("share/" + package_name + "/config", glob("config/*.yaml") + glob("config/*.xml")),
        ("share/" + package_name + "/docs", glob("docs/*.md")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="domku",
    maintainer_email="domkut@seznam.cz",
    description="OV9282 stereo camera node for the RB3 Gen 2 vision mezzanine",
    license="GPL-2.0-only",
    entry_points={
        "console_scripts": [
            "stereo_camera = rb3_stereo.node:main",
            "viewer = rb3_stereo.viewer:main",
        ],
    },
)
