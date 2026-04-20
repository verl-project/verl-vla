# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

from setuptools import find_packages, setup

THIS_DIRECTORY = Path(__file__).parent


def read_version() -> str:
    version_ns = {}
    version_file = THIS_DIRECTORY / "src" / "verl_vla" / "version.py"
    exec(version_file.read_text(), version_ns)
    return version_ns["__version__"]


def read_requirements(filename: str) -> list[str]:
    requirements = []
    for line in (THIS_DIRECTORY / filename).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)
    return requirements


INSTALL_REQUIRES = read_requirements("requirements.txt")
TEST_REQUIRES = read_requirements("requirements-test.txt")

extras_require = {
    "test": TEST_REQUIRES,
}

long_description = (THIS_DIRECTORY / "README.md").read_text()

setup(
    name="verl-vla",
    version=read_version(),
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    url="https://github.com/verl-project/verl-vla",
    license="Apache 2.0",
    author="Bytedance",
    description="VLA training framework of Volcano Engine",
    install_requires=INSTALL_REQUIRES,
    extras_require=extras_require,
    package_data={
        "verl_vla": [
            "configs/**/*.yaml",
        ],
    },
    include_package_data=True,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
