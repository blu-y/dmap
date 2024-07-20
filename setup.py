from setuptools import find_packages, setup

package_name = 'dmap'

setup(
    name=package_name,
    version='0.2.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='blu-y',
    maintainer_email='a_o@kakao.com',
    description='On-device Object-aware Mapping',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
        ],
    },
)
