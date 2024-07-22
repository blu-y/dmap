from setuptools import setup
import glob

package_name = 'dmap'

setup(
    name=package_name,
    version='0.2.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name, glob.glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='blu-y',
    maintainer_email='a_o@kakao.com',
    description='On-device Object-aware Mapping',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dmap = dmap.dmap_node:main',
            'map_server = dmap.map_server:main',
            'img_saver = dmap.img_saver:main',
            'model_downloader = dmap.model_downloader:main',
        ],
    },
)
