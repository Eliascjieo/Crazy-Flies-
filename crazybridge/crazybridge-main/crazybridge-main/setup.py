from setuptools import find_packages, setup

package_name = 'crazybridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/crazybridge.launch.py', 'launch/crazyoptitrack.launch.py', 'launch/controller_test.launch.py']),
        ('share/' + package_name + '/config', ['config/pid.conf']),
    ],
    install_requires=['setuptools', 'cflib', 'textual', 'rerun-sdk', 'numpy'],
    zip_safe=True,
    maintainer='Kevin Martinez',
    maintainer_email='kevin@nuclea.solutions',
    description='Python ROS2 port of the forerunner2 crazybridge module.',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'crazybridge = crazybridge.crazybridge_node:main',
            'rerun = crazybridge.rerun:main',
            'crazybridge_tui = crazybridge.tui:main',
            'controller_test = crazybridge.controller_test:main',
        ],
    },
)
