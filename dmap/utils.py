def make_utc(filename):
    import os
    import time
    filename = os.path.basename(filename).split('.')[0]
    yymmdd, hhmmss, ms = filename.split('_')
    utc = time.mktime(time.strptime(yymmdd+hhmmss, '%y%m%d%H%M%S'))
    return float(str(int(utc)) + '.' + ms)

from ament_index_python.packages import get_package_prefix, PackageNotFoundError

def find_ros2_package_src(package_name):
    try: package_path = get_package_prefix(package_name)
    except PackageNotFoundError: return None
    return package_path + '/../../src/' + package_name