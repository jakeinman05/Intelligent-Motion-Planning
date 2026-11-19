import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/jakei/Intelligent-Motion-Planning/ros2_ws/install/idsc_runtime'
