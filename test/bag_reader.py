#todo



import pip
import rosbag

pip.main(['install', 'rosbag'])

class BagReader:
    def __init__(self, bag_file):
        self.bag = rosbag.Bag(bag_file)

