"""
Allow non-daemonic pools that can spawn child processes.
This is a hack from: https://stackoverflow.com/questions/6974695/python-process-pool-non-daemonic
In the future one might switch to celery for a clean solution
"""

import multiprocessing
from multiprocessing.pool import Pool

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class MyPool(Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(MyPool, self).__init__(*args, **kwargs)
