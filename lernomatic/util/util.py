"""
UTIL
"""

# debug
#from pudb import set_trace; set_trace()

class AverageMeter(object):
    """
    Keep track of most recent, average, sum, and
    count of a metric
    """
    def __init__(self):
        self.reset()

    def __repr__(self):
        return 'AverageMeter %d' % self.count

    def __str__(self):
        s = []
        s.append('AverageMeter\n')
        s.append('val: %.4f avg: %.4f sum: %.4f count: %d\n' %\
                 (self.val, self.avg, self.sum, self.count))
        return ''.join(s)

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
