import sys


class Buffer:
    def __init__(self, size):
        self.mybuffer = {}
        for i in range(size):
            self.mybuffer[i] = {
                'prob': -sys.maxsize - 1,
                'pre_state': 'NA',
                'from_k_th': -1
            }

    def push(self, probability, previous_state, from_k_th):
        for i in range(self.getSize()):
            # if probability larger than the previous prob, then replace this dic with ith dic then shift the buffer forward
            if probability > self.mybuffer[i]['prob']:
                # move backward
                for j in range(self.getSize() - 1, i, -1):
                    self.mybuffer[j] = self.mybuffer[j - 1]
                # replace
                self.mybuffer[i] = {
                    'prob': probability,
                    'pre_state': previous_state,
                    'from_k_th': from_k_th
                }
                break

    def getBuffer(self):
        return self.mybuffer

    def getSize(self):
        return len(self.mybuffer)

    def getProb(self, k):
        return self.mybuffer[k]['prob']

    def getPre_state(self, k):
        return self.mybuffer[k]['pre_state']

    def __str__(self):
        return self.mybuffer