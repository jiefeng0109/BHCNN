# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 16:15:16 2019

@author: EGGSHELL
"""

from main import HDCNN

if __name__ == "__main__":

    data_name = 'PaviaU'
    model = HDCNN(data_name,30)
    oa=0
    for i in range(10):
        model.train_test()
        oa=model._oa+oa
        model.plot(i)
    oa=oa/10
    print(oa)
