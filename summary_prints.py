import os
import torch
import csv
from misc import bcolors
from progress.bar import Bar
import time
import sys

class SummaryStore():
    def __init__(self):
        self.ite_mean = 0
        self.ep_mean = 0
        self.iters = 0
        self.ep_iters = 0

    def __call__(self, value):
        self.ite_mean = (self.ite_mean * self.iters + value)/(self.iters+1)
        self.iters+=1

        self.ep_mean = (self.ep_mean * self.ep_iters + value)/(self.ep_iters+1)
        self.ep_iters += 1

    def iter_mean(self):
        mean = self.ite_mean
        self.ite_mean = 0
        self.iters = 0
        return mean

    def epoch_mean(self):
        mean = self.ep_mean
        self.ite_mean = 0
        self.iters = 0
        self.ep_mean = 0
        self.ep_iters = 0
        return mean

class AverageStore(object):
    def __init__(self):
        self.reset()

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

    def __call__(self, val, n=1):
        self.update(val, n=1)

class SummaryPrint():
    def __init__(self, args, names, log_dir, log_name='train', color=bcolors.OKGREEN):
        self.names = names
        self.stores = [AverageStore() for name in names]
        
        #write arguments to log
        self.argslog = open(os.path.join(log_dir, 'args_log_'+log_name+'.txt'),'w')
        self.argslog.write(str(args))
        self.argslog.close()

        #csv log
        self.log = open(os.path.join(log_dir, log_name+'_log.csv'),'w')
        self.csv_logger=csv.writer(self.log, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        self.csv_logger.writerow(names)
        self.color=color

        self.format_str = '({0[0]}/{0[1]}) | Data: {0[2]:0.03f} | Batch: {0[3]:0.03f}'
        for i, name in enumerate(names):
            self.format_str += ' |  %s: {0[%d]:0.5f}'%(name, i+4)
        self.data_time = AverageStore()
        self.batch_time = AverageStore()

    def __del__(self):
        self.close()

    def __call__(self, values):
        for store, value in zip(self.stores, values):
            store(value)

    def reset(self):
        self.data_time.reset()
        self.batch_time.reset()
        for store in self.stores:
            store.reset()

    def start_epoch(self, epoch, iters):
        self.iters = iters
        self.epoch = epoch
        #start bar and set init time
        sys.stdout.write(self.color)
        sys.stdout.flush()

        self.bar = Bar('epoch: %d'%epoch, max=iters)
        self.time = time.time()

        self.reset()

    def start_iter(self, i):
        #update time for data
        t = time.time()
        self.data_time.update(t-self.time)
        self.time = t

    def end_iter(self, i, values):
        #compute times and update bar
        t = time.time()
        self.batch_time.update(t-self.time)
        self.time = t

        #update stores
        for store, value in zip(self.stores, values):
            store.update(value)
        #update print output
        self.bar.suffix = self.format_str.format([i+1, self.iters, self.data_time.avg, self.batch_time.avg] + [store.avg for store in self.stores])
        self.bar.next()
        
    def end_epoch(self, epoch_metrics={}):
        #end the epoch, write averages to the log
        self.bar.finish()
        sys.stdout.write(bcolors.ENDC)
        sys.stdout.flush()

        #write to log
        store_list = [self.epoch] + [store.avg for store in self.stores] + [value for name,value in epoch_metrics.items()]
        self.csv_logger.writerow(store_list)

        if epoch_metrics:
            print(epoch_metrics)

        return store_list

    def close(self):
        self.log.close()
        #adding just to make sure we go back to white
        sys.stdout.write(bcolors.ENDC)
        sys.stdout.flush()
