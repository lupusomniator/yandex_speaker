import time
class Profiler(object):
    
    def __init__(self):
        self.n=0
        self.maxtime=0
        self.sum_time=0
        
    def __enter__(self):
        self.n+=1
        self.start_time = time.time()
        
    def __exit__(self,type = None, value = None, traceback = None):
        self.end_time = time.time()
        timem = self.end_time - self.start_time
        self.sum_time += timem
        if timem>self.maxtime:
            self.maxtime=timem
        print("Затраченное время:", timem)
        
    def stat(self):
        print("Количество вызовов:",self.n)
        print("Суммарное время:", self.sum_time)
        print("Среднее время:", self.sum_time/self.n)
        print("Максимальное время:", self.maxtime)