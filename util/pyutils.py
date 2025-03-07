import sys
import time
import datetime

#实现将输出信息记入日志
class Logger(object):
    def __init__(self, outfile,log_all = True):
        self.terminal = sys.stdout
        self.log = open(outfile, "w")
        self.log_all = log_all  # 是否记录所有输出到日志文件
        sys.stdout = self

    def write(self, message):
        if self.log_all: #判别是否全部存入
            self.log.write(message)
        self.terminal.write(message) # 所有输出都显示在终端


    def flush(self):
        self.terminal.flush()


class AverageMeter:
    def __init__(self, *keys):
        self.__data = dict()
        for k in keys:
            self.__data[k] = [0.0, 0]

    def add(self, dict):
        for k, v in dict.items():
            self.__data[k][0] += v
            self.__data[k][1] += 1
    #因加ReCAM修改于2023.11.17
    # def get(self, *keys):
    #     if len(keys) == 1:
    #         return self.__data[keys[0]][0] / self.__data[keys[0]][1]
    #     else:
    #         v_list = [self.__data[k][0] / self.__data[k][1] for k in keys]
    #         return tuple(v_list)
    def get(self, *keys):
        if len(keys) == 1:
            count = self.__data[keys[0]][1]
            if count == 0:
                return 0.0  # 如果计数为零，返回默认值 0.0，避免除以零错误
            return self.__data[keys[0]][0] / count
        else:
            v_list = [self.__data[k][0] / self.__data[k][1] if self.__data[k][1] != 0 else 0.0 for k in keys]
            return tuple(v_list)

    def pop(self, key=None):
        if key is None:
            for k in self.__data.keys():
                self.__data[k] = [0.0, 0]
        else:
            v = self.get(key)
            self.__data[key] = [0.0, 0]
            return v


class Timer:
    def __init__(self, starting_msg=None):
        self.start = time.time()
        self.stage_start = self.start
        if starting_msg is not None:
            print(starting_msg, time.ctime(time.time()))

    def update_progress(self, progress):
        self.elapsed = time.time() - self.start
        self.est_total = self.elapsed / progress
        self.est_remaining = self.est_total - self.elapsed
        self.est_finish = int(self.start + self.est_total)

    def str_est_finish(self):
        return str(time.ctime(self.est_finish))

    def get_stage_elapsed(self):
        return time.time() - self.stage_start

    def reset_stage(self):
        self.stage_start = time.time()

    def get_est_remain(self):
        return str(datetime.timedelta(seconds=int(self.est_remaining)))
    
def print_progress_bar(iteration, total, bar_length=50):
    progress = (iteration / total)
    arrow = '=' * int(round(bar_length * progress))
    spaces = ' ' * (bar_length - len(arrow))
    print(f'[{arrow}{spaces}] {int(progress * 100)}%', end='\r')
