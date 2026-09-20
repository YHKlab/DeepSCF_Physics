import os
from omegaconf import OmegaConf




def summary_model(args, num_parameter):

    with open('./summary.txt','w') as f:
        dicts = OmegaConf.to_object(args)
        for k, i in dicts.items():
            f.write(f'{k}: {i}\n')
        f.write(f'Number of parameters {num_parameter}\n')

class logger():

    def __init__(self, path = './log.txt', mode = 'w'):

        self.path = path
        if not os.path.isfile(path):
            f = open(path, 'w')
            f.close()
        else:
            f = open(path, 'w')
            f.close()
        self.log = ""

    def update(self, **kwags):

        if 'path' in kwags:
            path = kwags['path']
            loss =  kwags['loss']
            txt = f"Path: {path} loss = {loss} \n"

        elif 'epoch' in kwags:
            epoch = kwags['epoch']
            train_loss =  kwags['train_loss']
            test_loss = kwags['test_loss']
            test_acc = kwags['test_acc']

            txt = f"Epoch {epoch}: train loss = {train_loss} \t"

            if type(test_loss) == list:
                for i in range(len(test_loss)):
                    txt +=f"test loss{i+1} = {test_loss[i]} "
                txt += "\t"
            else:
                txt += f"test loss = {test_loss}\t"

            if type(test_acc) == list:
                for i in range(len(test_acc)):
                    txt +=f"accuracy{i+1} = {test_acc[i]} "
                txt += "\n"
            else:
                txt += f"accuracy = {test_acc}\n"

        self.log += txt

    def save(self):

        f = open(f'{self.path}', 'a')
        txt = self.log
        f.write(txt)
        f.close()

        # refresh the log
        self.log = ""
