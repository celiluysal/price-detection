import matplotlib.pyplot as plt
import os
from os.path import join
from datetime import date
from datetime import datetime

def time_stamp():
    today = date.today()
    d1 = today.strftime("%d/%m/%Y")

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")

    print(d1,current_time)
    return now

def draw_loss_graph(epoch_size, run_loss_list, val_loss_list, name):
    save_file_name = "../data/graphs/" 
    
    if not os.path.exists(save_file_name):
        os.mkdir(save_file_name)

    epoch_list = list()
    for i in range(epoch_size):
        epoch_list.append(i)

    fig = plt.figure()
    fig.add_subplot(221)
    plt.plot(epoch_list, run_loss_list, label = "training loss", color="C0") 
    plt.tick_params(labelsize=7)
    plt.title("training loss", fontsize=7)

    fig.add_subplot(222)
    plt.plot(epoch_list, val_loss_list, label = "validation loss", color="C1") 
    plt.tick_params(labelsize=7)
    plt.title("validation loss", fontsize=7)

    fig.add_subplot(212)
    plt.plot(epoch_list, run_loss_list, label = "training loss", color="C0") 
    plt.plot(epoch_list, val_loss_list, label = "validation loss", color="C1") 
    plt.tick_params(labelsize=7)
    plt.legend()

    plt.savefig(join(save_file_name, name +".png")) 

def norm(raw):
    return [float(i)/sum(raw) for i in raw]
