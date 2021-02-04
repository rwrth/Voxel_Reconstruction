import os
import datetime
from shutil import copyfile
import torch
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class log:
    def __init__(self):
        self.path = os.getcwd()
        i = 0
        while True:
            name = "Optimierung/LONG/Train_"+str(i)
            try:
                os.mkdir( self.path +"/" + name)
                os.mkdir( self.path +"/" + name + "/models")
            except OSError:
                pass
            else:
                print ("Successfully created the directory %s%s " % (self.path,name))
                break
            i+=1
        self.dir = self.path +"/"+ str(name)

    def give_dir(self):
        return self.dir

    def writelog_file_start(self, comment, path_dataset, path_validataset ,batch_size_t, batch_size_v,trainable_params, learning_rate, sigma, Dropout):
        copyfile(self.path + "/model_bs.py",self.dir + "/model_bs.py")
        file = open(self.dir + "/log.txt", "a")
        file.write("Training neuronal network" +"\n")
        file.write("For network architecture look at model.py" +"\n")
        file.write(comment +"\n")
        file.write("start time " + str(datetime.datetime.now()) +"\n")
        file.write("Train Data " + path_dataset +"\n")
        file.write("Train Data " + path_validataset +"\n")
        file.write("Batchsize: training: " + str(batch_size_t) + " Validation: " + str(batch_size_v)  +"\n" )
        file.write("Trainable Parameters : " + str(trainable_params) +"\n")
        file.write("Learning rate " + str(learning_rate) +"\n")
        file.write("Loss Function: My Loss Ohne Collinear"+ "sigma= "+ str(sigma)+"\n")
        file.write("Dropout: "+  str(Dropout)+"\n")

    def writelog_file_end(self, epochs, Loss, vali_Loss):
        file = open(self.dir + "/log.txt", "a")
        file.write("\n")
        file.write("Epochs " + str(epochs) +"\n")
        file.write("Loss " + str(Loss) +"\n")
        file.write("Validation Loss " + str(vali_Loss) +"\n")
        file.write("end time " + str(datetime.datetime.now()) +"\n")
        file.close()

    def save_plots(self, epochs,n, all_loss, all_m_loss, all_Vloss, all_m_Vloss, dif_all,dif_std_all, richtig):
        #plt.rcParams.update({'font.size': 20})
        x = np.arange(len(all_loss)/epochs/2,len(all_loss)+1, len(all_loss)/epochs)
        x_long = np.arange(0,len(all_loss),1)
        plt.title("Loss vs Validation Loss")
        plt.plot(x_long, all_Vloss, alpha = 0.5, color = "blue", label = "Validation loss")
        plt.plot(x_long, all_loss, alpha = 0.5, color = "orange", label = "Trainings loss")
        plt.plot(x, all_m_Vloss, '--', color = "blue",  label = "mean Validation loss")
        plt.plot(x, all_m_loss,'--', color = "orange", label = "mean Trainings loss")
        plt.legend()
        plt.xlabel("batches")
        plt.ylabel("loss")
        #plt.yscale("log")
        plt.grid(True)
        plt.savefig(self.dir + "/losses.pdf", format = "pdf")
        plt.close()

        plt.title("mittlerer Fehler auf Gesamtphotonzahl")
        x = np.arange(0,epochs,1)
        plt.errorbar(x, dif_all, yerr = dif_std_all, fmt = ".", label = "mittlerer Fehler auf Gesamtphotonzahl")
        #plt.yscale("log")
        plt.xlabel("Epochs")
        plt.ylabel("mittlerer Fehler in Prozent")
        plt.grid(True)
        plt.savefig(self.dir + "/photonzahl.pdf", format = "pdf")
        plt.close()

        plt.title("richtige Füllung")
        richtig = np.array(richtig)
        x = np.arange(0,epochs,1)
        plt.errorbar(x, richtig[:,0], yerr = richtig[:,1], fmt = "o", label = "richtig gefüllt")
        plt.errorbar(x, richtig[:,2], yerr = richtig[:,3], fmt = "o", label = "richtig leer")
        plt.grid(True)
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel("mittlerer Fehler in Prozent")
        plt.grid(True)
        plt.savefig(self.dir + "/richtig_verlauf.pdf", format = "pdf")
        plt.close()


def save_model(state, path, epoch):
    torch.save(state, path + "/model_"+ str(epoch) +".pth.tar")
