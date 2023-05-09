from pathlib import Path
import numpy as np
from matplotlib.colors import ListedColormap
from my_utils.pytorch import tensor2array


class Callback():

    def __init__(self, nr_epochs, data_class = {}, every_n_steps=None, every_n_epochs=None, log_dir=None, **kwargs):

        self.every_n_steps = every_n_steps
        self.every_n_epochs = every_n_epochs
        self.nr_epochs = nr_epochs

        if log_dir is not None:
            self.log_dir = Path(log_dir)
            self.log_dir.mkdir(exist_ok=True, parents=True)

        # List of string labels and corresponding codes, and dict to make the mappings between codes, strings and ordered integers for the extended labels.
        
        if isinstance(data_class, str):
            data_class_name = data_class
        else:
            data_class_name = data_class.get('dataset', data_class.get('root'))
    
        if 'mass' in data_class_name:
            self.label_codes = np.arange(6)
            self.string_labels =  ['NaN', 'N1', 'N2', 'N3', 'Wake', 'REM']

        elif 'random_walk_freq_sinusoids' in data_class_name:
            self.label_codes = np.inf
            self.string_labels = np.inf

        elif 'librispeech' in data_class_name:  
            self.label_codes = np.arange(data_class['speaker_classes'])
            self.string_labels = [str(i) for i in range(data_class['speaker_classes'])]


    def return_color_list(self, label_format='standard'):
        """ Returns a pyplot colormap to be used by the tsne plot dependent on how we want to color the dots.
        Args:
            label_format (str, optional): The type of label set to be used. One from: {standard, transition_labels, transition_labels_short, 'toy_<nr_classes>'}. Defaults to 'standard'.

        Returns:
            pyplot colormap
        """

        colors = []
        colors.append(np.divide([255,255,255], 255)) #NaN
        colors.append(np.array([51, 102,255])/255) #N1
        colors.append(np.array([230, 138, 0])/255) #N2
        colors.append(np.array([0.5,0.1,0.4])) #N3
        colors.append(np.array([219,112,147])/255) #Wake
        colors.append(np.array([34,139,34])/255) #REM
        
        if label_format.lower() == 'standard':
            return colors

        elif label_format.lower() == 'standard_wo_nan':
            return colors[1:]

    def return_cmap(self, nr_classes=6, zero_class_is_nan_class=True, label_format='standard'):
        """ Returns a pyplot colormap to be used by the tsne plot dependent on how we want to color the dots.
        Args:
            nr_classes (int, optional)
            zero_class_is_nan(bool, optional)
            label_format (str, optional): The type of label set to be used. One from: {standard, transition_labels, transition_labels_short}. Defaults to 'standard'.

        Returns:
            pyplot colormap
        """
        if nr_classes > 6 and nr_classes <= 10:
            return 'tab10' 
        elif nr_classes > 10 and nr_classes <= 20:
            return 'tab20'
        elif nr_classes > 20:
            return 'gnuplot2_r'
        else:
            label_list = self.return_color_list(label_format)
            if nr_classes < 6:
                if zero_class_is_nan_class:
                    label_list = label_list[:nr_classes]
                else:
                    label_list = label_list[1:nr_classes+1]
        return ListedColormap(label_list)

    def convert_onehot_to_scalar(self, label, soft_pred):        
        if label.shape[-1] == soft_pred.shape[-1] and len(label.shape)>1 and len(soft_pred.shape) > 1:
            scalar_label = np.argmax(tensor2array(label), -1)
            scalar_pred = np.argmax(tensor2array(soft_pred), -1)
        elif label.shape[-1] != soft_pred.shape[-1]:
            if len(label.shape) > 1:
                scalar_label = np.argmax(tensor2array(label), -1)
                scalar_pred = tensor2array(soft_pred)
            else:
                scalar_pred = np.argmax(tensor2array(soft_pred), -1)
                scalar_label = tensor2array(label)
        else:
            scalar_pred = tensor2array(soft_pred)
            scalar_label = tensor2array(label)
            assert scalar_pred.shape == scalar_label.shape

        assert scalar_label.shape == scalar_pred.shape
        return scalar_label, scalar_pred

        
    def on_train_begin(self, **kwargs):
        return

    def on_train_end(self, **kwargs):
        return

    def check_epoch(self, epoch):
        if self.every_n_epochs is None:
            return True
        elif self.every_n_epochs == 0:
            return False
        elif ((epoch % self.every_n_epochs == 0)) or (epoch==self.nr_epochs-1):
            return True
        else:
            return False    

    def check_step(self, step):
        if self.every_n_steps is None:
            return False
        elif (step % self.every_n_steps == 0) and step > 0:
            return True
        else:
            return False

    def on_epoch_begin(self, epoch, **kwargs):
        return

    def on_epoch_end(self, epoch, **kwargs):
        return

    def on_step_begin(self, step, **kwargs):
        return

    def on_step_end(self, step, **kwargs):
        return

    def at_inference(self, epoch, **kwargs):
        return

