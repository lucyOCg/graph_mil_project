#simplified training script for GNN


#%% imports 
import torch 
from torch_geometric.data import DataLoader
from sklearn.metrics import confusion_matrix, f1_score, \
    accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import auc as calc_auc
from sklearn.preprocessing import label_binarize
import numpy as np
from tqdm import tqdm
from dataset import PatchDataset
#from model import GNN
#from gnn_model_2 import GCN
from gnn_model import CLAM_SB
#import mlflow.pytorch
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import os
 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#train_dataset = PatchDataset(root="/nobackup/projects/bdlds05/lucyg/graph_project/20x_data_res18_histo/", filename = 'graph_patch_raw_resnet18_20x_train.csv')
#test_dataset = PatchDataset(root="/nobackup/projects/bdlds05/lucyg/graph_project/20x_data_res18_histo/", filename="graph_patch_raw_resnet18_20x_test.csv", test=True)
#val_dataset = PatchDataset(root="/nobackup/projects/bdlds05/lucyg/graph_project/20x_data_res18_histo/", filename="graph_patch_raw_resnet18_20x_val.csv", val=True)


train_dataset = PatchDataset(root="/nobackup/projects/bdlds05/lucyg/graph_project/10x_res18_histo/", filename = 'graph_raw_10x_res18_train.csv')
test_dataset = PatchDataset(root="/nobackup/projects/bdlds05/lucyg/graph_project/10x_res18_histo/", filename="graph_raw_10x_res18_test.csv", test=True)
val_dataset = PatchDataset(root="/nobackup/projects/bdlds05/lucyg/graph_project/10x_res18_histo/", filename="graph_raw_10x_res18_val.csv", val=True)


n_classes = 2
results_dir = '/nobackup/projects/bdlds05/lucyg/graph_project/10x_res18_histo/results/'

train_loader = DataLoader(train_dataset,batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=1, shuffle=True)

#for step, data in enumerate(train_loader):
#    print(f'Step {step + 1}:')
#    print('=======')
#    print(f'Number of graphs in the current batch: {data.num_graphs}')
#    print(data)
#    print()
    
    
model = CLAM_SB()#, out_channels=512)
#model = GCNConv(in_channels=512, out_channels=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()



class Accuracy_Logger(object):
    """Accuracy logger"""
    def __init__(self, n_classes):
        super(Accuracy_Logger, self).__init__()
        self.n_classes = n_classes
        self.initialize()

    def initialize(self):
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    
    def log(self, Y_hat, Y):
        Y_hat = int(Y_hat)
        Y = int(Y)
        self.data[Y]["count"] += 1
        self.data[Y]["correct"] += (Y_hat == Y)
    
    def log_batch(self, Y_hat, Y):
        Y_hat = np.array(Y_hat).astype(int)
        Y = np.array(Y).astype(int)
        for label_class in np.unique(Y):
            cls_mask = Y == label_class
            self.data[label_class]["count"] += cls_mask.sum()
            self.data[label_class]["correct"] += (Y_hat[cls_mask] == Y[cls_mask]).sum()
    
    def get_summary(self, c):
        count = self.data[c]["count"] 
        correct = self.data[c]["correct"]
        
        if count == 0: 
            acc = None
        else:
            acc = float(correct) / count
        
        return acc, correct, count
        
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, stop_epoch=50, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss

def calculate_error(Y_hat, Y):
    error = 1. - Y_hat.float().eq(Y.float()).float().mean().item()

    return error




def train():
    model.train()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    train_error = 0.
    train_loss = 0.
    running_loss = 0.0
    correct = 0
    for batch_idx, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
         #print(data.y)
         logits, Y_prob, Y_hat, _, _ = model(data.x, data.y, data.edge_index, data.batch)  # Perform a single forward pass.
         acc_logger.log(Y_hat, data.y)
         loss = criterion(logits, data.y)
         train_loss += loss.item()
         #running_loss += loss.item()
         
         if (batch_idx + 1) % 20 == 0:
            print('batch {}, loss: {:.4f}, label: {}, bag_size: {}'.format(batch_idx, loss.item(), data.y.item(), data.size(0)))
            
            
         error = calculate_error(Y_hat, data.y)
         train_error += error
         #pred = out.argmax(dim=print('Y_prob', Y_prob)1)  # Use the class with highest probability.
         #calculate_metrics(Y_hat, data.y, epoch, "train")
         #log_conf_matrix(Y_hat, data.y, epoch, "train")
         #correct += int((Y_hat == data.y).sum())
         # Check against ground-truth labels.
         #loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         #* 128 #batch_size         
         
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.
    #train_loss=train_running_loss/len(data_loader['train'])
    #epoch_loss = running_loss / len(train_loader)
    #print('Train Loss: %.3f'%(epoch_loss))
        # calculate loss and error for epoch
    train_loss /= len(train_loader)
    train_error /= len(train_loader)

    print('Epoch: {}, train_loss: {:.4f}, train_error: {:.4f}'.format(epoch, train_loss, train_error))
    for i in range(n_classes):
        acc, correct, count = acc_logger.get_summary(i)
        print('class {}: acc {}, correct {}/{}'.format(i, acc, correct, count))
        #if writer:
            #writer.add_scalar('train/class_{}_acc'.format(i), acc, epoch)

    #if writer:
        #writer.add_scalar('train/loss', train_loss, epoch)
        #writer.add_scalar('train/error', train_error, epoch)
        
        
def validate(epoch, loader, n_classes, early_stopping = None, results_dir=None):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    # loader.dataset.update_mode(True)
    val_loss = 0.
    val_error = 0.
    
    prob = np.zeros((len(loader), n_classes))
    labels = np.zeros(len(loader))

    with torch.no_grad():
        for batch_idx, data in enumerate(loader):
            #data, data.y = data.to(device, non_blocking=True), data.y.to(device, non_blocking=True)

            logits, Y_prob, Y_hat, _, _ = model(data.x, data.y, data.edge_index, data.batch)

            acc_logger.log(Y_hat, data.y)
            
            loss = criterion(logits, data.y)

            prob[batch_idx] = Y_prob#.cpu().numpy()
            labels[batch_idx] =data.y.item()
            
            val_loss += loss.item()
            error = calculate_error(Y_hat, data.y)
            val_error += error
            

    val_error /= len(loader)
    val_loss /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(labels, prob[:, 1])
    
    else:
        auc = roc_auc_score(labels, prob, multi_class='ovr')
        
    if early_stopping:
        assert results_dir
        early_stopping(epoch, val_loss, model, ckpt_name = os.path.join(results_dir, "s_{}_checkpoint.pt".format(epoch)))
        
    if early_stopping.early_stop:
        print("Early stopping")
        return True

    return False
    
    
def summary(loader, n_classes):
    #device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acc_logger = Accuracy_Logger(n_classes=n_classes)
    model.eval()
    test_loss = 0.
    test_error = 0.

    all_probs = np.zeros((len(loader), n_classes))
    all_labels = np.zeros(len(loader))

    #slide_ids = loader.dataset.slide_data['slide_id']
    #patient_results = {}

    for batch_idx, data in enumerate(train_loader):
        #data, label = data.to(device), data.y.to(device)
        #slide_id = slide_ids.iloc[batch_idx]
        if batch_idx==len(all_probs):
            #print('all_probs', all_probs)
            break
        with torch.no_grad():
            logits, Y_prob, Y_hat, _, _ = model(data.x, data.y, data.edge_index, data.batch)

        acc_logger.log(Y_hat, data.y)
        probs = Y_prob#.cpu().numpy()
        #print('batch_idx',batch_idx)
        #print('batch_idx', len(all_probs))
        all_probs[batch_idx] = probs
        all_labels[batch_idx] = data.y.item()
        
        #patient_results.update({slide_id: {'slide_id': np.array(slide_id), 'prob': probs, 'label': data.y.item()}})
        error = calculate_error(Y_hat, data.y)
        test_error += error

    test_error /= len(loader)

    if n_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
        aucs = []
    else:
        aucs = []
        binary_labels = label_binarize(all_labels, classes=[i for i in range(n_classes)])
        for class_idx in range(n_classes):
            if class_idx in all_labels:
                fpr, tpr, _ = roc_curve(binary_labels[:, class_idx], all_probs[:, class_idx])
                aucs.append(calc_auc(fpr, tpr))
            else:
                aucs.append(float('nan'))

        auc = np.nanmean(np.array(aucs))


    return test_error, auc, acc_logger

def test(loader, epoch):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         #out = model(data.x, data.edge_index, data.batch)  
         logits, Y_prob, Y_hat, _, _ = model(data.x, data.y, data.edge_index, data.batch)  # Perform a single forward pass.
         #pred = out.argmax(dim=1)  # Use the class with highest probability.
         #calculate_metrics(Y_hat, data.y, epoch, "test")
         #log_conf_matrix(Y_hat, data.y, epoch, "test")
         #correct += int((Y_hat == data.y).sum())  # Check against ground-truth labels.
     #return correct / len(loader.dataset)  # Derive ratio of correct predictions.
     
def log_conf_matrix(y_pred, y_true, epoch, type):
    # Log confusion matrix as image
    cm = confusion_matrix(y_pred, y_true)
    classes = ["0", "1"]#, "2"]
    df_cfm = pd.DataFrame(cm, index = classes, columns = classes)
    plt.figure(figsize = (10,7))
    cfm_plot = sns.heatmap(df_cfm, annot=True, cmap='Blues', fmt='g')
    cfm_plot.figure.savefig(f'data/images/{type}_cm_{epoch}.png')
    
    
def calculate_metrics(y_pred, y_true, epoch, type):
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"F1 Score: {f1_score(y_true, y_pred, average='weighted')}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    prec = precision_score(y_true, y_pred, average='weighted')
    rec = recall_score(y_true, y_pred, average='weighted')
    print(f"Precision: {prec}")
    print(f"Recall: {rec}")     


early_stopping=EarlyStopping(patience = 20, stop_epoch=50, verbose=True)

     
for epoch in range(1, 3000):
    #train()
    #results, test_auc, val_auc, test_acc, val_acc  = train(datasets, i, args)
    #train_acc = test(train_loader, epoch)
    #test_acc = test(test_loader, epoch)
    #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    #folds = np.arange(start, end)
    #for i in folds:
    #seed_torch(args.seed)
    #train_dataset, val_dataset, test_dataset = dataset.return_splits(from_id=False, 
     #           csv_path='{}/splits_{}.csv'.format(args.split_dir, i))
        
    #    datasets = (train_dataset, val_dataset, test_dataset)
    train()
    stop = validate(epoch,val_loader, n_classes=2, early_stopping=EarlyStopping(patience = 20, stop_epoch=50, verbose = True), results_dir=results_dir)
    if stop: 
            break


if early_stopping:
    model.load_state_dict(torch.load(os.path.join(results_dir, "s_{}_checkpoint.pt".format(epoch))))
else:
    torch.save(model.state_dict(), os.path.join(results_dir, "s_{}_checkpoint.pt".format(epoch)))

val_error, val_auc, _= summary(val_loader, n_classes)
print('Val error: {:.4f}, ROC AUC: {:.4f}'.format(val_error, val_auc))

test_error, test_auc, acc_logger = summary(test_loader, n_classes)
test_acc = 1 - test_error
val_acc = 1 - val_error
print('Test error: {:.4f}, ROC AUC: {:.4f}'.format(test_error, test_auc))


print('test_auc', test_auc)
print('val_auc', val_auc)
print('test_acc', test_acc)
print('val_acc', val_acc)



final_df = pd.DataFrame({'folds': 0, 'test_auc': test_auc, 
        'val_auc': val_auc, 'test_acc': test_acc, 'val_acc' : val_acc}, index=[0])

save_name = 'summary.csv'
final_df.to_csv(os.path.join(results_dir, save_name))


    
    

    
    
    
    
    
    
    
    
    
    
    
    
    