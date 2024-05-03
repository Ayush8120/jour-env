import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from pathlib import Path
import pdb,numpy
import data,constants,model
torch.manual_seed(42)
def calc_accuracy(y_true, y_pred):
    correct = torch.eq(y_true,y_pred).sum().item()
    assert len(y_pred) == len(y_true), "sizes of pred and labels dont match"
    acc = (correct/len(y_pred)) * 100
    return acc

def train(X_train,Y_train,X_valid,Y_valid,X_test,Y_test):
    epochs = constants.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)
    X_valid = X_valid.to(device)
    Y_valid = Y_valid.to(device)
    
    model_0 = model.ClassificationNet().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model_0.parameters(),lr = 0.1)
    # scheduler = StepLR(optimizer, step_size=1000, gamma=0.7)

    epoch_count = []
    test_loss_values = []
    train_loss_values = []
    valid_loss_values = []
    train_acc_values = []
    val_acc_values = []
    '''
    initial random loss
    '''
    model_0.eval()
    with torch.inference_mode():
        valid_preds = model_0(X_valid)
        valid_loss = loss_fn(valid_preds, Y_valid.type(torch.float))
        print(f'valid loss before training : {valid_loss}')

    for epoch in range(epochs):
        model_0.train()
        Y_pred_logits = model_0(X_train)
        # print(Y_pred[:5])
        # pdb.set_trace()
        loss = loss_fn(Y_pred_logits,Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # scheduler.step()
            
        model_0.eval()
        with torch.inference_mode():
            valid_pred_logits = model_0(X_valid)
            valid_loss = loss_fn(valid_pred_logits, Y_valid.type(torch.float))
            if epoch % 100 == 0:
                epoch_count.append(epoch)

                Y_pred = torch.round(torch.sigmoid(Y_pred_logits))
                train_acc = calc_accuracy(Y_train,Y_pred)
                train_acc_values.append(train_acc)
                valid_pred = torch.round(torch.sigmoid(valid_pred_logits))
                valid_acc = calc_accuracy(Y_valid, valid_pred)
                val_acc_values.append(valid_acc)
                
                train_loss_values.append(loss.cpu().numpy())
                valid_loss_values.append(valid_loss.cpu().numpy())
                print(f"Epoch: {epoch} | BCE Train Loss: {loss} Train Acc : {train_acc}| BCE Valid Loss: {valid_loss} | valid acc : {valid_acc}")

    '''
    Saving Model and state_dict
    '''
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "model.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)

    '''
    deleting and loading the model 
    '''
    del model_0
    loaded_model_0 = model.ClassificationNet().to(device)

    loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
    loaded_model_0.eval()
    with torch.inference_mode():
        '''
        turns off gradient tracking
        '''
        test_pred_logits = loaded_model_0(X_test)
        # pdb.set_trace()
        test_loss = loss_fn(test_pred_logits, Y_test.type(torch.float))

        Y_pred = torch.round(torch.sigmoid(test_pred_logits))
        train_acc = calc_accuracy(Y_test,Y_pred)
        print(train_acc)
        print()
        print(test_loss)
    
if __name__ == "__main__":
    X_train,Y_train, X_valid,Y_valid, X_test,Y_test = data.createDataset().giveDataset()
    # pdb.set_trace()
    train(X_train,Y_train,X_valid,Y_valid,X_test,Y_test)
    