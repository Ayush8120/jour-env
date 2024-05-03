import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from pathlib import Path

import data,constants,model


def train():
    epochs = constants.epochs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_0 = model.ClassificationNet()
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params=model_0.parameters(),lr = 0.1)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    epoch_count = []
    test_loss_values = []
    train_loss_values = []
    valid_loss_values = []

    for epoch in epochs:
        model_0.train()
        Y_pred = model_0(X_train)
        loss = loss_fn(Y_pred,Y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
            
        model_0.eval()
        with torch.inference_mode():
            valid_preds = model_0(X_valid)
            valid_loss = loss_fn(valid_preds, Y_valid.type(torch.float))
            if epoch % 10 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss.detach().numpy())
                valid_loss_values.append(valid_loss.detach().numpy())
                print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Valid Loss: {valid_loss} ")

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
    loaded_model_0 = model.ClassificationNet()

    loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
    loaded_model_0.eval()
    with torch.inference_mode():
        '''
        turns off gradient tracking
        '''
        test_preds = loaded_model_0(X_test)
        test_loss = loss_fn(test_preds, Y_test.type(torch.float))
        print(test_loss)
    
if __name__ == "__main__":
    X_train,Y_train, X_valid,Y_valid, X_test,Y_test = data.createDataset().giveDataset()
    train(X_train,Y_train,X_valid,Y_valid,X_test,Y_test)
    