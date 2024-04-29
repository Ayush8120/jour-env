import torch
from torch import nn
import model,data,utils
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from pathlib import Path


def train(seed:int):
    X_train,y_train,X_test,y_test = data.createDataset().giveSplits(0.8)

    # print(list(model_0.parameters()))
    # print(model_0.state_dict())
    model_0 = model.LinearRegressionModel()
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.1)
    scheduler = StepLR(optimizer, step_size= 10,gamma= 0.1)
    epochs = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    #bag inits
    train_loss_values = []
    test_loss_values = []
    epoch_count = []

    for epoch in range(epochs):
        model_0.train()
        y_pred = model_0(X_train)
        loss = loss_fn(y_pred, y_train)
        # print(loss)
        # print()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        model_0.eval()
        with torch.inference_mode():
            '''
            turns off gradient tracking
            '''
            test_preds = model_0(X_test)

            #visualization
            # Check the predictions
            # print(f"Number of testing samples: {len(X_test)}") 
            # print(f"Number of predictions made: {len(y_preds)}")
            # print(f"Predicted values:\n{y_preds}")
            test_loss = loss_fn(test_preds, y_test.type(torch.float))
            if epoch % 10 == 0:
                epoch_count.append(epoch)
                train_loss_values.append(loss.detach().numpy())
                test_loss_values.append(test_loss.detach().numpy())
                print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

                # utils.plot_predictions(train_data=X_train, train_labels=y_train,test_data=X_test,test_labels=y_test
            # , predictions=test_preds)
                
    # Plot the loss curves
    # plt.plot(epoch_count, train_loss_values, label="Train loss")
    # plt.plot(epoch_count, test_loss_values, label="Test loss")
    # plt.title("Training and test loss curves")
    # plt.ylabel("Loss")
    # plt.xlabel("Epochs")
    # plt.legend()
    
    '''
    Saving Model and state_dict
    '''
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True, exist_ok=True)

    MODEL_NAME = "seed_42_test_loss_0-05049.pth"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model_0.state_dict(), # only saving the state_dict() only saves the models learned parameters
           f=MODEL_SAVE_PATH)

    del model_0
    loaded_model_0 = model.LinearRegressionModel()
    
    # Load the state_dict of our saved model (this will update the new instance of our model with trained weights)
    loaded_model_0.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
    loaded_model_0.eval()
    with torch.inference_mode():
        '''
        turns off gradient tracking
        '''
        test_preds = loaded_model_0(X_test)
        test_loss = loss_fn(test_preds, y_test.type(torch.float))
        print(test_loss)
            
if __name__ == "__main__":
    # for i in range(10):
    train(torch.manual_seed(42))
    # plt.show()
