import torch
import torch.nn as nn
import numpy as np

class SurvivalLoss(nn.Module):
    """Survival loss.
    Samples need to be sorted with increasing survial time.
    """
    def __init__(self):
        super(SurvivalLoss, self).__init__()

    def forward(self, y_pred, y_true):
        """
        y_pred is the predicted risk,
        y_true is event indicator, when event=1 means uncensored and event=0 means censored.
        """
        risk = y_pred
        event = y_true
        risk_exp = torch.exp(risk)
        risk_exp_cumsum = torch.cumsum(risk_exp, dim=0)
        risk_exp_cumsum = torch.flip(risk_exp_cumsum, dims=[0])

        SurvivalLoss = - torch.sum(event * (risk - torch.log(risk_exp_cumsum))) / torch.sum(event)

        return SurvivalLoss




if __name__ == '__main__':



    y_pred = np.array([0.9,0.9,0.9,0.4,0.5,0.6,0.7,0.8,0.9,0.8])

    y_true = np.array([0,0,0,0,1,1,1,1,1,1])

    risk = y_pred
    # event = np.cast(y_true, dtype=risk.dtype)
    event = y_true

    risk_exp = np.exp(risk)
    risk_exp_cumsum = np.cumsum(risk_exp)
    risk_exp_cumsum = np.flip(risk_exp_cumsum)
    likelihood = risk - np.log(risk_exp_cumsum)
    uncensored_likelihood = np.multiply(likelihood, event)

    n_observed = np.sum(event)
    cox_loss = -np.sum(uncensored_likelihood) / n_observed
    print(cox_loss)


    y_pred = torch.as_tensor(y_pred)
    # y_pred = torch.as_tensor([0.2,0.,0.,0.,0.,0.,0.50,0.49,0.3,0.1])
    y_true = torch.as_tensor(y_true)

    risk = y_pred
    event = y_true
    risk_exp = torch.exp(risk)
    risk_exp_cumsum = torch.cumsum(risk_exp, dim=0)
    risk_exp_cumsum = torch.flip(risk_exp_cumsum, dims=[0])

    SurvivalLoss = - torch.sum(event * (risk - torch.log(risk_exp_cumsum))) / torch.sum(event)
    print(SurvivalLoss)
