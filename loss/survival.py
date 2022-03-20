import torch
import torch.nn as nn
import numpy as np

class SurvivalLoss(nn.Module):
    """Survival loss.
    Samples need to be sorted with increasing survial time.
    """
    def __init__(self):
        super(SurvivalLoss, self).__init__()

    def forward(self, y_pred, y_true, device='gpu'):
        """
        y_pred is the predicted risk,
        y_true is event indicator, when event=1 means uncensored and event=0 means censored.
        """
        if device == 'gpu':
            y_pred = y_pred.cuda()
            y_true = y_true.cuda()
        risk = y_pred
        event = y_true
        risk_exp = torch.exp(risk)
        risk_exp = torch.flip(risk_exp, dims=[0])
        risk_exp_cumsum = torch.cumsum(risk_exp, dim=0)
        risk_exp_cumsum = torch.flip(risk_exp_cumsum, dims=[0])

        likelihood = risk - torch.log(risk_exp_cumsum)
        uncensored_likelihood = event * likelihood
        SurvivalLoss = - torch.sum(uncensored_likelihood) / torch.sum(event)

        return SurvivalLoss





if __name__ == '__main__':
    y_pred = np.array([0.9, 0.6, 0.5, 0.4, 0.2, 0.1])

    y_true = np.array([1, 1, 0, 0, 0, 0])

    risk = y_pred
    event = y_true

    risk_exp = np.exp(risk)
    risk_exp = np.flip(risk_exp)
    risk_exp_cumsum = np.cumsum(risk_exp)
    risk_exp_cumsum = np.flip(risk_exp_cumsum)

    likelihood = risk - np.log(risk_exp_cumsum)
    uncensored_likelihood = np.multiply(likelihood, event)

    n_observed = np.sum(event)
    cox_loss = -np.sum(uncensored_likelihood) / n_observed
    print(cox_loss)



    a = torch.as_tensor(0.9)-torch.log(torch.exp(torch.as_tensor(0.6))+torch.exp(torch.as_tensor(0.9)))
    b = torch.as_tensor(0.6)-torch.log(torch.exp(torch.as_tensor(0.6)))
    print(-1/2*(a + b))


    y_pred = torch.as_tensor(y_pred)
    y_true = torch.as_tensor(y_true)

    risk = y_pred
    event = y_true
    risk_exp = torch.exp(risk)
    risk_exp = torch.flip(risk_exp, dims=[0])
    risk_exp_cumsum = torch.cumsum(risk_exp, dim=0)
    risk_exp_cumsum = torch.flip(risk_exp_cumsum, dims=[0])

    likelihood = risk - torch.log(risk_exp_cumsum)
    uncensored_likelihood = event * likelihood
    sl = - torch.sum(uncensored_likelihood) / torch.sum(event)
    print(sl)


    class test_mlp(nn.Module):
        def __init__(self, n):
            super(test_mlp, self).__init__()
            self.fc1 = nn.Linear(n, n * 2)
            self.activate1 = nn.Sigmoid()
            self.fc2 = nn.Linear(n * 2, n * 2)
            self.activate2 = nn.Sigmoid()
            self.fc3 = nn.Linear(n * 2, n)
            self.activate3 = nn.Sigmoid()

        def forward(self, x):
            x = self.fc1(x)
            x = self.activate1(x)
            x = self.fc2(x)
            x = self.activate2(x)
            x = self.fc3(x)
            return x

    net = test_mlp(n=len(y_true))
    loss = SurvivalLoss()
    temp_optimizer = torch.optim.SGD(net.parameters(), lr=2)
    for i in range(99):
        temp_optimizer.zero_grad()

        x = torch.as_tensor(y_true).float()
        y = torch.as_tensor(y_true).float()
        y_hat = net(x)
        l = loss(y_hat,y, device='cpu')
        l.backward()
        show_y = y_hat.detach().numpy()
        # show_y = [a for a in y_hat.item()]
        temp_optimizer.step()
        print("epoch:", str(i), " y_hat:", show_y, " loss:", str(l.item()))
