import torch
import numpy as np



class Optimizer():
    """Optimizer for the surrogate model to find optimal design parameters.
    Args:
        theta_nominal (list): Nominal design parameters.
        surrogate (nn.Module): Surrogate model to optimize.
        std_dev (float): Standard deviation for the covariance matrix.
        constraints (dict): Constraints for the optimization.
        step_sz (float): Step size for the optimization.
        lr (float): Learning rate for the optimizer.
        epoch_n (int): Number of epochs for the optimization.
        device (str): Device to run the model on ('cpu' or 'cuda').
    """

    def __init__(self, theta_nominal, surrogate, std_dev=1.0, constraints=None, step_sz=8e-1, lr=0.01, epoch_n=30, device='cpu'):
        self.theta = torch.tensor(theta_nominal,dtype=torch.float32).to(device)
        self.theta.requires_grad_()
        self.theta_nominal = theta_nominal
        cov_matrix = np.eye(len(theta_nominal)) * std_dev**2  # covariance matrix
        self.cov_matrix = torch.tensor(cov_matrix, dtype=torch.float32).to(device)
        self.surrogate = surrogate
        self.step_sz = step_sz
        self.optimizer = torch.optim.Adam([self.theta], lr=lr)
        self.epoch_n = epoch_n
        self.constraints = constraints or {'sum_params': 10}

    def is_local(self,theta_next, threshold=1.5):
        cov = self.cov_matrix.clone().detach().cpu().numpy()
        diff = theta_next - self.theta_nominal
        return np.dot(diff, np.dot(np.linalg.inv(cov), diff)) < threshold

    
    def add_constraint_penalty(self):
        """Add penalty to the loss function to enforce constraints.
            all constraints have to be defined in the constructor as a dictionary
            a check for the dictionary keys has to be added in this function
            currently 'sum_params' is implemented
        """
       
        curr_theta = self.theta.clone().detach()
        total_magnitude_loss = 0
        if self.constraints is not None:
            if 'sum_params' in self.constraints:
                total_magnitude = torch.sum(torch.nn.ReLU()(curr_theta))#<0 lengths don't count
                total_magnitude_loss = torch.mean(torch.nn.ReLU()(total_magnitude - self.constraints['sum_params'])**2)

        lower_para_bound = -self.cov_matrix/1.1
        bloss = torch.mean(torch.nn.ReLU()(lower_para_bound - curr_theta)**2)

        return total_magnitude_loss + bloss


    def optimize(self, with_constraints=True):
  
        self.surrogate.eval()
        thetas = []
  
        for _ in range(self.epoch_n):

            thetas.append(self.theta.clone().detach().cpu().numpy())

            self.optimizer.zero_grad()
            mi_hat = self.surrogate(self.theta.unsqueeze(0))
            loss = -mi_hat
            if with_constraints:
                loss += self.add_constraint_penalty()
            
            loss.backward()
            self.optimizer.step()

            theta_next = self.theta.clone().detach().cpu().numpy()
            if ((theta_next - thetas[-1])**2).sum() < 1e-5: # convergence
                break
            if not self.is_local(theta_next): # outside of local region
                break  
           
        return np.array(thetas)

