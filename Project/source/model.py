class Model:
    def __init__(self,):
        R = self.R  # word representation matrix 
        V = self.V  # Vocabulary

    def energy(self, theta, phi_w, b_w):
        return -theta.T @ phi_w 
    
    def obtain_word_distribution(self, theta, R, b):
        energy = - theta.T @ (R @ w)
