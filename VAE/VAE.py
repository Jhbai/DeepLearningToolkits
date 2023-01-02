class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        self.Model1 = nn.Sequential(nn.Linear(graph_dim, d1),
                                    nn.ReLU(),
                                    nn.Linear(d1, d2),
                                    nn.ReLU(),
                                    nn.Linear(d2, d3),
                                    nn.ReLU())
        self.Model2 = nn.Sequential(nn.Linear(graph_dim, d1),
                                    nn.ReLU(),
                                    nn.Linear(d1, d2),
                                    nn.ReLU(),
                                    nn.Linear(d2, d3),
                                    nn.ReLU())
        self.Model3 = nn.Sequential(nn.Linear(graph_dim, d1),
                                    nn.ReLU(),
                                    nn.Linear(d1, d2),
                                    nn.ReLU(),
                                    nn.Linear(d2, d3),
                                    nn.ReLU())
        self.MU1 = nn.Linear(d3, d_model)
        self.SI1 = nn.Linear(d3, d_model)
        self.MU2 = nn.Linear(d3, d_model)
        self.SI2 = nn.Linear(d3, d_model)
        self.MU3 = nn.Linear(d3, d_model)
        self.SI3 = nn.Linear(d3, d_model)
        self.Distribution1 = torch.distributions.Normal(0, 1)
        self.Distribution2 = torch.distributions.Normal(0, 1)
        self.Distribution3 = torch.distributions.Normal(0, 1)

    def forward(self, x):
        X1, X2, X3 = torch.flatten(x[:,:,0], start_dim=1), torch.flatten(x[:,:,1], start_dim=1), torch.flatten(x[:,:,2], start_dim=1)
        X1, X2, X3 = self.Model1(X1), self.Model1(X2), self.Model1(X3)
        self.mu1, self.sigma1 = self.MU1(X1), self.SI1(X1)
        self.mu2, self.sigma2 = self.MU1(X2), self.SI1(X2)
        self.mu3, self.sigma3 = self.MU1(X3), self.SI1(X3)

        z1 = self.mu1 + self.sigma1*self.Distribution1.sample(self.mu1.shape)
        z2 = self.mu2 + self.sigma2*self.Distribution2.sample(self.mu2.shape)
        z3 = self.mu3 + self.sigma3*self.Distribution3.sample(self.mu3.shape)
        return z1, z2, z3


class VariationalDecoder(nn.Module):
    def __init__(self):
        super(VariationalDecoder, self).__init__()
        self.Model1 = nn.Sequential(nn.Linear(d_model, d3),
                                    nn.ReLU(),
                                    nn.Linear(d3, d2),
                                    nn.ReLU(),
                                    nn.Linear(d2, d1),
                                    nn.ReLU(),
                                    nn.Linear(d1, graph_dim),
                                    nn.Sigmoid())
        self.Model2 = nn.Sequential(nn.Linear(d_model, d3),
                                    nn.ReLU(),
                                    nn.Linear(d3, d2),
                                    nn.ReLU(),
                                    nn.Linear(d2, d1),
                                    nn.ReLU(),
                                    nn.Linear(d1, graph_dim),
                                    nn.Sigmoid())
        self.Model3 = nn.Sequential(nn.Linear(d_model, d3),
                                    nn.ReLU(),
                                    nn.Linear(d3, d2),
                                    nn.ReLU(),
                                    nn.Linear(d2, d1),
                                    nn.ReLU(),
                                    nn.Linear(d1, graph_dim),
                                    nn.Sigmoid())    
    def forward(self, x): # (N, 2)
        n = x[0].shape[0]
        X1, X2, X3 = self.Model1(x[0]).reshape(n, graph_dim, 1),self.Model2(x[1]).reshape(n, graph_dim, 1),self.Model3(x[2]).reshape(n, graph_dim, 1),
        X = torch.cat((X1, X2, X3), 2)
        return X


class VariationalAutoencoder(nn.Module):
    def __init__(self):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder()
        self.decoder = VariationalDecoder()

    def forward(self, x):
        z = self.encoder(x)
        Result = self.decoder(z)
        return Result