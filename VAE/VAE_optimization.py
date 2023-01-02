BATCH = 16
for epoch in range(150):
    loss_val = 0
    for batch in range(0, X.shape[0], BATCH):
        if batch + BATCH > X.shape[0]:
            x = X[batch:]
        else:
            x = X[batch:batch + BATCH]
        OPT.zero_grad()
        output = model(x)
        BCE = LOSS(output, x)
        sigma1 = model.encoder.sigma1
        mu1 = model.encoder.mu1
        KL1 = - 0.5 * torch.sum(1 + sigma1 - mu1.pow(2) - torch.exp(sigma1))
        sigma2 = model.encoder.sigma2
        mu2 = model.encoder.mu2
        KL2 = - 0.5 * torch.sum(1 + sigma2 - mu2.pow(2) - torch.exp(sigma2))
        sigma3 = model.encoder.sigma3
        mu3 = model.encoder.mu3
        KL3 = - 0.5 * torch.sum(1 + sigma3 - mu3.pow(2) - torch.exp(sigma3))
        loss = BCE + KL1 + KL2 + KL3
        loss.backward()
        OPT.step()
        
    output = model(X)
    loss = LOSS(output, X)
    loss_val = float(loss)
    Losses.append(loss_val)
    print("Epoch: {0}; Loss: {1}".format(epoch + 1, loss_val))