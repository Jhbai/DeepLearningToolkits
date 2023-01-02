EPOCHS = 1000
model = Classification2()
Loss = nn.CrossEntropyLoss() # ignore_index = 0
OPT = torch.optim.Adam(model.parameters(), lr = 0.01)
losses = list()
ts_losses = list()
acces = list()
acces2 = list()
MIN = float('inf')
X_tr, y_tr, X_ts, y_ts = tr_ts_split(enc_input, y, 0.7)

for epoch in range(EPOCHS):
    output = model(X_tr.t())
    y_hat = output.reshape(output.shape[0], 4)
    loss = Loss(y_hat, y_tr)
    losses.append(float(loss))
    acc = (torch.argmax(output, axis = 1) == y_tr).sum()/X_tr.shape[0]
    acces.append(acc)
    output2 = model(X_ts.t())
    y_hat2 = output2.reshape(output2.shape[0], 4)
    loss2 = Loss(y_hat2, y_ts)
    ts_losses.append(float(loss2))
    acc2 = (torch.argmax(output2, axis = 1) == y_ts).sum()/X_ts.shape[0]
    acces2.append(acc2)
    if float(loss2) < MIN:
        MIN = float(loss2)
        flag = 0
        torch.save(model, 'TEMP.pt')
    else:
        flag += 1
    OPT.zero_grad()
    loss.backward()
    print("Epoch {0} --train Loss:{1}; Acc:{2} --test Loss:{3}; Acc:{4}".format(epoch + 1, float(loss), float(acc),
                                                                             float(loss2), float(acc2)))
    OPT.step()
