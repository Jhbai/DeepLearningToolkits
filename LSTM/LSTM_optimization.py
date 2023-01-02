EPOCHS = 10000
around = 100
time_elapse = EPOCHS/around
for epoch in range(EPOCHS):
    opt.zero_grad()
    pred = model(x.reshape(x.shape[0], x.shape[1], 1))
    LOSS = criterion(pred, y).requires_grad_(True)
    LOSS.backward()
    opt.step()
    print('epoch {}, loss: {}'.format(epoch + 1, round(float(LOSS), 5)))