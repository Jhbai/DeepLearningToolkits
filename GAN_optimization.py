import torch
import torch.nn as nn

criterion = nn.BCELoss()
optimizerG = torch.optim.Adam(netG.parameters(), lr = 0.0001, betas = (0.5, 0.999))
optimizerD = torch.optim.Adam(netD.parameters(), lr = 0.0001, betas = (0.5, 0.999))

import warnings
warnings.filterwarnings("ignore")
X = X.to(device = 'cuda')
LOSSES = list()
photo_show = torch.randn(5, nz, 1, 1).detach().to(device = 'cuda')
batch_size = 32
for epoch in range(950):
    for real_data in dataloader(batch_size, X):
        ##### 辨別器更新 #####
        optimizerD.zero_grad()
        label = torch.tensor(np.ones(shape = (real_data.shape[0], ))).to(torch.float32).to(device = 'cuda')
        output = netD(real_data).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()
        noise = torch.randn(real_data.shape[0], nz, 1, 1).to(device = 'cuda')
        fake = netG(noise)
        FAKE = torch.tensor(np.zeros(shape = (real_data.shape[0], ))).to(torch.float32).to(device = 'cuda')
        errD_fake = criterion(netD(fake.detach()).view(-1), FAKE)
        errD_fake.backward()
        optimizerD.step()

        ##### 生成器更新 #####
        optimizerG.zero_grad()
        noise.data.copy_(torch.randn(real_data.shape[0], nz, 1, 1)).to(device = 'cuda')
        fake = netG(noise)
        errG = criterion(netD(fake).view(-1), 1 - FAKE)
        errG.backward()
        optimizerG.step()
        
    print('Epoch:', epoch + 1)
    multi_Draw(netG(photo_show))
    cv2.imshow('graph',np.array((netG(photo_show)[0]*0.5 + 0.5).permute(2,1,0).detach().to(device = 'cpu')))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()