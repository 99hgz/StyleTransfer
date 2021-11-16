from model.net_build import *
from predata import Data_set
import torch
import itertools
from utils import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import itertools
from tensorboardX import SummaryWriter
import sys
if __name__ == '__main__':
    #id = int(sys.argv[1])
    id = 10
    writer = SummaryWriter('log')
    #dir_img = '../wikiart/{}/'.format(1)
    dir_img = '/home/cloudam/monet2photo/monet2photo/trainA/'
    dir_real = '/home/cloudam/monet2photo/monet2photo/trainB/'
    dir_checkpoint = './checkpoints/'
    path_checkpointgab = './checkpoints/checkpointGab_0_epoch30.pth'
    path_checkpointgba = './checkpoints/checkpointGba_0_epoch30.pth'
    path_checkpointda = './checkpoints/checkpointDa_0_epoch30.pth'
    path_checkpointdb = './checkpoints/checkpointDb_0_epoch30.pth'
    batch_size = 12
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    Gab=generator(n_channels=3, n_classes=3).to(device=device)
    Gba=generator(n_channels=3, n_classes=3).to(device=device)
    Gab.load_state_dict(torch.load(path_checkpointgab, map_location=device))
    Gba.load_state_dict(torch.load(path_checkpointgba, map_location=device))

    Da=discriminator(3).to(device=device)
    Db=discriminator(3).to(device=device)
    Da.load_state_dict(torch.load(path_checkpointda, map_location=device))
    Db.load_state_dict(torch.load(path_checkpointdb, map_location=device))

    #Gab.apply(weights_init)
    #Gba.apply(weights_init)
    #Da.apply(weights_init)
    #Db.apply(weights_init)

    G_optimizer = torch.optim.Adam(itertools.chain(Gab.parameters(), Gba.parameters()),lr=0.0002, betas=(0.5, 0.99))
    Da_optimizer = torch.optim.Adam(Da.parameters(), lr=0.0002, betas=(0.5, 0.99))
    Db_optimizer = torch.optim.Adam(Db.parameters(), lr=0.0002, betas=(0.5, 0.99))
    train_data=Data_set(dir_img,dir_real)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    GAN_loss = LSGanLoss()
    cycle_loss = torch.nn.L1Loss()
    identity_loss = torch.nn.L1Loss()
    #target_real = Variable(torch.cuda.FloatTensor(batch_size,1).fill_(1.0), requires_grad=False)
    #target_fake = Variable(torch.cuda.FloatTensor(batch_size,1).fill_(0.0), requires_grad=False)
    Gab.train()
    Gba.train()
    Da.train()
    Db.train()
    it = 0
    batches=len(train_loader)
    for epoch in range(epochs):
        num = 0
        for batch in train_loader:
            num += 1
            A=batch['fake']
            B=batch['real']
            A=A.to(device=device, dtype=torch.float32)
            B=B.to(device=device, dtype=torch.float32)
            G_optimizer.zero_grad()

            B1 = Gab(B)
            loss_identity_B = identity_loss(B1, B)
            A1 = Gba(A)
            loss_identity_A = identity_loss(A, A1)

            fake_B = Gab(A)
            disc_fake = Db(fake_B)
            loss_GAN_A2B = GAN_loss._g_loss(disc_fake)

            fake_A = Gba(B)
            disc_fake2 = Da(fake_A)
            loss_GAN_B2A = GAN_loss._g_loss(disc_fake2)

            recovered_A = Gba(fake_B)
            loss_cycle_ABA = cycle_loss(recovered_A, A)

            recovered_B = Gab(fake_A)
            loss_cycle_BAB = cycle_loss(recovered_B, B)

            #parameters here
            w1=5.0
            w2=10.0
            loss_sum=loss_identity_B*w1+loss_identity_A*w1+loss_GAN_A2B+loss_GAN_B2A+loss_cycle_ABA*w2+loss_cycle_BAB*w2
            loss_sum.backward()
            G_optimizer.step()

            Da_optimizer.zero_grad()
            A_disc = Da(A)
            A_disc_fake = Da(fake_A.detach())
            loss_sum1 = GAN_loss._d_loss(A_disc, A_disc_fake)
            loss_sum1.backward()

            Da_optimizer.step()

            Db_optimizer.zero_grad()
            B_disc = Db(B)
            B_disc_fake = Db(fake_B.detach())
            loss_sum2 = GAN_loss._d_loss(B_disc, B_disc_fake)
            loss_sum2.backward()

            Db_optimizer.step()
            if(num % 5 == 0):
                it+=1
                print("epoch:{}/{} batch:{}/{}".format(epoch,epochs,num,batches))
                writer.add_scalar('train{}/loss_identity_B'.format(id), loss_identity_B, it)
                writer.add_scalar('train{}/loss_identity_A'.format(id), loss_identity_A, it)
                writer.add_scalar('train{}/loss_GAN_A2B'.format(id), loss_GAN_A2B, it)
                writer.add_scalar('train{}/loss_GAN_B2A'.format(id), loss_GAN_B2A, it)
                writer.add_scalar('train{}/loss_cycle_ABA'.format(id), loss_cycle_ABA, it)
                writer.add_scalar('train{}/loss_cycle_BAB'.format(id), loss_cycle_BAB, it)
                writer.add_scalar('train{}/loss_sum'.format(id), loss_sum, it)
                writer.add_scalar('train{}/loss_sum1'.format(id), loss_sum1, it)
                writer.add_scalar('train{}/loss_sum2'.format(id), loss_sum2, it)

        print(epoch)
        torch.save(Gab.state_dict(), dir_checkpoint +'checkpointGab_{}_epoch{}.pth'.format(id,epoch+30))
        torch.save(Gba.state_dict(), dir_checkpoint +'checkpointGba_{}_epoch{}.pth'.format(id,epoch+30))
        torch.save(Da.state_dict(), dir_checkpoint +'checkpointDa_{}_epoch{}.pth'.format(id,epoch+30))
        torch.save(Db.state_dict(), dir_checkpoint +'checkpointDb_{}_epoch{}.pth'.format(id,epoch+30))