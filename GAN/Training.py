import os
import torch
import torch.nn as nn
import numpy as np
import gan_utils as utils
from GAN import Generator, Discriminator
import torchvision

if __name__ == '__main__':
    
    epochs = 100
    size_of_batch = 100 
    dimensions = 100
    dataloader = utils.get_dataloader(size_of_batch)
    system = utils.get_system()
    steps_per_epoch = np.ceil(dataloader.dataset.__len__() / size_of_batch)
    gan_image_dir = './gan_images'
    checkpoint_dir = './checkpoints'
    
    utils.makedirs(gan_image_dir, checkpoint_dir)
    
    Gen = Generator(dimensions = dimensions).to(system)
    Dis = Discriminator().to(system)
    
    gen_optim = utils.get_optim(Gen, 0.0002)
    dis_optim = utils.get_optim(Dis, 0.0002)
    
    gen_log = []
    dis_log = []

    criteria = nn.BCELoss()
    
    fix_z = torch.randn(size_of_batch, dimensions).to(system)
    for epoch_i in range(1, epochs + 1):
        for step, (real_image, _) in enumerate(dataloader):
            
            real_labels = torch.ones(size_of_batch).to(system)
            fake_labels = torch.zeros(size_of_batch).to(system)
            
            # Train D
            
            real_image = real_image.to(system)
            z = torch.randn(size_of_batch, dimensions).to(system)
            fake_image = Gen(z)
            
            real_score = Dis(real_image)
            fake_score = Dis(fake_image)
            
            real_loss = criteria(real_score, real_labels)
            fake_loss = criteria(fake_score, fake_labels)
            
            dis_loss = real_loss + fake_loss
            
            dis_optim.zero_grad()
            dis_loss.backward()
            dis_optim.step()
            dis_log.append(dis_loss.item())
            
            # Train G
            
            z = torch.randn(size_of_batch, dimensions).to(system)
            fake_image = Gen(z)
            
            fake_score = Dis(fake_image)
            
            gen_loss = criteria(fake_score, real_labels)
            
            gen_optim.zero_grad()
            gen_loss.backward()
            gen_optim.step()
            gen_log.append(gen_loss.item())
            
            utils.show_process(epoch_i, step + 1, steps_per_epoch ,gen_log ,dis_log)
        
        if epoch_i == 1:
            torchvision.utils.save_image(real_image, 
                                         os.path.join(gan_image_dir, 'real.png'),
                                         nrow = 10)
        fake_image = Gen(fix_z)
        utils.save_image(fake_image, 10, epoch_i, step + 1, gan_image_dir)

        utils.save_model(Gen, gen_optim, gen_log, checkpoint_dir, 'Gen.ckpt')
        utils.save_model(Dis, dis_optim, dis_log, checkpoint_dir, 'Dis.ckpt')
                
       
            
    
    
    
    
    