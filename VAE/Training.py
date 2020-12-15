import os
import torch
import torch.nn as nn
import numpy as np
import vae_utils as utils
from VAE import VAE
import torchvision

if __name__ == '__main__':
    
    epochs = 500
    size_of_batch = 100
    dimensions = 2
    dataloader = utils.get_dataloader(size_of_batch)
    system = utils.get_system()
    steps_per_epoch = np.ceil(dataloader.dataset.__len__() / size_of_batch)
    vae_image_dir = './vae_images'

    checkpoint_dir = './checkpoints'
    
    
    utils.makedirs(vae_image_dir,checkpoint_dir)
    
    net = VAE(latent_dim = dimensions).to(system)
    optim = torch.optim.Adam(net.parameters())
    
    rec_log = []
    kl_log = []
    
    criteria = nn.BCELoss(reduction = 'sum')
    
    result = None
    for epoch_i in range(1, epochs + 1):
        for step_i, (real_image, _) in enumerate(dataloader):
            N = real_image.shape[0]
            real_image = real_image.view(N, -1).to(system)
            
            if result is None:
                result = real_image
                
            reconstructed, mu, logvar = net(real_image)
            
            reconstruction_loss = criteria(reconstructed, real_image)
            kl_loss = utils.kl_loss(mu, logvar)
            
            loss = kl_loss + reconstruction_loss
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            rec_log.append(reconstruction_loss.item())
            kl_log.append(kl_loss.item())
            
            utils.show_process(epoch_i, step_i + 1, steps_per_epoch, rec_log, kl_log)
            
        if epoch_i == 1:
            torchvision.utils.save_image(result.reshape(-1, 1, 28, 28), 
                                         os.path.join(vae_image_dir, 'orig.png'), 
                                         nrow = 10)
        reconstructed, _, _ = net(result)
        utils.save_image(reconstructed.reshape(-1, 1, 28, 28), 10, epoch_i, step_i + 1, vae_image_dir)
        image = net.decoder(torch.randn((100, 2)).to(system))
        torchvision.utils.save_image(image.reshape(-1, 1, 28, 28), 
                                         os.path.join(vae_image_dir, 'image_{}.png'.format(epoch_i)), 
                                         nrow = 10)
                
        utils.save_model(net, optim, rec_log, checkpoint_dir, 'autoencoder.ckpt')

    steps = 50
    z = utils.box_muller(steps).to(system)
    result = net.decoder(z)
    torchvision.utils.save_image(result.reshape(-1, 1, 28, 28), 
                                 os.path.join(vae_image_dir, 'manifold.png'), 
                                 nrow = steps)
        
            
    
    
    
    
    