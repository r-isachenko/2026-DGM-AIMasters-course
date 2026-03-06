import torch
from torch import nn
import torch.optim as optim
import numpy as np

from seminar7_utils import computePotGrad
from seminar7_utils import make_inference, visualize_GAN_output
from seminar7_utils import FullyConnectedMLP
from tqdm.notebook import tqdm

class VanillaGAN():
    def __init__(self, G, D, noise_fn, data_fn,
                 batch_size=32, device='cpu', lr_D=1e-3, lr_G=2e-4):
        """A GAN class for holding and training a generator and discriminator
        Args:
            G: a Ganerator network
            D: A Discriminator network
            noise_fn: function f(num: int) -> pytorch tensor, (latent vectors)
            data_fn: function f(num: int) -> pytorch tensor, (real samples)
            batch_size: training batch size
            device: cpu or CUDA
            lr_D: learning rate for the discriminator
            lr_G: learning rate for the generator
        """
        self.G = G
        self.G = self.G.to(device)
        self.D = D
        self.D = self.D.to(device)
        self.noise_fn = noise_fn
        self.data_fn = data_fn
        self.batch_size = batch_size
        self.device = device
        # !
        self.criterion = nn.BCELoss()
        self.optim_D = optim.Adam(D.parameters(),
                                  lr=lr_D, betas=(0.5, 0.999))
        self.optim_G = optim.Adam(G.parameters(),
                                  lr=lr_G, betas=(0.5, 0.999))
        # is needed in D train loop
        self.target_ones = torch.ones((batch_size, 1)).to(device)
        self.target_zeros = torch.zeros((batch_size, 1)).to(device)
    
    def generate_samples(self, latent_vec=None, num=None):
        """Sample from the generator.
        Args:
            latent_vec: A pytorch latent vector or None
            num: The number of samples to generate if latent_vec is None
        If latent_vec and num are None then us self.batch_size random latent
        vectors.
        ! We don't need grad for generated samples
        """
        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        # your code here
        with torch.no_grad():
            samples = self.G(latent_vec)
        return samples

    def train_step_G(self):
        """Train the generator one step and return the loss."""
        self.optim_G.zero_grad()
        latent_vec = self.noise_fn(self.batch_size)
        # your code here
        # use self.target_ones
        generated = self.G(latent_vec)
        classifications = self.D(generated)
        loss = self.criterion(classifications, self.target_ones)
        loss.backward()
        self.optim_G.step()
        return loss.item()

    def train_step_D(self):
        """Train the discriminator one step and return the losses."""
        self.optim_D.zero_grad()

        # real samples
        real_samples = self.data_fn(self.batch_size)
        # calc real loss
        # you code here
        pred_real = self.D(real_samples)
        loss_real = self.criterion(pred_real, self.target_ones)

        # generated samples
        latent_vec = self.noise_fn(self.batch_size)
        # calc fake loss
        # you shouldn't optimize G here
        # you code here
        
        with torch.no_grad():
            fake_samples = self.G(latent_vec)
        pred_fake = self.D(fake_samples)
        loss_fake = self.criterion(pred_fake, self.target_zeros)

        # combine
        loss = (loss_real + loss_fake) / 2
        loss.backward()
        self.optim_D.step()
        
        return loss_real.item(), loss_fake.item()

    def train_step(self):
        """Train both networks and return the losses."""
        loss_D = self.train_step_D()
        loss_G = self.train_step_G()
        return loss_G, loss_D


class WGAN():
    def __init__(self, G, D, noise_fn, data_fn,
                 batch_size=32, device='cpu', lr_D=5e-5, lr_G=5e-5, n_critic=5, clip_c=0.1):
        """A GAN class for holding and training a generator and discriminator
        Args:
            G: a Ganerator network
            D: A Discriminator network
            noise_fn: function f(num: int) -> pytorch tensor, (latent vectors)
            data_fn: function f(num: int) -> pytorch tensor, (real samples)
            batch_size: training batch size
            device: cpu or CUDA
            lr_D: learning rate for the discriminator
            lr_G: learning rate for the generator
        """
        self.G = G
        self.G = self.G.to(device)
        self.D = D
        self.D = self.D.to(device)
        self.noise_fn = noise_fn
        self.data_fn = data_fn
        self.batch_size = batch_size
        self.device = device
        
        #self.optim_D = optim.Adam(D.parameters(),
        #                          lr=lr_D, betas=(0.5, 0.999))
        #self.optim_G = optim.Adam(G.parameters(),
        #                          lr=lr_G, betas=(0.5, 0.999))
        
        self.optim_D = optim.RMSprop(D.parameters(),
                                  lr=lr_D)
        self.optim_G = optim.RMSprop(G.parameters(),
                                  lr=lr_G)
        
        
        # is needed in D train loop
        self.target_ones = torch.ones((batch_size, 1)).to(device)
        self.target_zeros = torch.zeros((batch_size, 1)).to(device)
    
        self.n_critic = n_critic
        self.clip_c = clip_c
    
    def generate_samples(self, latent_vec=None, num=None):
        """Sample from the generator.
        Args:
            latent_vec: A pytorch latent vector or None
            num: The number of samples to generate if latent_vec is None
        If latent_vec and num are None then us self.batch_size random latent
        vectors.
        ! We don't need grad for generated samples
        """
        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        # your code here
        with torch.no_grad():
            samples = self.G(latent_vec)
        return samples

    def train_step_G(self):
        """Train the generator one step and return the loss."""
        self.optim_G.zero_grad()
        latent_vec = self.noise_fn(self.batch_size)
        # your code here
        # use self.target_ones
        generated = self.G(latent_vec)
        classifications = self.D(generated)
        loss = -1 * classifications.mean()
        loss.backward()
        self.optim_G.step()
        return -1*loss.item()

    def train_step_D(self):
        """Train the discriminator one step and return the losses."""
        self.optim_D.zero_grad()

        # real samples
        real_samples = self.data_fn(self.batch_size)
        # calc real loss
        # you code here
        pred_real = self.D(real_samples)
        loss_real = -1*pred_real.mean()

        # generated samples
        latent_vec = self.noise_fn(self.batch_size)
        # calc fake loss
        # you shouldn't optimize G here
        # you code here
        
        with torch.no_grad():
            fake_samples = self.G(latent_vec)
        pred_fake = self.D(fake_samples)
        loss_fake = pred_fake.mean()

        # combine
        loss = loss_real + loss_fake
        loss.backward()
        self.optim_D.step()
        
        # clip weights
        if self.clip_c is not None:
            for p in self.D.parameters():
                p.data.clamp_(-self.clip_c, self.clip_c)
        
        return -1*loss_real.item(), loss_fake.item()

    def train_step(self):
        """Train both networks and return the losses."""
        loss_D = []
        for i in range(self.n_critic):
            loss_D.append(self.train_step_D())
        loss_D = np.mean(loss_D, axis=0)
        loss_G = self.train_step_G()
        return loss_G, loss_D
    
from torch import autograd

class WGAN_GP():
    def __init__(self, G, D, noise_fn, data_fn,
                 batch_size=32, device='cpu', lr_D=1e-4, lr_G=1e-4, n_critic=5, Lambda=10):
        """A GAN class for holding and training a generator and discriminator
        Args:
            G: a Ganerator network
            D: A Discriminator network
            noise_fn: function f(num: int) -> pytorch tensor, (latent vectors)
            data_fn: function f(num: int) -> pytorch tensor, (real samples)
            batch_size: training batch size
            device: cpu or CUDA
            lr_D: learning rate for the discriminator
            lr_G: learning rate for the generator
        """
        self.G = G
        self.G = self.G.to(device)
        self.D = D
        self.D = self.D.to(device)
        self.noise_fn = noise_fn
        self.data_fn = data_fn
        self.batch_size = batch_size
        self.device = device
        
        self.optim_D = optim.Adam(D.parameters(),
                                  lr=lr_D, betas=(0.5, 0.999))
        self.optim_G = optim.Adam(G.parameters(),
                                  lr=lr_G, betas=(0.5, 0.999))
        
        
        # is needed in D train loop
        self.target_ones = torch.ones((batch_size, 1)).to(device)
        self.target_zeros = torch.zeros((batch_size, 1)).to(device)
    
        self.n_critic = n_critic
        self.Lambda = Lambda
    
    def generate_samples(self, latent_vec=None, num=None):
        """Sample from the generator.
        Args:
            latent_vec: A pytorch latent vector or None
            num: The number of samples to generate if latent_vec is None
        If latent_vec and num are None then us self.batch_size random latent
        vectors.
        ! We don't need grad for generated samples
        """
        num = self.batch_size if num is None else num
        latent_vec = self.noise_fn(num) if latent_vec is None else latent_vec
        # your code here
        with torch.no_grad():
            samples = self.G(latent_vec)
        return samples

    def train_step_G(self):
        """Train the generator one step and return the loss."""
        self.optim_G.zero_grad()
        latent_vec = self.noise_fn(self.batch_size)
        # your code here
        # use self.target_ones
        generated = self.G(latent_vec)
        classifications = self.D(generated)
        loss = -1 * classifications.mean()
        loss.backward()
        self.optim_G.step()
        return -1 * loss.item()

    def train_step_D(self):
        """Train the discriminator one step and return the losses."""
        self.optim_D.zero_grad()

        # real samples
        real_samples = self.data_fn(self.batch_size)
        # calc real loss
        # you code here
        pred_real = self.D(real_samples)
        loss_real = -1 * pred_real.mean()

        # generated samples
        latent_vec = self.noise_fn(self.batch_size)
        # calc fake loss
        # you shouldn't optimize G here
        # you code here
        
        with torch.no_grad():
            fake_samples = self.G(latent_vec)
            
        pred_fake = self.D(fake_samples)
        loss_fake = pred_fake.mean()

        loss_gp = self.calc_gradient_penalty(real_samples, fake_samples)
        
        # combine
        loss = loss_real + loss_fake + self.Lambda * loss_gp
        loss.backward()
        self.optim_D.step()
        
        return -1*loss_real.item(), loss_fake.item(), loss_gp.item()

    
    def calc_gradient_penalty(self, real_samples, fake_samples):
        alpha = torch.rand(self.batch_size, 1, device=self.device)
        alpha = alpha.expand(real_samples.size())

        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples))        
        interpolates = autograd.Variable(interpolates, requires_grad=True)
        
        disc_interpolates = self.D(interpolates)
        
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        # should we change dim if deal with 2d data?
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_step(self):
        """Train both networks and return the losses."""
        loss_D = []
        for i in range(self.n_critic):
            loss_D.append(self.train_step_D())
        loss_D = np.mean(loss_D, axis=0)
        loss_G = self.train_step_G()
        return loss_G, loss_D
    

###

class WGAN_MLPCritic(FullyConnectedMLP):

    def clip_weights(self, c):
        for layer in self.net:
            if isinstance(layer, torch.nn.Linear):
                layer.weight.data = torch.clamp(layer.weight.data, -c, c)
                layer.bias.data = torch.clamp(layer.bias.data, -c, c)

def train_wgan(
    generator, 
    critic, 
    train_loader,
    critic_steps, 
    batch_size,
    n_epochs,
    lr, 
    clip_c, 
    use_cuda = True,
    visualize_steps=10):

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0, 0.9))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0, 0.9))

    generator.train()
    critic.train()

    curr_iter = 0
    d_loss, g_loss = torch.zeros(1), torch.zeros(1)
    batch_loss_history = {'discriminator_losses': [], 'generator_losses': []}

    for epoch_i in tqdm(range(n_epochs)):
        for (batch_i, x) in enumerate(train_loader):
            curr_iter += 1
            if use_cuda:
                x = x.cuda()

            # CRITIC UPDATE
            with torch.no_grad():
                fake_data = generator.sample(x.shape[0])

            critic_optimizer.zero_grad()
            d_loss = (critic(fake_data) - critic(x)).mean()
            d_loss.backward()
            critic_optimizer.step()
            critic.clip_weights(clip_c)

            # GENERATOR UPDATE
            if curr_iter % critic_steps == 0:
                gen_optimizer.zero_grad()
                fake_data = generator.sample(batch_size)
                g_loss = -critic(fake_data).mean()
                g_loss.backward()
                gen_optimizer.step()

                batch_loss_history['generator_losses'].append(g_loss.data.cpu().numpy())
                batch_loss_history['discriminator_losses'].append(d_loss.data.cpu().numpy())
        if visualize_steps and epoch_i % visualize_steps == 0:
            print('Epoch {}'.format(epoch_i))
            samples, grid, critic_output, critic_grad_norms = make_inference(generator, critic)
            visualize_GAN_output(samples, train_loader.dataset, grid, critic_output, critic_grad_norms)

    return batch_loss_history



def train_wgan_gp_mnist(
    generator, 
    critic,
    train_loader,
    critic_steps=5,
    batch_size=64,
    n_epochs=50,
    lr=0.0001,
    lambda_gp=10,  # Gradient penalty lambda
    visualize_steps=5):
    
    # Use Adam optimizer with beta1=0, beta2=0.9 for WGAN-GP
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.9))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=lr, betas=(0.0, 0.9))
    
    generator.train()
    critic.train()
    curr_iter = 0
    c_loss, g_loss = torch.zeros(1), torch.zeros(1)
    batch_loss_history = {'critic_losses': [], 'generator_losses': []}
    
    for epoch_i in tqdm(range(n_epochs)):
        for (batch_i, real_data) in enumerate(train_loader):  # Ignoring labels in MNIST
            curr_iter += 1
            
            # Flatten MNIST images
            real_data = real_data.view(real_data.size(0), -1).to(DEVICE)
            
            # CRITIC UPDATE
            for _ in range(critic_steps):
                critic_optimizer.zero_grad()
                
                # Real data
                pred_real = critic(real_data)
                
                # Generated data
                fake_data = generator.sample(real_data.shape[0])
                pred_fake = critic(fake_data)
                
                # Compute gradient penalty
                batch_size = real_data.size(0)
                alpha = torch.rand(batch_size, 1, device=DEVICE)
                # Expand alpha to match dimensions of real_data
                alpha = alpha.expand_as(real_data)
                
                # Interpolate between real and fake data
                interpolates = alpha * real_data + ((1 - alpha) * fake_data)
                interpolates.requires_grad_(True)
                
                # Calculate critic output for interpolated data
                disc_interpolates = critic(interpolates)
                
                # Calculate gradients with respect to inputs
                gradients = torch.autograd.grad(
                    outputs=disc_interpolates,
                    inputs=interpolates,
                    grad_outputs=torch.ones_like(disc_interpolates),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True
                )[0]
                
                # Calculate gradient penalty
                gradients = gradients.view(batch_size, -1)
                gradient_penalty = lambda_gp * ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                
                # Wasserstein loss with gradient penalty
                c_loss = torch.mean(pred_fake) - torch.mean(pred_real) + gradient_penalty
                c_loss.backward()
                critic_optimizer.step()
            
            # GENERATOR UPDATE
            gen_optimizer.zero_grad()
            fake_data = generator.sample(batch_size)
            pred_fake = critic(fake_data)
            
            # Generator aims to maximize critic's output
            g_loss = -torch.mean(pred_fake)
            g_loss.backward()
            gen_optimizer.step()
            
            batch_loss_history['generator_losses'].append(g_loss.item())
            batch_loss_history['critic_losses'].append(c_loss.item())
        
        if visualize_steps and epoch_i % visualize_steps == 0:
            print(f'Epoch {epoch_i}')
            print(f'Critic Loss: {c_loss.item():.4f}, Generator Loss: {g_loss.item():.4f}')
            
            # Visualize generated images
            with torch.no_grad():
                fake_images = generator.sample(16)
                fake_images = fake_images.view(16, 1, 28, 28)  # Reshape to MNIST image format
                
                # Create a grid of images
                grid = torchvision.utils.make_grid(fake_images, nrow=4, normalize=True)
                plt.figure(figsize=(8, 8))
                plt.imshow(grid.permute(1, 2, 0).cpu())
                plt.axis('off')
                plt.show()
    
    return batch_loss_history

