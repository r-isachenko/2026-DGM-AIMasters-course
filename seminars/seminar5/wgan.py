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
        for (batch_i, (real_data, _)) in enumerate(train_loader):  # Ignoring labels in MNIST
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

