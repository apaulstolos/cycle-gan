import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import DataParallel
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import optuna
from optuna.pruners import HyperbandPruner
from generator import Generator
from discriminator import Discriminator
from data_load import ImageTransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(dataloader, G_XY, G_YX, D_X, D_Y, g_optimizer, d_X_optimizer, d_Y_optimizer,
                num_epochs, lambda_cycle, lambda_identity, 
                output_dir="output_images", training=True):
    os.makedirs(output_dir, exist_ok=True)
    final_g_loss = None

    for epoch in range(num_epochs):
        for i, batch in enumerate(dataloader):
            real_X = batch["photo"].to(device)
            real_Y = batch["nature"].to(device)

            fake_Y = G_XY(real_X).detach()
            d_Y_real = D_Y(real_Y)
            d_Y_fake = D_Y(fake_Y)
            d_Y_loss = (nn.MSELoss()(d_Y_real, torch.ones_like(d_Y_real)) +
                        nn.MSELoss()(d_Y_fake, torch.zeros_like(d_Y_fake))) * 0.5

            fake_X = G_YX(real_Y).detach()
            d_X_real = D_X(real_X)
            d_X_fake = D_X(fake_X)
            d_X_loss = (nn.MSELoss()(d_X_real, torch.ones_like(d_X_real)) +
                        nn.MSELoss()(d_X_fake, torch.zeros_like(d_X_fake))) * 0.5

            d_X_optimizer.zero_grad()
            d_Y_optimizer.zero_grad()
            d_X_loss.backward()
            d_Y_loss.backward()
            d_X_optimizer.step()
            d_Y_optimizer.step()

            fake_Y = G_XY(real_X)
            fake_X = G_YX(real_Y)

            d_Y_fake = D_Y(fake_Y)
            d_X_fake = D_X(fake_X)
            adv_loss_XY = nn.MSELoss()(d_Y_fake, torch.ones_like(d_Y_fake))
            adv_loss_YX = nn.MSELoss()(d_X_fake, torch.ones_like(d_X_fake))

            cycle_X = G_YX(fake_Y)
            cycle_Y = G_XY(fake_X)
            cycle_loss = nn.L1Loss()(cycle_X, real_X) + nn.L1Loss()(cycle_Y, real_Y)

            identity_X = G_YX(real_X)
            identity_Y = G_XY(real_Y)
            identity_loss = nn.L1Loss()(identity_X, real_X) + nn.L1Loss()(identity_Y, real_Y)

            g_loss = adv_loss_XY + adv_loss_YX + lambda_cycle * cycle_loss + lambda_identity * identity_loss

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            if training and epoch % 5 == 0 and i == 0:
                with torch.no_grad():
                    fake_Y_vis = G_XY(real_X)
                    fake_X_vis = G_YX(real_Y)

                    grid_X = make_grid(torch.cat([real_X[:4], fake_Y_vis[:4]], dim=0), nrow=4, normalize=True, value_range=(-1, 1))
                    save_image(grid_X, f"{output_dir}/epoch_{epoch+1}_X_to_Y.png")

                    grid_Y = make_grid(torch.cat([real_Y[:4], fake_X_vis[:4]], dim=0), nrow=4, normalize=True, value_range=(-1, 1))
                    save_image(grid_Y, f"{output_dir}/epoch_{epoch+1}_Y_to_X.png")

        if training:
            print(f"Epoch [{epoch+1}/{num_epochs}] | D Loss: {(d_X_loss + d_Y_loss).item():.4f} | G Loss: {g_loss.item():.4f}")
        
        final_g_loss = g_loss.item()

    return final_g_loss



def objective(trial):
    batch_size = trial.suggest_int("batch_size", 4, 16, step=4)
    lr_G = trial.suggest_float("lr_G", 1e-5, 1e-3, log=True)
    lr_D = trial.suggest_float("lr_D", 1e-5, 1e-3, log=True)
    beta1 = trial.suggest_float("beta1", 0.3, 0.9)
    beta2 = trial.suggest_float("beta2", 0.8, 0.999)
    lambda_cycle = trial.suggest_float("lambda_cycle", 5, 15)
    lambda_identity = trial.suggest_float("lambda_identity", 0.1, 1)

    dataset = ImageTransform("input_photos/cubism", "input_photos/nature")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    G_XY = Generator(img_channels=3).to(device)
    G_YX = Generator(img_channels=3).to(device)
    D_X = Discriminator(in_channels=3).to(device)
    D_Y = Discriminator(in_channels=3).to(device)

    # Data parallelism
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for data parallelism")
        G_XY = nn.DataParallel(G_XY)
        G_YX = nn.DataParallel(G_YX)
        D_X = nn.DataParallel(D_X)
        D_Y = nn.DataParallel(D_Y)

    g_optimizer = optim.Adam(list(G_XY.parameters()) + list(G_YX.parameters()), lr=lr_G, betas=(beta1, beta2))
    d_X_optimizer = optim.Adam(D_X.parameters(), lr=lr_D, betas=(beta1, beta2))
    d_Y_optimizer = optim.Adam(D_Y.parameters(), lr=lr_D, betas=(beta1, beta2))

    num_epochs = 5
    g_loss = train_model(dataloader, G_XY, G_YX, D_X, D_Y, g_optimizer, d_X_optimizer, d_Y_optimizer, num_epochs, lambda_cycle, lambda_identity, training=False)

    return g_loss


if __name__ == "__main__":
    # Hyperband
    study = optuna.create_study(direction="minimize", pruner=HyperbandPruner())
    study.optimize(objective, n_trials=20)

    best_params = study.best_trial.params
    with open("best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)

    print("Best trial:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")

    dataset = ImageTransform("input_photos/cubism", "input_photos/nature")
    dataloader = DataLoader(dataset, batch_size=best_params['batch_size'], shuffle=True, num_workers=2)

    G_XY = Generator(img_channels=3).to(device)
    G_YX = Generator(img_channels=3).to(device)
    D_X = Discriminator(in_channels=3).to(device)
    D_Y = Discriminator(in_channels=3).to(device)

    # Data parallelism
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for data parallelism")
        G_XY = nn.DataParallel(G_XY)
        G_YX = nn.DataParallel(G_YX)
        D_X = nn.DataParallel(D_X)
        D_Y = nn.DataParallel(D_Y)

    g_optimizer = optim.Adam(list(G_XY.parameters()) + list(G_YX.parameters()), lr=best_params['lr_G'], betas=(best_params['beta1'], best_params['beta2']))
    d_X_optimizer = optim.Adam(D_X.parameters(), lr=best_params['lr_D'], betas=(best_params['beta1'], best_params['beta2']))
    d_Y_optimizer = optim.Adam(D_Y.parameters(), lr=best_params['lr_D'], betas=(best_params['beta1'], best_params['beta2']))

    adv_criterion = nn.MSELoss()
    cycle_criterion = nn.L1Loss()
    identity_criterion = nn.L1Loss()

    train_model(dataloader, G_XY, G_YX, D_X, D_Y, g_optimizer, d_X_optimizer, d_Y_optimizer, num_epochs=100, lambda_cycle=best_params['lambda_cycle'], lambda_identity=best_params['lambda_identity'],
                training=True)
