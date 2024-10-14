import os
import torch
import numpy as np
from data_management import get_dataloader_EEGED
from convnext import ConvNeXt1D

from torch.optim import Adam
from torch.nn.functional import one_hot, cross_entropy
from helper_modules import ContrastiveLoss, OwnContrastiveLoss

from argparse import ArgumentParser
from train_utils import TrainSaveObject

def train(parser: ArgumentParser) -> None:
    # Parsing
    args = parser.parse_args()
    data_folder = args.data_folder
    sample_duration = args.sample_duration
    batch_size = args.batch_size
    lr = args.lr
    n_epochs = args.n_epochs
    save_folder = args.save_folder
    negative_mode = args.negative_mode
    time_delta = args.time_delta
    fs = args.fs
    num_workers = args.num_workers
    
    # Loader 
    loader = get_dataloader_EEGED(folder_in=data_folder, negative_mode=negative_mode,
                                  max_positive_delta=time_delta, sample_duration=sample_duration,
                                  batch_size=batch_size, fs=fs, num_workers=num_workers)
    
    # Get train dataset specific parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_channels = loader.dataset[0][0].shape[0]
    num_subjects = len(np.unique(loader.dataset.ids))
    num_classes = len(np.unique(loader.dataset.drugs))
    
    # Setup model
    depths = [3, 3, 9, 3]
    dims = [96, 192, 384, 768]
    drop_path_rate = 0
    layer_scale_init_value = 0.000001
    head_init_scale = 1
    return_embedding = True

    model = ConvNeXt1D(in_chans=n_channels,
                       num_classes=num_classes, num_subjects=num_subjects,
                       depths=depths, dims=dims, drop_path_rate=drop_path_rate,
                       layer_scale_init_value=layer_scale_init_value, head_init_scale=head_init_scale,
                       return_embedding=return_embedding).float().to(device)
    
    model_config = {"in_chans": n_channels, "num_classes": num_classes, "num_subjects": num_subjects,
                    "depths": depths, "dims": dims, "drop_path_rate": drop_path_rate,
                    "layer_scale_init_value": layer_scale_init_value, "head_init_scale": head_init_scale,
                    "return_embedding": return_embedding}
    

    optimizer = Adam(model.parameters(), lr=lr)

    # TODO - implement better garbage
    loss_fcn = OwnContrastiveLoss()
    
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
        

    # Train loop
    for n in range(n_epochs):
        # Save arrays
        L_emb = []
        L_cls = []
        L_dmn = []
        L_all = []
        ACC = []
        
        # Epoch iterations
        model = model.train().to(device)
        for it, (A, P, N, sA, sP, sN, cA, cP, cN) in enumerate(loader):
            optimizer.zero_grad()
            
            # Prepare data for training
            X = torch.concatenate((A, P, N)).float().to(device)
            T_subject = one_hot(torch.concatenate((sA, sP, sN)), num_classes=num_subjects).float().to(device)
            T_class = one_hot(torch.concatenate((cA, cP, cN)), num_classes=num_classes).float().to(device)
            
            # Forward
            E, Y_subject, Y_class = model(X)
            
            # Loss calculation
            Ea, Ep, En = torch.chunk(E, 3) # Chunk embeddings based on 
            loss_domain = cross_entropy(Y_subject, T_subject)
            loss_class = cross_entropy(Y_class, T_class)
            loss_embed = loss_fcn(Ea, Ep, En)
            loss = loss_embed + loss_domain + loss_class
            loss.backward()
            
            optimizer.step()
            
            # Epoch stats
            with torch.no_grad():
                acc = 100.0 * (torch.argmax(Y_class, dim=1) == torch.argmax(T_class, dim=1)).float().mean()
                L_emb.append(loss_embed.item())
                L_cls.append(loss_class.item())
                L_dmn.append(loss_domain.item())
                L_all.append(loss.item())
                ACC.append(acc.item())
                
                # DEBUG
                print(f"\rE:{n+1} | IT: {it + 1} | LE: {loss_embed.item():.4f} LD: {loss_domain.item():.4f} LC: {loss_class.item():.4f} (acc = {acc:.2f})", end="                      ")
                # DEBUG - END
            
        # Save epoch stats
        train_save = TrainSaveObject(epoch=n+1,
                                        model=model,
                                        lr=optimizer.param_groups[0]["lr"],
                                        L_emb=L_emb,
                                        L_cls=L_cls,
                                        L_dmn=L_dmn,
                                        L_all=L_all,
                                        ACC=ACC,
                                        model_config=model_config)
        train_save.save(save_folder)
        


if __name__ == "__main__":
    parser = ArgumentParser(description="Server argument parses")
    parser.add_argument("--data_folder", default="data", type=str) 
    parser.add_argument("--sample_duration", default=15, type=float)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--n_epochs", default=100, type=int)
    parser.add_argument("--save_folder", default="train_progress", type=str)
    parser.add_argument("--negative_mode", default="easy", type=str, choices=["easy", "hard", "all"])
    parser.add_argument("--time_delta", default=15, type=float)
    parser.add_argument("--fs", default=250, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    
    train(parser)