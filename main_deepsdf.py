import os
import torch
from deepsdf.utils import *
from config.config_deepsdf import Config


if __name__ == "__main__":
    parser = Config.get_argparser()
    cfg: Config = parser.parse_args()
    os.makedirs(cfg.trainer.workspace, exist_ok=True)
    cfg.as_yaml(f"{cfg.trainer.workspace}/config.yaml")
    cfg.as_yaml(os.path.join(cfg.trainer.workspace, "config.yaml"))

    seed_everything(cfg.seed)
    from deepsdf.dataset import SDFDataset
    train_dataset = SDFDataset(cfg.data.dataset_path, size=cfg.data.train_size, num_samples_surf=cfg.data.num_samples_surf,
                               num_samples_space=cfg.data.num_samples_space)
    train_dataset.plot_dataset_sdf_slice(workspace=cfg.trainer.workspace)  # plot ground truth SDF slice

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)

    valid_dataset = SDFDataset(cfg.data.dataset_path, size=cfg.data.valid_size, num_samples_surf=cfg.data.num_samples_surf,
                               num_samples_space=cfg.data.num_samples_space)  # just a dummy
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1)

    from deepsdf.network import DeepSDF
    model = DeepSDF()
    optimizer = lambda m: torch.optim.Adam(
    m.parameters(),
    lr=cfg.optimizer.lr,
    betas=cfg.optimizer.betas,
    eps=cfg.optimizer.eps
    )
    scheduler = lambda optimizer: optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma)
    trainer = Trainer(
        name=cfg.trainer.name,
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        workspace=cfg.trainer.workspace,
        max_keep_ckpt=cfg.trainer.max_keep_ckpt,
        use_checkpoint=cfg.trainer.use_checkpoint,
        eval_interval=cfg.trainer.eval_interval,
        ema_decay=cfg.trainer.ema_decay,
        use_tensorboardX=cfg.trainer.use_tensorboardX,        
    )
    trainer.train(train_loader, valid_loader, cfg.epochs)

    trainer.save_mesh(os.path.join(cfg.trainer.workspace, 'results', 'output.ply'),  resolution=cfg.trainer.resolution)
