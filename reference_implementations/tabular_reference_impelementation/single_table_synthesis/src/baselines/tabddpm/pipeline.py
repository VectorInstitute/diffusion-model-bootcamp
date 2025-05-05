import json
import os
import time

import src
import torch
from src.baselines.tabddpm.model.gaussian_multinomial_diffusion import (
    GaussianMultinomialDiffusion,
)
from src.baselines.tabddpm.model.modules import (
    MLPDiffusion,
)
from src.baselines.tabddpm.sample import (
    recover_data,
    split_num_cat_target,
)
from src.baselines.tabddpm.train import Trainer


class TabDDPM:
    def __init__(
        self,
        dataset,
        model_type,
        model_params,
        num_numerical_features,
        num_classes,
        num_timesteps=1000,
        seed=2024,
        gaussian_loss_type="mse",
        scheduler="cosine",
        change_val=False,
        device=torch.device("cuda:0"),
    ):
        self.seed = seed
        self.dataset = dataset
        self.num_numerical_features = num_numerical_features
        self.device = device
        self.num_classes = num_classes
        self.num_timesteps = num_timesteps
        self.gaussian_loss_type = gaussian_loss_type
        self.scheduler = scheduler
        self.change_val = change_val
        print(model_params)
        self.model = self.get_model(
            model_type,
            model_params,
        )
        self.model.to(device)
        self.diffusion = GaussianMultinomialDiffusion(
            num_classes=self.num_classes,
            num_numerical_features=self.num_numerical_features,
            denoise_fn=self.model,
            gaussian_loss_type=self.gaussian_loss_type,
            num_timesteps=self.num_timesteps,
            scheduler=self.scheduler,
            device=self.device,
        )

        num_params = sum(p.numel() for p in self.diffusion.parameters())
        print("the number of parameters", num_params)

        self.diffusion.to(self.device)

    def get_model(self, model_name, model_params):
        if model_name == "mlp":
            model = MLPDiffusion(**model_params)
        else:
            raise "Unknown model!"
        return model

    def train(
        self,
        model_save_path,
        steps=1000,
        lr=0.002,
        weight_decay=1e-4,
        batch_size=1024,
    ):
        train_loader = src.prepare_fast_dataloader(
            self.dataset, split="train", batch_size=batch_size
        )

        trainer = Trainer(
            self.diffusion,
            train_loader,
            lr=lr,
            weight_decay=weight_decay,
            steps=steps,
            model_save_path=model_save_path,
            device=self.device,
        )
        trainer.run_loop()

        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

        torch.save(
            self.diffusion._denoise_fn.state_dict(),
            os.path.join(model_save_path, "model.pt"),
        )
        torch.save(
            trainer.ema_model.state_dict(),
            os.path.join(model_save_path, "model_ema.pt"),
        )

        trainer.loss_history.to_csv(
            os.path.join(model_save_path, "loss.csv"), index=False
        )

    def load_model(self, ckpt_path):
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        print("Loaded model from", ckpt_path)

    def sample(
        self,
        info_path,
        sample_save_path,
        batch_size=2000,
        num_samples=1000,
        ddim=False,
        steps=1000,
    ):
        self.diffusion.eval()

        start_time = time.time()
        if not ddim:
            x_gen = self.diffusion.sample_all(num_samples, batch_size, ddim=False)
        else:
            x_gen = self.diffusion.sample_all(
                num_samples, batch_size, ddim=True, steps=steps
            )

        syn_data = x_gen
        num_inverse = self.dataset.num_transform.inverse_transform
        cat_inverse = self.dataset.cat_transform.inverse_transform

        with open(info_path, "r") as f:
            info = json.load(f)

        syn_num, syn_cat, syn_target = split_num_cat_target(
            syn_data, info, num_inverse, cat_inverse
        )
        syn_df = recover_data(syn_num, syn_cat, syn_target, info)

        idx_name_mapping = info["idx_name_mapping"]
        idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

        syn_df.rename(columns=idx_name_mapping, inplace=True)
        end_time = time.time()

        print("Sampling time:", end_time - start_time)

        save_path = sample_save_path
        syn_df.to_csv(save_path, index=False)
