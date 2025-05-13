import numpy as np
import torch
from diff_models import diff_CSDI
from torch import nn


class CSDI_base(nn.Module):
    """
    The `CSDI_base` class is a PyTorch module that facilitates Conditional Score-based Diffusion Modeling for
    time series imputation.

    Key Components and Initializations:

    - `__init__`: Constructor that initializes the model with:
      - `target_dim`: The dimensionality of the target data, dictating the model's output features.
      - `config`: Configuration settings for model parameters such as embeddings and the diffusion process.
      - `device`: Computation device (CPU or GPU) for tensor operations, ensuring optimal performance.

    Configuration and Setup:

    1. Embedding Dimensions:
       - `emb_time_dim` and `emb_feature_dim`: Define embedding sizes for time points and features,
          transforming input data for the network.
       - `is_unconditional`: Adjusts embedding dimensions based on whether the model is conditional,
          integrating observed data when necessary.

    2. Network Layers:
       - `embed_layer`: A PyTorch embedding layer that maps features to dense vectors, enhancing data pattern recognition.

    3. Diffusion Model Configuration:
       - Extracts diffusion-related settings from `config`, like the number of diffusion steps and the beta schedule,
         which dictate the noise addition process.

    4. Diffusion Parameters:
       - `beta`, `alpha_hat`, `alpha`: Arrays that manage the noise levels and data transformation during the diffusion process.
       - `alpha_torch`: Converts the `alpha` array to a PyTorch tensor formatted for use in the model's operations and
          placed on the designated device.
    """

    def __init__(self, target_dim, config, device):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        input_dim = 1 if self.is_unconditional else 2
        self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = (
                np.linspace(
                    config_diff["beta_start"] ** 0.5,
                    config_diff["beta_end"] ** 0.5,
                    self.num_steps,
                )
                ** 2
            )
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = (
            torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)
        )

    def time_embedding(self, pos, d_model=128):
        """
        Def time_embedding
        - Constructs a time-based embedding for input positions using sinusoidal functions, commonly used in models like transformers.
        - `pos`: Tensor of positions for which embeddings are generated.
        - `d_model`: Dimensionality of the embedding; defaults to 128.
        - The function uses sine and cosine functions to embed time positions, which helps in capturing periodic patterns.
        """
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        """
        Def get_randmask
        - Generates a random mask for observed data points to simulate missingness during training,
            enhancing the model's robustness.
        - `observed_mask`: A tensor indicating which data points are observed.
        - This function randomly masks observed data points based on their existing observed status,
            useful in self-supervised setups where true missing patterns need to be approximated.
        """
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        """
        Def get_hist_mask
        - Generates a historical mask based on the observed data and possibly an external pattern.
        - `observed_mask`: A tensor indicating observed data points.
        - `for_pattern_mask`: An optional tensor to provide historical patterns for masking, defaults to the observed_mask.
        - Depending on the `target_strategy`, it combines random masking and historical patterns to simulate different training conditions.

        """
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        return cond_mask

    """
    def get_test_pattern_mask
       - Produces a test pattern mask that is the element-wise product of observed data points and a specified test pattern.
       - `observed_mask`: A tensor of observed data points.
       - `test_pattern_mask`: A predefined tensor describing specific test conditions or missing data patterns.
       - This function is useful for simulating specific scenarios during model evaluation or testing.
    """

    def get_test_pattern_mask(self, observed_mask, test_pattern_mask):
        return observed_mask * test_pattern_mask

    """
    def get_side_info
       - Constructs side information by combining time and feature embeddings with an optional conditional mask.
       - `observed_tp`: Tensor of observed time points.
       - `cond_mask`: A condition mask indicating available data points.
       - This function enriches the model input with necessary contextual information, adjusting for the presence of observed data
         and enhancing the model's ability to interpret and impute based on time and feature dependencies.
    """

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)

        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if not self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    """
    def calc_loss_valid
       - Computes the validation loss by aggregating losses over all diffusion steps.
       - `observed_data`: Tensor containing the observed data points.
       - `cond_mask`, `observed_mask`: Masks indicating conditions and observed data points.
       - `side_info`: Additional contextual information provided to the diffusion model.
       - `is_train`: A flag indicating whether the function is being called in training mode.
       - Iterates over all diffusion steps, accumulating loss calculated at each step and averages it.
    """

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        """
        Def calc_loss
        - Calculates loss for a specific step in the diffusion process during training or validation.
        - `is_train`: Flag to distinguish between training and validation modes.
        - `set_t`: Specific time step at which to calculate the loss; if not in training mode, the step is predetermined.
        - Determines the diffusion time step randomly during training or uses the provided step during validation.
        - Applies the diffusion model to compute predictions for noisy data, compares these predictions against actual noise added,
            and computes the loss based on differences weighted by the target mask (areas not covered by the condition mask).
        """
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha**0.5) * observed_data + (
            1.0 - current_alpha
        ) ** 0.5 * noise

        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual**2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        """
        Def set_input_to_diffmodel
        - Prepares the input tensors for the diffusion model based on the model's conditionality.
        - `noisy_data`: Tensor containing the data perturbed with noise.
        - `observed_data`: Original data tensor containing observed values.
        - `cond_mask`: A mask that distinguishes between observed and unobserved data points.
        - If the model is unconditional, it uses only the noisy data as input.
        - For conditional models, it creates separate channels for conditioned observed data and noisy targets, then concatenates them.
        """
        if self.is_unconditional:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            cond_obs = (cond_mask * observed_data).unsqueeze(1)
            noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
            total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)

        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        """
        Def impute
        - Conducts the imputation of missing values based on observed data, condition masks, and side information.
        - `observed_data`: The observed portion of the data.
        - `cond_mask`: Mask indicating observed (1) and missing (0) parts of the data.
        - `side_info`: Contextual information that aids the model in imputation.
        - `n_samples`: Number of imputation samples to generate.
        - Initializes storage for the imputed samples and iteratively generates them using a diffusion process.
        - For unconditional models, creates noisy versions of the observed data for each diffusion step.
        - For conditional models, combines observed data and generated samples in a structured input for the diffusion model.
        - Utilizes backward diffusion (from the last step to the first) to progressively refine each imputed sample, adjusting with noise and model predictions.
        - Returns a tensor of imputed samples across all specified instances.
        """
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # generate noisy observation for unconditional model
            if self.is_unconditional:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[
                        t
                    ] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                if self.is_unconditional:
                    diff_input = (
                        cond_mask * noisy_cond_history[t]
                        + (1.0 - cond_mask) * current_sample
                    )
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    cond_obs = (cond_mask * observed_data).unsqueeze(1)
                    noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                    diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                predicted = self.diffmodel(
                    diff_input, side_info, torch.tensor([t]).to(self.device)
                )

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        """
        Def forward
        - Defines the forward pass of the model, integrating preprocessing, conditioning, and loss calculation.
        - `batch`: Input batch containing data and various masks.
        - `is_train`: Flag to indicate if the model is in training (1) or validation (0) mode.
        - Extracts observed data, masks, and time points from the batch.
        - Depending on the training mode and target strategy, determines the conditional mask for the data.
        - Computes side information based on observed time points and the conditional mask.
        - Selects the appropriate loss function based on the training mode.
        - Calculates and returns the loss, facilitating both training and validation processes.
        """
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _,
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        """
        Def evaluate
        - Conducts evaluation of the model by generating imputed samples and preparing metrics.
        - `batch`: Input batch containing observed data and masks.
        - `n_samples`: Number of imputation samples to generate for each point.
        - Processes the batch to extract data and masks, including the length for each sequence to avoid redundancy.
        - Sets up a non-gradient context for evaluation to prevent backpropagation and save computation.
        - Uses ground truth masks to determine conditional and target masks.
        - Retrieves side information based on observed time points and conditional masks.
        - Generates multiple imputed data samples.
        - Adjusts target masks to avoid evaluation on predefined cut lengths in data.
        - Returns the generated samples along with the observed data and masks for further assessment.
        """
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp


class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Forecasting(CSDI_base):
    def __init__(self, config, device, target_dim):
        super(CSDI_Forecasting, self).__init__(target_dim, config, device)
        self.target_dim_base = target_dim
        self.num_sample_features = config["model"]["num_sample_features"]

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        feature_id = (
            torch.arange(self.target_dim_base)
            .unsqueeze(0)
            .expand(observed_data.shape[0], -1)
            .to(self.device)
        )

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            feature_id,
        )

    def sample_features(self, observed_data, observed_mask, feature_id, gt_mask):
        size = self.num_sample_features
        self.target_dim = size
        extracted_data = []
        extracted_mask = []
        extracted_feature_id = []
        extracted_gt_mask = []

        for k in range(len(observed_data)):
            ind = np.arange(self.target_dim_base)
            np.random.shuffle(ind)
            extracted_data.append(observed_data[k, ind[:size]])
            extracted_mask.append(observed_mask[k, ind[:size]])
            extracted_feature_id.append(feature_id[k, ind[:size]])
            extracted_gt_mask.append(gt_mask[k, ind[:size]])
        extracted_data = torch.stack(extracted_data, 0)
        extracted_mask = torch.stack(extracted_mask, 0)
        extracted_feature_id = torch.stack(extracted_feature_id, 0)
        extracted_gt_mask = torch.stack(extracted_gt_mask, 0)
        return extracted_data, extracted_mask, extracted_feature_id, extracted_gt_mask

    def get_side_info(self, observed_tp, cond_mask, feature_id=None):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, self.target_dim, -1)

        if self.target_dim == self.target_dim_base:
            feature_embed = self.embed_layer(
                torch.arange(self.target_dim).to(self.device)
            )  # (K,emb)
            feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        else:
            feature_embed = (
                self.embed_layer(feature_id).unsqueeze(1).expand(-1, L, -1, -1)
            )
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)

        return side_info

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id,
        ) = self.process_data(batch)
        if is_train == 1 and (self.target_dim_base > self.num_sample_features):
            observed_data, observed_mask, feature_id, gt_mask = self.sample_features(
                observed_data, observed_mask, feature_id, gt_mask
            )
        else:
            self.target_dim = self.target_dim_base
            feature_id = None

        if is_train == 0:
            cond_mask = gt_mask
        else:  # test pattern
            cond_mask = self.get_test_pattern_mask(observed_mask, gt_mask)

        side_info = self.get_side_info(observed_tp, cond_mask, feature_id)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            _,
            feature_id,
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask * (1 - gt_mask)

            side_info = self.get_side_info(observed_tp, cond_mask)

            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

        return samples, observed_data, target_mask, observed_mask, observed_tp
