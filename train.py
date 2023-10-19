import numpy as np
from datasets.SEG_transunet_datasetings import get_loader_btcv
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.utils.lr_scheduler import LinearWarmupCosineAnnealingLR
from light_training.utils.files_helper import save_new_model_and_delete_last
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
import argparse
from monai.losses.dice import DiceLoss
import yaml
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler

set_determinism(123)
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'

data_dir = "./datasets/"
logdir = "./logs_SEG/diffusion_seg_all_loss_embed/"

model_save_path = os.path.join(logdir, "model")

env = "DDP"  # or env = "pytorch" if you only have one gpu.

max_epoch = 3000
batch_size = 8
val_every = 300
num_gpus = 1
device = "cuda:5"


class DiffUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_model = BasicUNetEncoder(2, 1, 2, [64, 128, 256, 512, 1024, 128])

        self.model = BasicUNetDe(2, 4, 1, [64, 128, 256, 512, 1024, 128],
                                 act=("LeakyReLU", {"negative_slope": 0.1, "inplace": False}))

        betas = get_named_beta_schedule("linear", 1000)
        self.diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [1000]),
                                         betas=betas,
                                         model_mean_type=ModelMeanType.START_X,
                                         model_var_type=ModelVarType.FIXED_LARGE,
                                         loss_type=LossType.MSE,
                                         )

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [50]),
                                                betas=betas,
                                                model_mean_type=ModelMeanType.START_X,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                loss_type=LossType.MSE,
                                                )
        self.sampler = UniformSampler(1000)

    def forward(self,image=None, x=None, pred_type=None, step=None,label=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(label=label,x=x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            sample_out = self.sample_diffusion.ddim_sample_loop(self.model, (1, 1, 256, 256),
                                                                model_kwargs={"label":label,"image": image, "embeddings": embeddings})
            sample_out = sample_out["pred_xstart"]
            return sample_out

from metrics import precision, recall, F2, dice_score, jac_score
from sklearn.metrics import accuracy_score
from light_training.evaluation.metric import dice, hausdorff_distance_95
class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cuda", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=29050, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port,
                         training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[256, 256],
                                                 sw_batch_size=1,
                                                 overlap=0.25)
        self.model = DiffUNet()

        self.best_mean_dice = 0.0
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=1e-3)
        self.ce = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.scheduler = LinearWarmupCosineAnnealingLR(self.optimizer,
                                                       warmup_epochs=30,
                                                       max_epochs=max_epochs)

        self.bce = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss(sigmoid=True)

    def training_step(self, batch):
        image, label,label_text,num_polyps,polyp_sizes = self.get_input(batch)
        x_start = label

        x_start = (x_start) * 2 - 1
        x_t, t, noise = self.model(label=label_text,x=x_start, pred_type="q_sample")
        pred_xstart = self.model(label=label_text,x=x_t, step=t, image=image, pred_type="denoise")
        loss_dice = self.dice_loss(pred_xstart, label)
        loss_bce = self.bce(pred_xstart, label)

        pred_xstart = torch.sigmoid(pred_xstart)
        loss_mse = self.mse(pred_xstart, label)

        loss = loss_dice + loss_bce + loss_mse

        self.log("train_loss", loss, step=self.global_step)

        return loss

    def get_input(self, batch):
        image = batch[0]
        label = batch[1]
        label_text = batch[2]
        num_polyps = batch[3]
        polyp_sizes = batch[4]

        label = label.float()
        image = image.float()
        label_text = label_text.float()
        num_polyps = num_polyps.float()
        polyp_sizes =polyp_sizes.float()
        return image, label,label_text,num_polyps,polyp_sizes


    def calculate_metrics(y_true, y_pred):
        y_true = y_true.detach().cpu().numpy()
        y_pred = y_pred.detach().cpu().numpy()

        y_pred = y_pred > 0.5
        y_pred = y_pred.reshape(-1)
        y_pred = y_pred.astype(np.uint8)

        y_true = y_true > 0.5
        y_true = y_true.reshape(-1)
        y_true = y_true.astype(np.uint8)

        ## Score
        score_jaccard = jac_score(y_true, y_pred)
        score_f1 = dice_score(y_true, y_pred)
        score_recall = recall(y_true, y_pred)
        score_precision = precision(y_true, y_pred)
        score_fbeta = F2(y_true, y_pred)
        score_acc = accuracy_score(y_true, y_pred)

        return [score_jaccard, score_f1, score_recall, score_precision, score_acc, score_fbeta]
    def validation_step(self, batch):
        image, label,label_text,num_polyps,polyp_sizes = self.get_input(batch)

        output = self.window_infer(image, self.model,label=label_text,pred_type="ddim_sample")

        output = torch.sigmoid(output)

        output = (output > 0.5).float().cpu().numpy()

        target = label.cpu().numpy()
        dices = []
        hd = []
        c = 1
        for i in range(0, c):
            pred_c = output[:, i]
            target_c = target[:, i]

            dices.append(dice(pred_c, target_c))
            hd.append(hausdorff_distance_95(pred_c, target_c))
        return dices


    def validation_end(self, mean_val_outputs):
        dices = mean_val_outputs
        print(dices)
        mean_dice = sum(dices) / len(dices)

        self.log("mean_dice", mean_dice, step=self.epoch)

        if mean_dice > self.best_mean_dice:
            self.best_mean_dice = mean_dice
            save_new_model_and_delete_last(self.model,
                                           os.path.join(model_save_path,
                                                        f"best_model_{mean_dice:.4f}.pt"),
                                           delete_symbol="best_model")

        save_new_model_and_delete_last(self.model,
                                       os.path.join(model_save_path,
                                                    f"final_model_{mean_dice:.4f}.pt"),
                                       delete_symbol="final_model")

        print(f" mean_dice is {mean_dice}")


if __name__ == "__main__":
    train_ds, val_ds = get_loader_btcv(data_dir=data_dir)
    trainer = BraTSTrainer(env_type=env,
                           max_epochs=max_epoch,
                           batch_size=batch_size,
                           device=device,
                           logdir=logdir,
                           val_every=val_every,
                           num_gpus=num_gpus,
                           master_port=29050,
                           training_script=__file__)

    trainer.train(train_dataset=train_ds, val_dataset=val_ds)
