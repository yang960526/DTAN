import numpy as np
from datasets.SEG_transunet_datasetings import get_loader_btcv
import torch
import torch.nn as nn
from monai.inferers import SlidingWindowInferer
from light_training.evaluation.metric import dice
from light_training.trainer import Trainer
from monai.utils import set_determinism
from light_training.evaluation.metric import dice, hausdorff_distance_95, recall, fscore
import argparse
import yaml
from guided_diffusion.gaussian_diffusion import get_named_beta_schedule, ModelMeanType, ModelVarType, LossType
from guided_diffusion.respace import SpacedDiffusion, space_timesteps
from guided_diffusion.resample import UniformSampler
from unet.basic_unet_denose import BasicUNetDe
from unet.basic_unet import BasicUNetEncoder
from operator import add

set_determinism(123)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
data_dir = "./datasets/"

max_epoch = 300
batch_size = 1
val_every = 10
device = "cuda:0"



def compute_uncer(pred_out):
    pred_out = torch.sigmoid(pred_out)
    pred_out[pred_out < 0.001] = 0.001
    uncer_out = - pred_out * torch.log(pred_out)
    return uncer_out


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

        self.sample_diffusion = SpacedDiffusion(use_timesteps=space_timesteps(1000, [10]),
                                                betas=betas,
                                                model_mean_type=ModelMeanType.START_X,
                                                model_var_type=ModelVarType.FIXED_LARGE,
                                                loss_type=LossType.MSE,
                                                )
        self.sampler = UniformSampler(1000)

    def forward(self, image=None, x=None, pred_type=None, step=None, embedding=None, label=None):
        if pred_type == "q_sample":
            noise = torch.randn_like(x).to(x.device)
            t, weight = self.sampler.sample(x.shape[0], x.device)
            return self.diffusion.q_sample(x, t, noise=noise), t, noise

        elif pred_type == "denoise":
            embeddings = self.embed_model(image)
            return self.model(label=label, x=x, t=step, image=image, embeddings=embeddings)

        elif pred_type == "ddim_sample":
            embeddings = self.embed_model(image)

            uncer_step = 2


            sample_outputs = []
            for i in range(uncer_step):
                sample_outputs.append(self.sample_diffusion.ddim_sample_loop(self.model, (1, 1, 256, 256),
                                                                             model_kwargs={"label": label,
                                                                                           "image": image,
                                                                                           "embeddings": embeddings}))

            sample_return = torch.zeros((1, 1, 256, 256))

            for index in range(10):
                uncer_out = 0
                for i in range(uncer_step):
                    uncer_out += sample_outputs[i]["all_model_outputs"][index]
                uncer_out = uncer_out / uncer_step
                uncer = compute_uncer(uncer_out).cpu()

                w = torch.exp(torch.sigmoid(torch.tensor((index + 1) / 10)) * (1 - uncer))

                for i in range(uncer_step):
                    sample_return += w * sample_outputs[i]["all_samples"][index].cpu()

            return sample_return / uncer_step


def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)


def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)


def F2(y_true, y_pred, beta=2):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (1 + beta ** 2.) * (p * r) / float(beta ** 2 * p + r + 1e-15)


def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


from sklearn.metrics import accuracy_score


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


from medpy import metric


def precision(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_pred.sum() + 1e-15)


def recall(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    return (intersection + 1e-15) / (y_true.sum() + 1e-15)


def F2(y_true, y_pred, beta=2):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return (1 + beta ** 2.) * (p * r) / float(beta ** 2 * p + r + 1e-15)


def dice_score(y_true, y_pred):
    return (2 * (y_true * y_pred).sum() + 1e-15) / (y_true.sum() + y_pred.sum() + 1e-15)


def jac_score(y_true, y_pred):
    intersection = (y_true * y_pred).sum()
    union = y_true.sum() + y_pred.sum() - intersection
    return (intersection + 1e-15) / (union + 1e-15)


from sklearn.metrics import accuracy_score


class BraTSTrainer(Trainer):
    def __init__(self, env_type, max_epochs, batch_size, device="cpu", val_every=1, num_gpus=1, logdir="./logs/",
                 master_ip='localhost', master_port=17750, training_script="train.py"):
        super().__init__(env_type, max_epochs, batch_size, device, val_every, num_gpus, logdir, master_ip, master_port,
                         training_script)
        self.window_infer = SlidingWindowInferer(roi_size=[256, 256],
                                                 sw_batch_size=1,
                                                 overlap=0.5)

        self.model = DiffUNet()

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
        polyp_sizes = polyp_sizes.float()
        return image, label, label_text, num_polyps, polyp_sizes

    def validation_step(self, val_name, batch):
       
        image, label, label_text, num_polyps, polyp_sizes = self.get_input(batch)

        output = self.window_infer(image, self.model, label=label_text, pred_type="ddim_sample")
        output = torch.sigmoid(output)

        output = (output > 0.5).float().cpu().numpy()

        target = label.cpu().numpy()

        dices = []
        hd = []
        score_jaccards = []
        score_f1s = []
        score_recalls = []
        score_precisions = []
        score_fbetas = []

        c = 1
        for i in range(0, c):
            pred = output[:, i]
            gt = target[:, i]
            if pred.sum() > 0 and gt.sum() > 0:
                score_jaccard = jac_score(gt, pred)
                score_f1 = dice_score(gt, pred)
                score_recall = recall(gt, pred)
                score_precision = precision(gt, pred)
                score_fbeta = F2(gt, pred)
                pred = pred.squeeze()
                pred_normalized = (pred - pred.min()) / (pred.max() - pred.min()) * 255
                pred = pred_normalized.astype(np.uint8)
                name = val_name.split("/")[-1].split(".")[0]
                print(name)
                pred_path = os.path.join(save_path, f"{name}.jpg")
                cv2.imwrite(pred_path, pred)

            elif pred.sum() > 0 and gt.sum() == 0:
                dice = 1
                hd95 = 0
            else:
                dice = 0
                hd95 = 0

            score_jaccards.append(score_jaccard)
            score_f1s.append(score_f1)
            score_recalls.append(score_recall)
            score_precisions.append(score_precision)
            score_fbetas.append(score_fbeta)

        all_m = []
        for j in score_jaccards:
            all_m.append(j)
        for f in score_f1s:
            all_m.append(f)
        for r in score_recalls:
            all_m.append(r)
        for p in score_precisions:
            all_m.append(p)
        for f in score_fbetas:
            all_m.append(f)
        print(all_m)
        return all_m


import cv2


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


if __name__ == "__main__":
    train_ds, val_ds, val_name = get_loader_btcv(data_dir=data_dir)
    # print(val_name)

    trainer = BraTSTrainer(env_type="pytorch",
                           max_epochs=max_epoch,
                           batch_size=batch_size,
                           device=device,
                           val_every=val_every,
                           num_gpus=1,
                           master_port=17751,
                           training_script=__file__)

    logdir = "./logs_SEG/diffusion_seg_all_loss_embed/model/best_model_0.8850.pt"
    trainer.load_state_dict(logdir)
    save_path = f"./results/"
    create_dir(f"{save_path}/pred")
    v_mean, _ = trainer.validation_single_gpu(val_name, val_dataset=val_ds)

    print(f"v_mean is {v_mean}")
