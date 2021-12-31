#!/usr/bin/env python

import argparse
import numpy as np
import torch
import typing
import torch.nn.functional as F
from clip_files import model, clip
import pickle

from utils import *

from torchvision.datasets import ImageNet
from pytorch_lightning import Trainer, LightningModule, LightningDataModule, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data  import DataLoader

from sklearn.linear_model import Lasso

class ContrastiveUnsupervisedDataset(torch.utils.data.Dataset):
    """
    This class takes a dataset and creates a contrastive version of that dataset.
    Each item of the dataset is a tuple of a clean image and a noisy image (two
    separate transformations.)
    """
    def __init__(self, clean_dataset, transform_contrastive=None, return_label=False):
        self.base = clean_dataset
        self.transform_contrastive = transform_contrastive
        self.return_label = return_label

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        image_orig, label = self.base[idx]
        image_clean, image_noisy = self.transform_contrastive(image_orig) if self.transform_contrastive is not None else (image_orig, image_orig)
        if self.return_label:
            return image_clean, image_noisy, label
        else:
            return image_clean, image_noisy

class ImageNetCLIPDataset(LightningDataModule):
    """
    Wrapper class for the ImageNet dataset, handles all data manipulations
    required in order to train the NoisyCLIP model.
    """
    def __init__(self, args):
        super(ImageNetCLIPDataset, self).__init__()

        self.hparams = args

        self.dataset_dir = self.hparams.dataset_dir
        self.batch_size = self.hparams.batch_size

        if self.hparams.distortion == "None":
            self.train_set_transform = ImageNetBaseTrainContrastive(self.hparams)
            self.val_set_transform = ImageNetBaseTransformVal(self.hparams)
        elif self.hparams.distortion == 'multi':
            #set up the training transform and if we want a fixed mask, transfer the same mask to the validation transform
            self.train_set_transform = ImageNetDistortTrainMultiContrastive(self.hparams)
            self.val_set_transform = ImageNetDistortValMulti(self.hparams)
        else:
            #set up the training transform and if we want a fixed mask, transfer the same mask to the validation transform
            self.train_set_transform = ImageNetDistortTrainContrastive(self.hparams)

            if self.hparams.fixed_mask:
                self.val_set_transform = ImageNetDistortVal(self.hparams, fixed_distortion=self.train_set_transform.distortion)
            else:
                self.val_set_transform = ImageNetDistortVal(self.hparams)

    def setup(self, stage=None):

        if self.hparams.dataset.lower() == 'imagenet100' or self.hparams.dataset.lower() == 'imagenet-100':
            train_data = ImageNet100(
            	root=self.hparams.dataset_dir,
                split="train",
                transform=None
            )
            self.val_data = ImageNet100(
                root=self.hparams.dataset_dir,
                split="val",
                transform=self.val_set_transform
            )
        elif self.hparams.dataset.lower() == 'imagenet':
            train_data = ImageNet(
            	root=self.hparams.dataset_dir,
                split="train",
                transform=None
            )
            self.val_data = ImageNet(
                root=self.hparams.dataset_dir,
                split="val",
                transform=self.val_set_transform
            )
        else:
            raise NotImplementedError('Dataset chosen not implemented.')

        # Get the subset, as well as its labels as text.
        self.text_labels = []
        for i in range(100):
            self.text_labels.append(train_data.idx_to_class[i])
        # text_labels = list(train_data.idx_to_class.values())

        self.train_contrastive = ContrastiveUnsupervisedDataset(train_data, transform_contrastive=self.train_set_transform, return_label=True)


    def train_dataloader(self):
        return DataLoader(self.train_contrastive, batch_size=self.batch_size, num_workers=self.hparams.workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=2*self.batch_size, num_workers=self.hparams.workers, pin_memory=True, shuffle=False) # Only used for evaluation.

    def test_dataloader(self):
        return self.val_dataloader() # Same data to be used for testing, for our purposes.


class NoisyCLIP(LightningModule):
    def __init__(self, args, text_labels):
        """
        This class trains a student to produce logit sketches which approximate those provided by a teacher model.
        These label skethes are then used to retrieve the predicted labels for the input images.
        """
        super(NoisyCLIP, self).__init__()
        self.hparams = args
        self.world_size = self.hparams.num_nodes * self.hparams.gpus

        #(1) Load the correct dataset class names
        if self.hparams.dataset.lower() == "imagenet100" or self.hparams.dataset.lower() == "imagenet-100" or self.hparams.dataset.lower() == 'imagenet':
            self.text_list = ['A photo of a '+label.strip().replace('_',' ') for label in text_labels]
        else:
            raise NotImplementedError('Handling of the dataset not implemented yet.')

        if self.hparams.baseclip_type.startswith('RN'):
            embed_size = 512
        else:
            raise NotImplementedError('Unknown embedding size.')


        #(2) set up the teacher CLIP network - freeze it and don't use gradients!
        self.baseclip = clip.load(self.hparams.baseclip_type, self.hparams.device, jit=False)[0]
        self.baseclip.eval()
        self.baseclip.requires_grad_(False)
        if not self.hparams.sketch_size == "None":
            self.random_on_clean = torch.randn(self.hparams.num_classes, self.hparams.sketch_size)

        self.text_features = self.baseclip.encode_text(clip.tokenize(self.text_list))
        self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        self.text_features = self.text_features.T
        self.text_features.requires_grad_(False)

        #(3) set up the student CLIP network - unfreeze it and use gradients!
        self.noisy_visual_encoder = clip.load(self.hparams.baseclip_type, self.hparams.device, jit=False)[0].visual
        self.noisy_visual_encoder.train()

        if self.hparams.sketch_size == "None":
            self.extra_layer = torch.nn.Linear(embed_size, self.hparams.num_classes, bias=False)
            with torch.no_grad():
                self.extra_layer.weight.copy_(self.text_features.T)
        else:
            self.extra_layer = torch.nn.Linear(embed_size, self.hparams.sketch_size, bias=False)
            with torch.no_grad():
                self.extra_layer.weight.copy_((self.text_features @ self.random_on_clean).T)

        #(4) set up the training and validation accuracy metrics.
        self.train_top_1 = Accuracy(top_k=1)
        self.train_top_5 = Accuracy(top_k=5)
        self.val_top_1 = Accuracy(top_k=1)
        self.val_top_5 = Accuracy(top_k=5)

    def criterion(self, input1, input2):
        """
        Args:
            input1: Logit sketches of the clean images from the teacher. Size [N, sketch_size].
            input2: Logit sketches of the noisy images from the student. Size [N, sketch_size].
        """

        # MSE loss between Logit sketches.
        if self.hparams.loss_type == 'mse':
            return F.mse_loss(input2, input1)

        # L1 loss between Logit sketches.
        elif self.hparams.loss_type == 'l1':
            return F.l1_loss(input2, input1)

        # Contrastive losses between logit sketches.
        elif self.hparams.loss_type.startswith('simclr_'):
            assert self.hparams.loss_type in ['simclr_ss', 'simclr_st', 'simclr_both']
            # Various schemes for the negative examples
            teacher_embeds = input1 / input1.norm(dim=-1, keepdim=True)
            student_embeds = input2 / input2.norm(dim=-1, keepdim=True)
            # First compute positive examples by taking <S(x_i), T(x_i)>/T for all i
            pos_term = (teacher_embeds * student_embeds).sum(dim=1) / self.hparams.loss_tau
            # Then generate the negative term by constructing various similarity matrices
            if self.hparams.loss_type == 'simclr_ss':
                cov = torch.mm(student_embeds, student_embeds.t())
                sim = torch.exp(cov / self.hparams.loss_tau) # shape is [bsz, bsz]
                neg_term = torch.log(sim.sum(dim=-1) - sim.diag())
            elif self.hparams.loss_type == 'simclr_st':
                cov = torch.mm(student_embeds, teacher_embeds.t())
                sim = torch.exp(cov / self.hparams.loss_tau) # shape is [bsz, bsz]
                neg_term = torch.log(sim.sum(dim=-1)) # Not removing the diagonal here!
            else:
                cat_embeds = torch.cat([student_embeds, teacher_embeds])
                cov = torch.mm(student_embeds, cat_embeds.t())
                sim = torch.exp(cov / self.hparams.loss_tau) # shape is [bsz, 2 * bsz]
                # and take row-wise sums w/o diagonals and
                neg_term = torch.log(sim.sum(dim=-1) - sim.diag())
            # Final loss is
            loss = -1 * (pos_term - neg_term).mean() # (summed and mean-reduced)
            return loss

        else:
            raise ValueError('Loss function not understood.')

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.noisy_visual_encoder.parameters(), lr=self.hparams.lr)
        num_steps = 126689 // (self.hparams.batch_size * self.hparams.gpus) #divide N_train by number of distributed iters
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=num_steps)
        return [optim], [sched]

    def encode_noisy_image(self, image):
        """
        Return S(yi) where S() is the student network and yi is distorted images.
        The result is an approximation of the sketch of the logits of the output.
        """
        y = self.noisy_visual_encoder(image.type(torch.float16))
        y = y / y.norm(dim=-1, keepdim=True)
        return self.extra_layer(y)

    def forward(self, images):
        """
        Provide logits for input images. This function is used for validation and evaluation of model.

        Args:
            images: the noisy input images to be classified.
        """

        # 1) Retrieve the logit sketches for the images
        label_sketch = self.encode_noisy_image(images)

        # If no sketch size is provided, then the student just outputs the logits.
        if self.hparams.sketch_size == 'None':
            out = F.softmax(label_sketch, dim=-1)

        elif self.hparams.reconstruction == 'lasso':
            # 2) Solve a lasso reconstruction to retrieve the actual logits.
            image_probs = torch.zeros(label_sketch.shape[0], self.random_on_clean.shape[0])
            for i in range(label_sketch.shape[0]):
                image_probs[i,:] = torch.FloatTensor(solve_lasso_on_simplex(self.random_on_clean.T.detach().cpu().numpy(), label_sketch[i,:].detach().cpu().numpy()))

            # 3) apply softmax to force summation to 1.
            out = F.softmax(image_probs, dim=-1).to(label_sketch.device)

        elif self.hparams.reconstruction == 'adjoint':
            # Adjoint method to retrieve logits. Results equivalent to one step of OMP for support recovery.
            out = F.softmax(torch.matmul(label_sketch, self.random_on_clean.T.to(label_sketch.device)), dim=-1) # Note that this is not accurate beyond top-1!

        return out


    # Training methods - here we train the student to approximate the logit sketches, as provided by the teacher.
    def training_step(self, train_batch, batch_idx):
        """
        Takes a batch of clean and noisy images and returns their respective logit sketches.

        Returns:
            sketch_clean: A softmax(G T(xi)) where T() is the teacher, xi are clean images, A is iid gaussian and G is the text embedding matrix. Shape [N, sketch_size]
            sketch_noisy: S(yi) where S() is the student and yi are noisy images. Shape [N, sketch_size]
        """
        image_clean, image_noisy, labels = train_batch
        with torch.no_grad():
            if self.hparams.training_labels == 'clip':
                # If using the logits provided by the teacher, calculate them and sketch them using the random projection matrix.
                self.baseclip.eval()
                sketch_clean = self.baseclip.encode_image(image_clean.type(torch.float16))
                sketch_clean = sketch_clean / sketch_clean.norm(dim=-1, keepdim=True)
                sketch_clean = self.hparams.sharpening * torch.matmul(sketch_clean, self.text_features.to(image_clean.device))
                sketch_clean = F.softmax(sketch_clean, dim=-1)
                if not self.hparams.sketch_size == 'None':
                    sketch_clean = torch.matmul(sketch_clean, self.random_on_clean.to(image_clean.device))
            elif self.hparams.training_labels == 'truth':
                # If using the ground truth labels, treat them as one-hot encoded logits and then use the random projection matrix.
                sketch_clean = torch.matmul(F.one_hot(labels, num_classes=self.hparams.num_classes).float(), self.random_on_clean.to(image_clean.device))

        sketch_noisy = self.encode_noisy_image(image_noisy)

        return {'sketch_clean': sketch_clean, 'sketch_noisy': sketch_noisy}

    def training_step_end(self, outputs):
        """
        Given all the clean and noisy image sketches form across GPUs from training_step, gather them onto a single GPU and calculate overall loss.
        """
        sketch_clean_full = outputs['sketch_clean']
        sketch_noisy_full = outputs['sketch_noisy']
        loss = self.criterion(sketch_clean_full, sketch_noisy_full)
        self.log('train_loss', loss, prog_bar=False, logger=True, sync_dist=True, on_step=True, on_epoch=True)
        return loss

    # Validation methods - here we retrieve the predicted labels from the sketches and evaluate.
    def validation_step(self, test_batch, batch_idx):
       """
       Grab the noisy image embeddings: S(yi), where S() is the student and yi = Distort(xi). Done on each GPU.
       Return these to be evaluated in validation step end.
       """

       images_noisy, labels = test_batch
       logits = self.forward(images_noisy)
       return {'logits': logits, 'labels': labels}

    def validation_step_end(self, outputs):
       """
       Gather the noisy image features and their labels from each GPU.
       Then calculate their similarities, convert to probabilities, and calculate accuracy on each GPU.
       """
       logits_full = outputs['logits']
       labels_full = outputs['labels']

       self.log('val_top_1_step', self.val_top_1(logits_full, labels_full), prog_bar=False, logger=False)
       self.log('val_top_5_step', self.val_top_5(logits_full, labels_full), prog_bar=False, logger=False)

    def validation_epoch_end(self, outputs):
       """
       Gather the zero-shot validation accuracies from across GPUs and reduce.
       """
       self.log('val_top_1', self.val_top_1.compute(), prog_bar=True, logger=True)
       self.log('val_top_5', self.val_top_5.compute(), prog_bar=True, logger=True)
       self.val_top_1.reset()
       self.val_top_5.reset()


def run_noisy_clip():
    args = grab_config()

    seed_everything(args.seed)

    dataset = ImageNetCLIPDataset(args)
    dataset.setup()
    model = NoisyCLIP(args, dataset.text_labels)

    logger = TensorBoardLogger(
        save_dir=args.logdir,
        version=args.experiment_name,
        name='NoisyCLIP_Logs'
    )
    if args.dataset.lower() == 'imagenet100' or args.dataset.lower() == 'imagenet-100':
        trainer = Trainer.from_argparse_args(args, logger=logger)
    elif args.dataset.lower() == 'imagenet':
        # In case of ImageNet, use fewer data per epoch (for speed)
        trainer = Trainer.from_argparse_args(args, logger=logger, limit_train_batches=0.1, reload_dataloaders_every_epoch=True)
    trainer.fit(model, datamodule=dataset)


def grab_config():
    parser = argparse.ArgumentParser(description="NoisyCLIP")

    parser.add_argument('--config_file')

    config = yaml_config_hook(parser.parse_args().config_file)
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    run_noisy_clip()
