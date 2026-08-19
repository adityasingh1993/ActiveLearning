"""Weighted multitask SSL trainer for the controlled Final62 experiment.

This subclasses the existing SSLPretrainer so all data flow, masking, rotation, stronger
4x4x4 spatial-token contrastive views, optimizer, scheduler, and checkpoint behavior remain
unchanged. Only the relative task weights and contribution logging differ.
"""

import os

import torch

from .ssl_pretrainer import SSLPretrainer


class WeightedSSLPretrainer(SSLPretrainer):
    """SSL pretrainer with configurable task weights.

    Controlled Final62 defaults:
      inpainting = 1.0
      rotation = 0.25
      contrastive = 5.0
    """

    def __init__(self, config, dataloader, tracker):
        super().__init__(config=config, dataloader=dataloader, tracker=tracker)
        self.lambda_inp = float(getattr(config, "ssl_lambda_inpainting", 1.0))
        self.lambda_rot = float(getattr(config, "ssl_lambda_rotation", 0.25))
        self.lambda_cont = float(getattr(config, "ssl_lambda_contrastive", 5.0))
        if min(self.lambda_inp, self.lambda_rot, self.lambda_cont) < 0:
            raise ValueError("SSL task weights must be non-negative")
        if self.lambda_inp + self.lambda_rot + self.lambda_cont <= 0:
            raise ValueError("At least one SSL task weight must be positive")

    def train(self, num_epochs: int):
        log_img_freq = getattr(self.config, "log_image_every_n_epochs", 10)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_inp = 0.0
            epoch_rot = 0.0
            epoch_cont = 0.0
            first_batch_sample = None

            should_log_image = (
                ((epoch + 1) % log_img_freq == 0)
                or (epoch == num_epochs - 1)
                or (epoch == 0)
            )

            for batch_idx, batch in enumerate(self.dataloader):
                x = batch["image"].to(self.device)
                self.optimizer.zero_grad()

                with torch.amp.autocast(
                    self.device_type, enabled=(self.device_type == "cuda")
                ):
                    # 1) Masked-volume inpainting.
                    x_masked, mask = self._apply_masking(x)
                    out_inp = self.model(x_masked)
                    if isinstance(out_inp, (list, tuple)):
                        out_inp = out_inp[0]
                    elif out_inp.ndim == 6:
                        out_inp = out_inp[:, 0]

                    masked_count = (1 - mask).sum()
                    if masked_count > 0:
                        loss_inp = (
                            self.mse_loss(
                                out_inp * (1 - mask), x * (1 - mask)
                            )
                            * mask.numel()
                        ) / masked_count
                    else:
                        loss_inp = self.mse_loss(out_inp, x)

                    # 2) Rotation + 3) stronger spatial-token contrastive task.
                    x_rot, rot_labels = self._apply_rotation(x)
                    x_aug1 = self._make_contrastive_view(x)
                    x_aug2 = self._make_contrastive_view(x)

                    batch_size = x.size(0)
                    x_combined = torch.cat([x_rot, x_aug1, x_aug2], dim=0)
                    bottlenecks = self._extract_bottleneck_features(x_combined)

                    b_rot = bottlenecks[:batch_size]
                    b1 = bottlenecks[batch_size : 2 * batch_size]
                    b2 = bottlenecks[2 * batch_size :]

                    rot_feats = torch.nn.functional.adaptive_avg_pool3d(
                        b_rot, 1
                    ).view(batch_size, -1)
                    rot_preds = self.rot_head(rot_feats)
                    loss_rot = self.ce_loss(rot_preds, rot_labels)

                    g = self.contrastive_grid_size
                    p1 = (
                        torch.nn.functional.adaptive_avg_pool3d(b1, (g, g, g))
                        .permute(0, 2, 3, 4, 1)
                        .reshape(-1, b1.size(1))
                    )
                    p2 = (
                        torch.nn.functional.adaptive_avg_pool3d(b2, (g, g, g))
                        .permute(0, 2, 3, 4, 1)
                        .reshape(-1, b2.size(1))
                    )
                    feat1 = self.proj_head(p1)
                    feat2 = self.proj_head(p2)
                    loss_cont = self._infonce_loss(
                        feat1,
                        feat2,
                        temperature=getattr(
                            self.config, "ssl_contrastive_temp", 0.07
                        ),
                    )

                    weighted_inp = self.lambda_inp * loss_inp
                    weighted_rot = self.lambda_rot * loss_rot
                    weighted_cont = self.lambda_cont * loss_cont
                    loss = weighted_inp + weighted_rot + weighted_cont

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                epoch_loss += float(loss.item())
                epoch_inp += float(loss_inp.item())
                epoch_rot += float(loss_rot.item())
                epoch_cont += float(loss_cont.item())

                if first_batch_sample is None and should_log_image:
                    first_batch_sample = (
                        x[0].detach().cpu(),
                        x_masked[0].detach().cpu(),
                        out_inp[0].detach().cpu(),
                    )

            n_batches = max(1, len(self.dataloader))
            avg_loss = epoch_loss / n_batches
            avg_inp = epoch_inp / n_batches
            avg_rot = epoch_rot / n_batches
            avg_cont = epoch_cont / n_batches

            w_inp = self.lambda_inp * avg_inp
            w_rot = self.lambda_rot * avg_rot
            w_cont = self.lambda_cont * avg_cont
            weighted_total = max(w_inp + w_rot + w_cont, 1e-12)
            inp_pct = 100.0 * w_inp / weighted_total
            rot_pct = 100.0 * w_rot / weighted_total
            cont_pct = 100.0 * w_cont / weighted_total

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            current_lr = self.optimizer.param_groups[0]["lr"]
            if getattr(self, "scheduler", None) is not None:
                self.scheduler.step()

            print(
                f"  [SSL Pre-train] Epoch {epoch + 1:3d}/{num_epochs} | "
                f"Loss: {avg_loss:.4f} "
                f"(Inp: {avg_inp:.4f}, Rot: {avg_rot:.4f}, Cont: {avg_cont:.4f}) | "
                f"Weighted: I={w_inp:.4f} R={w_rot:.4f} C={w_cont:.4f} | "
                f"Mix: I={inp_pct:.1f}% R={rot_pct:.1f}% C={cont_pct:.1f}% | "
                f"LR: {current_lr:.6f}"
            )

            self.tracker.log_metrics(
                {
                    "ssl_loss": avg_loss,
                    "ssl_loss_inpainting_raw": avg_inp,
                    "ssl_loss_rotation_raw": avg_rot,
                    "ssl_loss_contrastive_raw": avg_cont,
                    "ssl_loss_inpainting_weighted": w_inp,
                    "ssl_loss_rotation_weighted": w_rot,
                    "ssl_loss_contrastive_weighted": w_cont,
                    "ssl_lambda_inpainting": self.lambda_inp,
                    "ssl_lambda_rotation": self.lambda_rot,
                    "ssl_lambda_contrastive": self.lambda_cont,
                    "ssl_contribution_inpainting_pct": inp_pct,
                    "ssl_contribution_rotation_pct": rot_pct,
                    "ssl_contribution_contrastive_pct": cont_pct,
                    "ssl_contrastive_grid_size": float(self.contrastive_grid_size),
                    "ssl_contrastive_tokens_per_volume": float(
                        self.contrastive_grid_size ** 3
                    ),
                    "ssl_learning_rate": current_lr,
                },
                step=epoch,
            )

            if should_log_image and first_batch_sample is not None:
                self.log_ssl_inpainting_samples(epoch, *first_batch_sample)

            if (
                getattr(self, "early_stopper", None) is not None
                and self.early_stopper(avg_loss)
            ):
                print(
                    "  [SSL Early Stopping] Weighted SSL loss did not improve for "
                    f"{self.early_stopper.patience} consecutive epochs. "
                    f"Early stopping at epoch {epoch + 1}."
                )
                break

        ckpt_dir = getattr(
            self.config, "checkpoint_dir", "./experiments/checkpoints"
        )
        os.makedirs(ckpt_dir, exist_ok=True)
        save_path = os.path.join(ckpt_dir, "ssl_pretrained.pth")
        torch.save(self.model.state_dict(), save_path)
        self.tracker.log_artifact(save_path, name="ssl_pretrained_weights")
        print(f"  Pre-trained model saved to {save_path}")
