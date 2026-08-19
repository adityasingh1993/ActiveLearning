"""Cross-volume memory-queue SSL trainer for the controlled Final62 experiment.

The earlier spatial-token InfoNCE objective was solved almost immediately because two
intensity-perturbed views of the same volume preserved token positions. This trainer uses a
single global bottleneck embedding per volume and a FIFO memory queue of embeddings from
other volumes. The positive pair is the two independently intensity-perturbed views of the
same volume; negatives come from previous volumes in the queue.

Same-case queue entries are masked so a volume revisited in a later epoch never becomes its
own negative.
"""

from __future__ import annotations

import os
from typing import List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .ssl_pretrainer import SSLPretrainer


class QueueSSLPretrainer(SSLPretrainer):
    """Multitask SSL with cross-volume queue-based contrastive learning.

    Controlled Final62 defaults:
      inpainting = 1.0
      rotation = 0.1
      contrastive = 1.0
      queue size = 256 embeddings
    """

    def __init__(self, config, dataloader, tracker):
        super().__init__(config=config, dataloader=dataloader, tracker=tracker)

        self.lambda_inp = float(getattr(config, "ssl_lambda_inpainting", 1.0))
        self.lambda_rot = float(getattr(config, "ssl_lambda_rotation", 0.1))
        self.lambda_cont = float(getattr(config, "ssl_lambda_contrastive", 1.0))
        if min(self.lambda_inp, self.lambda_rot, self.lambda_cont) < 0:
            raise ValueError("SSL task weights must be non-negative")
        if self.lambda_inp + self.lambda_rot + self.lambda_cont <= 0:
            raise ValueError("At least one SSL task weight must be positive")

        self.queue_size = int(getattr(config, "ssl_contrastive_queue_size", 256))
        if self.queue_size < 1:
            raise ValueError("ssl_contrastive_queue_size must be >= 1")

        self.queue_embeddings = torch.empty(
            (0, int(getattr(config, "ssl_embedding_dim", 128))),
            dtype=torch.float32,
            device=self.device,
        )
        self.queue_ids: List[str] = []

    @staticmethod
    def _normalize_case_ids(case_ids, batch_size: int) -> List[str]:
        if case_ids is None:
            raise RuntimeError(
                "Queue SSL requires batch['id'] so same-case queue negatives can be masked"
            )
        if isinstance(case_ids, str):
            values = [case_ids]
        elif isinstance(case_ids, (list, tuple)):
            values = [str(x) for x in case_ids]
        else:
            try:
                values = [str(x) for x in list(case_ids)]
            except TypeError as exc:
                raise RuntimeError(f"Unsupported batch id type: {type(case_ids)!r}") from exc
        if len(values) != batch_size:
            raise RuntimeError(
                f"Expected {batch_size} case IDs for Queue SSL, got {len(values)}"
            )
        return values

    def _global_embedding(self, bottleneck: torch.Tensor) -> torch.Tensor:
        pooled = F.adaptive_avg_pool3d(bottleneck, 1).flatten(1)
        projected = self.proj_head(pooled)
        return F.normalize(projected.float(), dim=1)

    def _queue_logits(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        case_ids: Sequence[str],
        temperature: float,
    ) -> Tuple[torch.Tensor, float, float]:
        """InfoNCE logits with positive in column 0 and queue entries as negatives.

        Returns loss, mean usable negative count, and positive-pair cosine similarity.
        The first queue-less batch returns a differentiable zero contrastive loss; after that
        the queue fills rapidly because the controlled experiment has 103 shuffled volumes.
        """
        if anchors.shape != positives.shape:
            raise RuntimeError(
                f"Contrastive embedding shape mismatch: {anchors.shape} vs {positives.shape}"
            )
        if anchors.ndim != 2:
            raise RuntimeError(f"Expected [B,D] embeddings, got {anchors.shape}")

        positive_logits = (anchors * positives).sum(dim=1, keepdim=True) / temperature
        positive_cosine = float((anchors * positives).sum(dim=1).mean().detach().item())

        if self.queue_embeddings.numel() == 0:
            return anchors.sum() * 0.0, 0.0, positive_cosine

        queue = self.queue_embeddings.to(device=anchors.device, dtype=anchors.dtype)
        negative_logits = anchors @ queue.T
        negative_logits = negative_logits / temperature

        usable_counts = []
        for row_idx, case_id in enumerate(case_ids):
            valid = torch.tensor(
                [queued_id != case_id for queued_id in self.queue_ids],
                dtype=torch.bool,
                device=anchors.device,
            )
            usable_counts.append(float(valid.sum().item()))
            negative_logits[row_idx, ~valid] = torch.finfo(negative_logits.dtype).min

        if not any(count > 0 for count in usable_counts):
            return anchors.sum() * 0.0, 0.0, positive_cosine

        logits = torch.cat([positive_logits, negative_logits], dim=1)
        targets = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, targets)
        return loss, float(sum(usable_counts) / len(usable_counts)), positive_cosine

    def _contrastive_loss(
        self,
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        case_ids: Sequence[str],
    ) -> Tuple[torch.Tensor, float, float]:
        temperature = float(getattr(self.config, "ssl_contrastive_temp", 0.07))
        if temperature <= 0:
            raise ValueError("ssl_contrastive_temp must be > 0")

        loss_12, neg_12, pos_12 = self._queue_logits(
            emb1, emb2, case_ids, temperature
        )
        loss_21, neg_21, pos_21 = self._queue_logits(
            emb2, emb1, case_ids, temperature
        )
        return (
            0.5 * (loss_12 + loss_21),
            0.5 * (neg_12 + neg_21),
            0.5 * (pos_12 + pos_21),
        )

    @torch.no_grad()
    def _enqueue(self, embeddings: torch.Tensor, case_ids: Sequence[str]) -> None:
        embeddings = F.normalize(embeddings.detach().float(), dim=1)
        if len(case_ids) != embeddings.size(0):
            raise RuntimeError("Queue enqueue case-ID count mismatch")

        self.queue_embeddings = torch.cat(
            [self.queue_embeddings, embeddings.to(self.device)], dim=0
        )
        self.queue_ids.extend(str(x) for x in case_ids)

        if self.queue_embeddings.size(0) > self.queue_size:
            excess = self.queue_embeddings.size(0) - self.queue_size
            self.queue_embeddings = self.queue_embeddings[excess:]
            self.queue_ids = self.queue_ids[excess:]

    def train(self, num_epochs: int):
        log_img_freq = getattr(self.config, "log_image_every_n_epochs", 10)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_inp = 0.0
            epoch_rot = 0.0
            epoch_cont = 0.0
            epoch_negatives = 0.0
            epoch_positive_cosine = 0.0
            n_contrastive_steps = 0
            first_batch_sample = None

            should_log_image = (
                ((epoch + 1) % log_img_freq == 0)
                or (epoch == num_epochs - 1)
                or (epoch == 0)
            )

            for batch_idx, batch in enumerate(self.dataloader):
                x = batch["image"].to(self.device)
                batch_size = x.size(0)
                case_ids = self._normalize_case_ids(batch.get("id"), batch_size)
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
                            self.mse_loss(out_inp * (1 - mask), x * (1 - mask))
                            * mask.numel()
                        ) / masked_count
                    else:
                        loss_inp = self.mse_loss(out_inp, x)

                    # 2) Rotation + 3) cross-volume contrastive embeddings.
                    x_rot, rot_labels = self._apply_rotation(x)
                    x_aug1 = self._make_contrastive_view(x)
                    x_aug2 = self._make_contrastive_view(x)

                    x_combined = torch.cat([x_rot, x_aug1, x_aug2], dim=0)
                    bottlenecks = self._extract_bottleneck_features(x_combined)

                    b_rot = bottlenecks[:batch_size]
                    b1 = bottlenecks[batch_size : 2 * batch_size]
                    b2 = bottlenecks[2 * batch_size :]

                    rot_feats = F.adaptive_avg_pool3d(b_rot, 1).flatten(1)
                    rot_preds = self.rot_head(rot_feats)
                    loss_rot = self.ce_loss(rot_preds, rot_labels)

                    emb1 = self._global_embedding(b1)
                    emb2 = self._global_embedding(b2)
                    loss_cont, mean_negatives, positive_cosine = self._contrastive_loss(
                        emb1, emb2, case_ids
                    )

                    weighted_inp = self.lambda_inp * loss_inp
                    weighted_rot = self.lambda_rot * loss_rot
                    weighted_cont = self.lambda_cont * loss_cont
                    loss = weighted_inp + weighted_rot + weighted_cont

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Queue receives one detached representation per current volume after the
                # optimization step, so current positives never appear among their negatives.
                self._enqueue(0.5 * (emb1.detach() + emb2.detach()), case_ids)

                epoch_loss += float(loss.item())
                epoch_inp += float(loss_inp.item())
                epoch_rot += float(loss_rot.item())
                epoch_cont += float(loss_cont.item())
                epoch_negatives += float(mean_negatives)
                epoch_positive_cosine += float(positive_cosine)
                n_contrastive_steps += 1

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
            avg_negatives = epoch_negatives / max(1, n_contrastive_steps)
            avg_positive_cosine = epoch_positive_cosine / max(1, n_contrastive_steps)

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
                f"Queue: {len(self.queue_ids)}/{self.queue_size} Neg={avg_negatives:.1f} "
                f"PosCos={avg_positive_cosine:.3f} | LR: {current_lr:.6f}"
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
                    "ssl_queue_depth": float(len(self.queue_ids)),
                    "ssl_queue_capacity": float(self.queue_size),
                    "ssl_mean_usable_negatives": avg_negatives,
                    "ssl_positive_pair_cosine": avg_positive_cosine,
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
                    "  [SSL Early Stopping] Queue SSL loss did not improve for "
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
