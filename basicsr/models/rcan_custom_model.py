import torch
import numpy as np
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class RCANCustom(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(RCANCustom, self).__init__(opt)

        # define network
        self.net_g = build_network(opt["network_g"])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt["path"].get("pretrain_network_g", None)
        if load_path is not None:
            param_key = self.opt["path"].get("param_key_g", "params")
            self.load_network(
                self.net_g,
                load_path,
                self.opt["path"].get("strict_load_g", True),
                param_key,
            )

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt["train"]

        self.ema_decay = train_opt.get("ema_decay", 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f"Use Exponential Moving Average with decay: {self.ema_decay}")
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt["network_g"]).to(self.device)
            # load pretrained model
            load_path = self.opt["path"].get("pretrain_network_g", None)
            if load_path is not None:
                self.load_network(
                    self.net_g_ema,
                    load_path,
                    self.opt["path"].get("strict_load_g", True),
                    "params_ema",
                )
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get("pixel_opt"):
            self.cri_pix = build_loss(train_opt["pixel_opt"]).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get("perceptual_opt"):
            self.cri_perceptual = build_loss(train_opt["perceptual_opt"]).to(
                self.device
            )
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError("Both pixel and perceptual losses are None.")

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt["train"]
        optim_params = []
        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f"Params {k} will not be optimized.")

        optim_type = train_opt["optim_g"].pop("type")
        self.optimizer_g = self.get_optimizer(
            optim_type, optim_params, **train_opt["optim_g"]
        )
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data):
        self.lq = data["lq"].to(self.device)
        if "gt" in data:
            self.gt = data["gt"].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict["l_pix"] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict["l_percep"] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict["l_style"] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt["rank"] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """Non-distributed validation with metrics calculation and image saving."""
        dataset_name = dataloader.dataset.opt["name"]
        with_metrics = self.opt["val"].get("metrics") is not None
        use_pbar = self.opt["val"].get("pbar", False)

        # Initialize metrics tracking
        if with_metrics:
            self._initialize_metrics_tracking(dataset_name)

        # Setup progress bar
        pbar = tqdm(total=len(dataloader), unit="image") if use_pbar else None

        try:
            # Process each validation image
            num_images = 0
            for idx, val_data in enumerate(dataloader):
                self._process_validation_image(
                    val_data, current_iter, dataset_name, with_metrics, save_img, pbar
                )
                num_images = idx + 1
        finally:
            if pbar:
                pbar.close()

        # Finalize metrics
        if with_metrics:
            self._finalize_metrics(dataset_name, current_iter, tb_logger, num_images)

    def _initialize_metrics_tracking(self, dataset_name):
        """Initialize metrics tracking for validation."""
        if not hasattr(self, "metric_results"):
            self.metric_results = {
                metric: 0 for metric in self.opt["val"]["metrics"].keys()
            }

        # Initialize best metric results for this dataset
        self._initialize_best_metric_results(dataset_name)

        # Reset current metric results
        self.metric_results = {metric: 0 for metric in self.metric_results}

    def _process_validation_image(
        self, val_data, current_iter, dataset_name, with_metrics, save_img, pbar
    ):
        """Process a single validation image."""
        img_name = osp.splitext(osp.basename(val_data["path"][0]))[0]

        # Run inference
        self.feed_data(val_data)
        self.test()

        # Get model outputs
        visuals = self.get_current_visuals()

        # Prepare data for metrics (convert tensors to numpy)
        metric_data = self._prepare_metric_data(visuals)

        # Clean up GPU memory
        self._cleanup_gpu_memory()

        # Save image if requested
        if save_img:
            self._save_validation_image(visuals, img_name, current_iter, dataset_name)

        # Calculate metrics
        if with_metrics:
            self._calculate_image_metrics(metric_data)

        # Update progress bar
        if pbar:
            pbar.update(1)
            pbar.set_description(f"Test {img_name}")

    def _prepare_metric_data(self, visuals):
        """Prepare metric data by converting tensors to numpy arrays."""
        metric_data = {}

        # Get normalization parameters from dataset
        target_mean = self.opt["datasets"]["val"]["target_mean"]
        target_std = self.opt["datasets"]["val"]["target_std"]

        # Convert result tensor to numpy and denormalize
        sr_img = [x.numpy() for x in visuals["result"]][0]
        sr_img_denorm = sr_img * target_std + target_mean
        metric_data["img"] = sr_img_denorm

        # Convert GT tensor to numpy and denormalize
        if "gt" in visuals:
            gt_img = [x.numpy() for x in visuals["gt"]][0]
            gt_img_denorm = gt_img * target_std + target_mean
            metric_data["img2"] = gt_img_denorm
            del self.gt

        return metric_data

    def _cleanup_gpu_memory(self):
        """Clean up GPU memory after processing."""
        del self.lq
        del self.output
        torch.cuda.empty_cache()

    def _save_validation_image(self, visuals, img_name, current_iter, dataset_name):
        """Save validation image to disk."""
        sr_img = [x.numpy() for x in visuals["result"]][0]

        # Get normalization parameters and denormalize
        target_mean = self.opt["datasets"]["val"]["target_mean"]
        target_std = self.opt["datasets"]["val"]["target_std"]
        sr_img_denorm = sr_img * target_std + target_mean

        # Debug: Print image shape and dtype before processing
        logger = get_root_logger()
        logger.debug(f"Original image shape: {sr_img.shape}, dtype: {sr_img.dtype}, range: [{sr_img.min():.3f}, {sr_img.max():.3f}]")
        logger.debug(f"Denormalized image range: [{sr_img_denorm.min():.3f}, {sr_img_denorm.max():.3f}]")

        # Convert to (H, W) format since we know it's always 1 channel
        if len(sr_img_denorm.shape) == 3:
            sr_img_denorm = sr_img_denorm.squeeze()

        # Debug: Print final image shape and dtype
        logger.debug(f"Final image shape: {sr_img_denorm.shape}, dtype: {sr_img_denorm.dtype}, range: [{sr_img_denorm.min()}, {sr_img_denorm.max()}]")

        if self.opt["is_train"]:
            save_img_path = osp.join(
                self.opt["path"]["visualization"],
                img_name,
                f"{img_name}_{current_iter}.tif",
            )
        else:
            if self.opt["val"]["suffix"]:
                save_img_path = osp.join(
                    self.opt["path"]["visualization"],
                    dataset_name,
                    f'{img_name}_{self.opt["val"]["suffix"]}.tif',
                )
            else:
                save_img_path = osp.join(
                    self.opt["path"]["visualization"],
                    dataset_name,
                    f'{img_name}_{self.opt["name"]}.tif',
                )

        imwrite(sr_img_denorm, save_img_path)

    def _calculate_image_metrics(self, metric_data):
        """Calculate metrics for a single image."""
        for name, opt_ in self.opt["val"]["metrics"].items():
            self.metric_results[name] += calculate_metric(metric_data, opt_)

    def _finalize_metrics(self, dataset_name, current_iter, tb_logger, num_images):
        """Finalize metrics calculation and logging."""
        # Average metrics across all images
        for metric in self.metric_results.keys():
            self.metric_results[metric] /= num_images

            # Update best metric result
            self._update_best_metric_result(
                dataset_name, metric, self.metric_results[metric], current_iter
            )

        # Log validation results
        self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f"Validation {dataset_name}\n"
        for metric, value in self.metric_results.items():
            log_str += f"\t # {metric}: {value:.4f}"
            if hasattr(self, "best_metric_results"):
                log_str += (
                    f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                    f'{self.best_metric_results[dataset_name][metric]["iter"]} iter'
                )
            log_str += "\n"

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(
                    f"metrics/{dataset_name}/{metric}", value, current_iter
                )

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict["lq"] = self.lq.detach().cpu()
        out_dict["result"] = self.output.detach().cpu()
        if hasattr(self, "gt"):
            out_dict["gt"] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, "net_g_ema"):
            self.save_network(
                [self.net_g, self.net_g_ema],
                "net_g",
                current_iter,
                param_key=["params", "params_ema"],
            )
        else:
            self.save_network(self.net_g, "net_g", current_iter)
        self.save_training_state(epoch, current_iter)
