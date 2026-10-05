from __future__ import annotations

import datetime
import logging
from pathlib import Path

import hydra
import pandas as pd
import rioxarray
import torch
from omegaconf import DictConfig, OmegaConf

from trainer.datamodules.flood_datamodule import FloodDataModule
from trainer.evaluation.validate import _pick_checkpoint, forward_full_image_tiled, load_model_from_checkpoint
from trainer.utils.normalization_methods import denormalize_min_max
from trainer.utils.utils import rewrite_cfg, save_array_as_gtiff

log = logging.getLogger(__name__)
# Quiet only the credential chatter
logging.getLogger("botocore.credentials").setLevel(logging.WARNING)
# also quiet broader AWS libs
for name in ("botocore", "boto3", "s3transfer", "s3fs", "fsspec"):
    logging.getLogger(name).setLevel(logging.WARNING)


@hydra.main(
    version_base="1.3",
    config_path="../config",
    config_name="eval_config",
)
def main(cfg: DictConfig) -> None:
    """
    forward-runs on requested dataset based on what mentioned in eval_config

    :param cfg: configuration file for evaluation (different from configuration used for training the model
    :return: None, saves the outputs into tif files
    """
    run_dir = Path(cfg.eval.run_dir or "")
    if not run_dir or not (run_dir / ".hydra" / "config.yaml").exists():
        raise FileNotFoundError(
            "Please set eval.run_dir to a Hydra training run directory (with .hydra/config.yaml)."
        )

    # 1) Load the **original composed config** from that training run
    train_cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    if cfg["forecast_data_source"]:
        train_cfg = rewrite_cfg(cfg=train_cfg, key="data_sources", items=cfg["data_sources"])
        # adding forecast_data_source to train_cfg. we need this when FloodDataModule wants to find
        # appropriate class to read inputs. Depending on the input type, the reading function is different
        train_cfg["forecast_data_source"] = True
        train_cfg["forecast_end_time"] = cfg["inference"]["end_time"]

    # Rebuild dataset with training cfg
    dm = FloodDataModule(train_cfg)
    dm.setup(mode=cfg["eval_mode"])
    # read eval.full_image_eval from cfg; default False
    dm.eval_ds.full_image_eval = bool(OmegaConf.select(cfg, "eval.full_image_eval", default=False))

    # Build eval loader with **eval overrides**
    g = torch.Generator().manual_seed(int(train_cfg.seed))

    eval_loader = dm.eval_dataloader(
        batch_size=int(cfg.eval.batch_size),
        num_workers=int(cfg.eval.num_workers),
        shuffle=False,
        generator=g,
    )
    if cfg["eval_mode"] == "eval":
        aoi_bbox_crs = None
    elif cfg["eval_mode"] == "train":
        eval_loader.dataset.flood_instances = dm.train_ds.flood_instances
        aoi_bbox_crs = None
    elif cfg["eval_mode"] == "user_defined":
        eval_loader.dataset.flood_instances = pd.DataFrame.from_dict(
            {
                "flood_id": 0,
                "tif_path": "_",
                "start_time": "_",
                "end_time": cfg["inference"]["end_time"],
                "left": cfg["inference"]["bbox"]["left"],
                "bottom": cfg["inference"]["bbox"]["bottom"],
                "right": cfg["inference"]["bbox"]["right"],
                "top": cfg["inference"]["bbox"]["top"],
            },
            orient="index",
        ).T
        aoi_bbox_crs = cfg["inference"]["crs"]

    # 2) Choose checkpoint
    ckpt_path = _pick_checkpoint(run_dir, cfg.eval.ckpt)
    log.info(f"Using checkpoint: {ckpt_path}")

    # Load model and evaluate
    device = str(train_cfg.device)
    model = load_model_from_checkpoint(train_cfg, ckpt_path, device=device)
    # Run eval
    for idx in range(len(eval_loader.dataset.flood_instances)):
        pred, out_transform, out_crs = forward_full_image_tiled(
            model=model,
            ds=eval_loader.dataset,
            aoi_bbox=(
                float(eval_loader.dataset.flood_instances.iloc[idx]["left"]),
                float(eval_loader.dataset.flood_instances.iloc[idx]["bottom"]),
                float(eval_loader.dataset.flood_instances.iloc[idx]["right"]),
                float(eval_loader.dataset.flood_instances.iloc[idx]["top"]),
            ),  # in master CRS. it can be any bound that the user wants
            aoi_bbox_crs=aoi_bbox_crs,
            end_time=eval_loader.dataset.flood_instances.iloc[idx][
                "end_time"
            ],  # or any time that the user wants
            tile_h=int(cfg.eval.No_pixels_y),  # or cfg.train.No_pixels_y
            tile_w=int(cfg.eval.No_pixels_x),  # or cfg.train.No_pixels_x
            device=str(cfg.device),
            use_amp=True,  # if CUDA, saves memory
            master_path=eval_loader.dataset.master_path,  # master grid path
        )
        # 1) denormalize to target units
        tstats = eval_loader.dataset.target_stats.get("band_1", None)
        if tstats is not None:
            pred_to_save = denormalize_min_max(pred, tstats)
        else:
            # fallback: save normalized if stats missing
            pred_to_save = pred

        # decide CRS fallback (if reference lacks CRS)
        master_grid = rioxarray.open_rasterio(eval_loader.dataset.master_path)
        master_crs = master_grid.rio.crs
        if Path(eval_loader.dataset.flood_instances.iloc[idx]["tif_path"]).exists():
            # UTC timestamp
            ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
            ref_path = eval_loader.dataset.flood_instances.iloc[idx]["tif_path"]
            ref_path = str(Path(ref_path).name)  # has .tif
            file_name = f"pred_{ts}_{ref_path}"
        else:
            # UTC timestamp
            ts = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%SZ")
            f_id = str(eval_loader.dataset.flood_instances.iloc[idx]["flood_id"])
            file_name = f"pred_{ts}_{f_id}.tif"
        out_path = Path(cfg.eval.run_dir) / "preds" / file_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # saving the prediction into Tiff file with appropriate crs
        save_array_as_gtiff(
            arr_hw=pred_to_save, out_path=out_path, transform=out_transform, crs=master_crs, nodata=1e20
        )

        print(f"[EVAL] wrote in {out_path}")


if __name__ == "__main__":
    main()
