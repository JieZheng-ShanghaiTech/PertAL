from __future__ import annotations

import sys


def _require_hydra() -> None:
    try:
        import hydra  # noqa: F401
        from omegaconf import DictConfig  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "This entrypoint requires hydra-core. Install it first, e.g.\n"
            "  pip install hydra-core\n"
            f"Original import error: {e}"
        )


_require_hydra()

import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    from pertal.pertal import PertAL

    device = f"cuda:{cfg.device}" if isinstance(cfg.device, int) else str(cfg.device)

    interface = PertAL(
        weight_bias_track=bool(cfg.tracking.enable),
        exp_name=f"{cfg.strategy.name}_{cfg.llm.name}_{cfg.dataset.name}_seed{cfg.seed}",
        device=device,
        seed=int(cfg.seed),
        llm_weight=float(cfg.strategy.llm_weight),
        gradient_weight=float(cfg.strategy.gradient_weight),
    )

    interface.initialize_data(
        path=str(cfg.dir.data),
        dataset_name=str(cfg.dataset.name),
        batch_size=int(cfg.dataset.batch_size),
        test_fraction=float(cfg.dataset.test_fraction),
        llm_name=str(cfg.llm.name),
    )

    interface.initialize_model(
        epochs=int(cfg.model.epochs),
        hidden_size=int(cfg.model.hidden_size),
    )

    interface.initialize_active_learning_strategy(
        strategy=str(cfg.strategy.name),
        prior_scfm_kernel=str(cfg.dataset.prior_scfm_kernel),
    )

    interface.start(
        n_init_labeled=int(cfg.al.n_init_labeled),
        n_round=int(cfg.al.n_rounds),
        n_query=int(cfg.al.n_query),
        save_path=str(cfg.dir.results),
    )


if __name__ == "__main__":
    sys.exit(main())

