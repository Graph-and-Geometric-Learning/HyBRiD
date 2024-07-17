import os

from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.loggers import Logger


class SaveConfigCallback(SaveConfigCallback):
    def setup(self, trainer, pl_module, stage):
        if self.already_saved:
            return

        if self.save_to_log_dir:
            assert trainer.log_dir is not None
            log_dir = os.path.join(
                trainer.log_dir, trainer.logger.name, trainer.logger.version
            )  # this broadcasts the directory
            if trainer.is_global_zero and not os.path.exists(log_dir):
                os.makedirs(log_dir)
            config_path = os.path.join(log_dir, self.config_filename)
            fs = get_filesystem(log_dir)

            if not self.overwrite:
                # check if the file exists on rank 0
                file_exists = (
                    fs.isfile(config_path) if trainer.is_global_zero else False
                )
                # broadcast whether to fail to all ranks
                file_exists = trainer.strategy.broadcast(file_exists)
                if file_exists:
                    raise RuntimeError(
                        f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                        " results of a previous run. You can delete the previous config file,"
                        " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                        ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                    )

            if trainer.is_global_zero:
                fs.makedirs(log_dir, exist_ok=True)
                self.parser.save(
                    self.config,
                    config_path,
                    skip_none=False,
                    overwrite=self.overwrite,
                    multifile=self.multifile,
                )

        if trainer.is_global_zero:
            self.save_config(trainer, pl_module, stage)
            self.already_saved = True

        # broadcast so that all ranks are in sync on future calls to .setup()
        self.already_saved = trainer.strategy.broadcast(self.already_saved)

    def save_config(self, trainer, pl_module, stage: str) -> None:
        if isinstance(trainer.logger, Logger):
            config = self.parser.dump(
                self.config, skip_none=False
            )  # Required for proper reproducibility
            trainer.logger.log_hyperparams({"config": config})


class Main(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments(
            "data.entropy", "model.init_args.node_entropy", apply_on="instantiate"
        )


def main():
    cli = Main(save_config_callback=SaveConfigCallback, run=False)
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)

    cli.trainer.test(cli.model, datamodule=cli.datamodule)


if __name__ == "__main__":
    main()
