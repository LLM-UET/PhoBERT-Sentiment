import click

@click.group()
def cli():
    pass

@cli.command()
def segment():
    from .SEGMENT import do_SEGMENT
    do_SEGMENT()

@cli.group()
def download():
    pass

@download.group(name="models")
def download_models():
    pass

@download_models.command(name="BASE")
def download_models_BASE():
    from .DOWNLOAD_MODELS import do_DOWNLOAD_MODELS_BASE
    do_DOWNLOAD_MODELS_BASE()

@download_models.command(name="FINETUNED")
def download_models_FINETUNED():
    from .DOWNLOAD_MODELS import do_DOWNLOAD_MODELS_FINETUNED
    do_DOWNLOAD_MODELS_FINETUNED()

@cli.command()
def split():
    from .SPLIT import do_SPLIT
    do_SPLIT()

@cli.command()
def finetune():
    from .FINETUNE import do_FINETUNE
    do_FINETUNE()

if __name__ == "__main__":
    cli()
