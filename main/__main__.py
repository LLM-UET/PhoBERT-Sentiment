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

@cli.command()
@click.argument("text", type=str, required=False, default=None)
@click.option("--input-model-dir", type=str, required=False, default=None)
@click.option("--interactive", is_flag=True, type=bool, required=False, default=False)
def infer(text: str, input_model_dir: str, interactive):
    if not interactive:
        if text is None:
            raise RuntimeError(f"ERROR: No input text?")
        else:
            text = text.replace("\\n", '\n')

    from .INFER import do_INFER
    
    result = do_INFER(
        text=text,
        input_model_dir=input_model_dir,
        interactive=interactive,
    )

    print(result)

@cli.command()
@click.option("--host", type=str, required=False, default=None)
@click.option("--port", type=int, required=False, default=None)
@click.option("--input-model-dir", type=str, required=False, default=None)
def serve(host: str, port: int, input_model_dir: str):
    from .SERVER import run_server
    run_server(
        input_model_dir=input_model_dir,
        host=host,
        port=port,
    )

if __name__ == "__main__":
    cli()
