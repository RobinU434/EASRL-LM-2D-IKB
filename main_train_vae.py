from utils.parser import setup_base_parser
from process.vae_process import VAEProcess


if __name__ == "__main__":
    process = VAEProcess(setup_base_parser)
    process.start()

