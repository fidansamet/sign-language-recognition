from src.dataset.download import download_dataset
from dataset.compute_flow import save_optical_flow

if __name__ == '__main__':
    download_dataset()
    save_optical_flow()
