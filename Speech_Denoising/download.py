import os
import requests
import zipfile
from tqdm import tqdm


def create_dirs():
    dirs = [
        'datasets/train/clean',
        'datasets/train/noisy',
        'datasets/test/clean',
        'datasets/test/noisy'
    ]
    for dir in dirs:
        os.makedirs(dir, exist_ok=True)


def download_file(url, filename):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(filename, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            bar.update(size)


def process_dataset(url, extract_path):
    zip_name = url.split('/')[-1].split('?')[0]

    print(f'Downloading {zip_name}...')
    download_file(url, zip_name)

    print(f'Extracting {zip_name}...')
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    os.remove(zip_name)


def download():
    datasets = [
        {
            'url': 'https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_trainset_28spk_wav.zip?sequence=2&isAllowed=y',
            'path': 'datasets/train/clean'
        },
        {
            'url': 'https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_trainset_28spk_wav.zip?sequence=6&isAllowed=y',
            'path': 'datasets/train/noisy'
        },
        {
            'url': 'https://datashare.ed.ac.uk/bitstream/handle/10283/2791/clean_testset_wav.zip?sequence=1&isAllowed=y',
            'path': 'datasets/test/clean'
        },
        {
            'url': 'https://datashare.ed.ac.uk/bitstream/handle/10283/2791/noisy_testset_wav.zip?sequence=5&isAllowed=y',
            'path': 'datasets/test/noisy'
        }
    ]

    create_dirs()

    for dataset in datasets:
        try:
            process_dataset(dataset['url'], dataset['path'])
        except Exception as e:
            print(f'Error processing {dataset["url"]}: {str(e)}')

