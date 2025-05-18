import os
from urllib import request
import zipfile

DATA_DIR = 'data'


NAME_URL_DICT_UCI = {
    'Rice': 'https://archive.ics.uci.edu/static/public/545/rice+cammeo+and+osmancik.zip',
    'adult': 'https://archive.ics.uci.edu/static/public/2/adult.zip',
    'default': 'https://archive.ics.uci.edu/static/public/350/default+of+credit+card+clients.zip',
    'magic': 'https://archive.ics.uci.edu/static/public/159/magic+gamma+telescope.zip',
    'connect4': 'https://archive.ics.uci.edu/static/public/26/connect+4.zip',
    'chess': 'https://archive.ics.uci.edu/static/public/22/chess+king+rook+vs+king+pawn.zip',
    'letter_recog': 'https://archive.ics.uci.edu/static/public/59/letter+recognition.zip',
    'nursery': 'https://archive.ics.uci.edu/static/public/76/nursery.zip',
    'room_occupancy': 'https://archive.ics.uci.edu/static/public/864/room+occupancy+estimation.zip',
    'car': 'https://archive.ics.uci.edu/static/public/19/car+evaluation.zip',
    'maternal_health': 'https://archive.ics.uci.edu/static/public/863/maternal+health+risk.zip'
}

def unzip_file(zip_filepath, dest_path):
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(dest_path)


def download_from_uci(name):

    print(f'Start processing dataset {name} from UCI.')
    save_dir = f'{DATA_DIR}/{name}'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

        url = NAME_URL_DICT_UCI[name]
        request.urlretrieve(url, f'{save_dir}/{name}.zip')
        print(f'Finish downloading dataset from {url}, data has been saved to {save_dir}.')
        
        unzip_file(f'{save_dir}/{name}.zip', save_dir)
        print(f'Finish unzipping {name}.')
    
    else:
        print('Aready downloaded.')

if __name__ == '__main__':
    for name in NAME_URL_DICT_UCI.keys():
        download_from_uci(name)
    