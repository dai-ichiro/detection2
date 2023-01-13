import os
from mim.commands.download import download
os.makedirs('models', exist_ok=True)
checkpoint_name = 'siamese_rpn_r50_20e_lasot'
checkpoint = download(package="mmtrack", configs=[checkpoint_name], dest_root="models")[0]
