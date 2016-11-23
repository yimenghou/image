# specify dataset path in .yaml file before running this scrpits

from getImgLite import getImg
import yaml

config_file = "harbour_config.yaml"
with open(config_file, 'r') as fhandle:
    config = yaml.load(fhandle)            

numOBJ = getImg(config['dataset'])

dataset = numOBJ.dataset
labelset = numOBJ.labelset

print dataset.shape, labelset.shape