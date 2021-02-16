import os
from .esri_ascii import read_esri_ascii

def _convert_to_asc(fname):
    '''
    The recipe here is to convert geotif to asc, so it can be directly loaded in read_esri_ascii
    '''
    os.system('gdal_translate -of AAIGrid -ot Float32 %s %s'%(fname, 'temp.asc'))


def read_geotif(path, fields, names=None, clobber=False):
    _convert_to_asc(path)
    grid, data= read_esri_ascii('temp.asc', fields, names, clobber)

    return (grid, data)
