import pandas as pd
import numpy as np
import re

def _read_project_name(arg):
    """Read project name from csv file or pandas object
    """
    if type(arg) == str: return str(pd.read_csv(arg).keys()[1])

    elif type(arg) == pd.core.frame.DataFrame: return str(arg.keys()[1])

def _read_acquisition_date(arg):
    """Read project name from csv file or pandas object
    """
    if type(arg) == str:

        s = pd.read_csv(arg)
        
        return str(s[s.keys()[1]][0])

    elif type(arg) == pd.core.frame.DataFrame: return str(arg[arg.keys()[1]][0])

def _read_analysed_area(arg):
    """Read analysed area from csv file or pandas object
    """
    if type(arg) == str:

        s = pd.read_csv(arg)
        
        return str(s[s.keys()[1]][3]) + ' ' + str(s['Project name'][5].replace('Analyzed area ', ''))

    elif type(arg) == pd.core.frame.DataFrame: return str(arg[arg.keys()[1]][3]) + ' ' + str(s['Project name'][5].replace('Analyzed area ', ''))

def _read_analysed_views(arg):
    """Read analysed area from csv file or pandas object
    """
    if type(arg) == str:

        s = pd.read_csv(arg)
        
        return str(s[s.keys()[1]][4])

    elif type(arg) == pd.core.frame.DataFrame: return float(arg[arg.keys()[1]][4])

def _read_magnification(arg):
    """Get the magnification used during data acquisition from a pandas object
    """
    
    numbers = re.findall(r'\d+',arg[arg.keys()[1]][15])

    number = ''
    
    for num in numbers: number += num

    return float(number)

def _get_magnification(arg):
    """Read analysed area from csv file or pandas object
    """  
    if type(arg) == str: s = pd.read_csv(str(arg))

    elif type(arg) == pd.core.frame.DataFrame: s = arg
    
    return _read_magnification(s)

def _get_analysed_views(arg):
    """Get the number of analysed views"""
    if type(arg) == str: s = pd.read_csv(arg)

    else: s = arg

    return int(s[s.keys()[1]][6])

def _get_acquisition_time(arg):
    """Read the acquisition time"""
    if type(arg) == str: s = pd.read_csv(arg)

    else: s = arg

    return str(s[s.keys()[1]][22])

def _read_particle_label_names(arg):
    """Read the particles' label name into a list"""
    return list(arg['Label name'])
    
def _get_particle_label_names(arg):
    """Get the particles' label name as a list"""
    if type(arg) == str: s = pd.read_csv(arg)

    else: s = arg

    return _read_particle_label_names(s)