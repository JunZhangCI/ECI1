
###################
#### Variables ####
###################

# Definition of which manner classes contain which phonemes 
MANNER2PHON = {
    'Vowel': ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'], 
    'Nasal-approximant': ['L', 'M', 'N', 'NG', 'R', 'W', 'Y'],
    'Fricative': ['DH', 'F', 'HH', 'S', 'SH', 'V', 'Z', 'TH', 'ZH'],
    'Stop': ['B', 'D', 'G', 'K', 'P', 'T', 'CH', 'JH', 'Q']
    }

# The reverse mapping from phoneme to manner will be created automatically
PHON2MANNER = {phoneme: manner for manner, phonemes in MANNER2PHON.items() for phoneme in phonemes}

##################
#### Plotting ####
##################

# Colors for manner classes
MANNER_COLORS = {'Vowel': '#D7255D', 
                 'Nasal-approximant': '#FFA500', 
                 'Fricative': '#603E95', 
                 'Stop': '#009DA1'}