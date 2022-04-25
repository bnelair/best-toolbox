

from best import DELIMITER
from best._config import get_files, config

configs_ArtifactEraser = dict([(f.split(DELIMITER)[-1].split('.')[0], config(f)) for f in get_files(DELIMITER.join(__file__.split(DELIMITER)[:-1]), 'yaml') if not '.-' in f.split(DELIMITER)[-1]])





