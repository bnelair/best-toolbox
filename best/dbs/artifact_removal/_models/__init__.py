from best import DELIMITER
from best._config import get_files
from best.dbs.artifact_removal._configs import configs_ArtifactEraser
from best.dbs.artifact_removal.model import *
import torch


class _models_ArtifactEraser(dict):

    _keys = dict([
        (
            # '_'.join(f[:-3].split(DELIMITER)[-1].split('_')[:3]),
            f[:-3].split(DELIMITER)[-1],
            f
        )
        for f in get_files(DELIMITER.join(__file__.split(DELIMITER)[:-1]), 'pt')
        if not '.-' in f.split(DELIMITER)[-1]
    ])

    def keys(self):
        return self._keys.keys()

    def get_model(self, item):
        f = self._keys[item]
        cfg = configs_ArtifactEraser[item]
        state_dict = torch.load(f, map_location='cpu')
        #mod = None
        # mod = GRU_Denoiser(n_filters = cfg.MODEL.ARCHITECTURE.N_FILTERS)
        exec(f"self.mod={cfg.MODEL.MODEL.split('.')[-1]}(n_filters={cfg.MODEL.ARCHITECTURE.N_FILTERS})")
        self.mod.load_state_dict(state_dict, strict=True)
        self.mod.eval()
        mod = self.mod
        del self.mod
        return mod

    def __getitem__(self, item):
        return self.get_model(item)

    def __call__(self, item):
        return self[item]

    def __str__(self):
        return 'models_ArtifactEraser: ' + str(self._keys.keys())

    def __repr__(self):
        return self.__str__()

models_ArtifactEraser = _models_ArtifactEraser()






