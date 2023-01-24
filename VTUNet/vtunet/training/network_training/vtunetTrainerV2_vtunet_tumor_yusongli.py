import contextlib
from vtunet.training.network_training.vtunetTrainerV2_vtunet_tumor_base import vtunetTrainerV2_vtunet_tumor_base

class new_fine_vtunet_epoch500(vtunetTrainerV2_vtunet_tumor_base):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_num_epochs = 500

    def load_plans_file(self):
        super().load_plans_file()
        batch_size = 4
        self.plans['plans_per_stage'][0]['batch_size'] = batch_size
        with contextlib.suppress(Exception):
            self.plans['plans_per_stage'][1]['batch_size'] = batch_size
