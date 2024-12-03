import os
from mypt.code_utilities import directories_and_files as dirf
from mypt.models.simClr.simClrModel import ResnetSimClr
from mypt.subroutines.neighbors import model_embs as me

from model_train_pytorch.training import run_pipeline, OUTPUT_SHAPE, _TRACK_PROJECT_NAME

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

_BATCH_SIZE = 32

_OUTPUT_DIM = 128



def train_main(model):
    dataset_name = 'imagenette'
    train_data_folder = os.path.join(SCRIPT_DIR, 'data', dataset_name, 'train')
    val_data_folder = os.path.join(SCRIPT_DIR, 'data', dataset_name, 'val')
    
    ckpnt_dir_parent = dirf.process_path(os.path.join(SCRIPT_DIR, 'logs', 'train_logs'), dir_ok=True, file_ok=False)

    # delete any empty folders in the 'train_logs' folder
    dirf.clear_directory(ckpnt_dir_parent, condition=lambda x: os.path.isdir(x) and len(os.listdir(x)) == 0)

    n = len(os.listdir(ckpnt_dir_parent))

    # the run_name will be
    run_name = f'{_TRACK_PROJECT_NAME}_iteration_{n + 1}'

    return run_pipeline(model=model, 
        train_data_folder=train_data_folder, 
        val_data_folder=val_data_folder,
        dataset=dataset_name,
        output_shape=OUTPUT_SHAPE,
        ckpnt_dir=os.path.join(ckpnt_dir_parent, f'iteration_{len(os.listdir(ckpnt_dir_parent)) + 1}'),
        num_epochs=5, 
        batch_size=_BATCH_SIZE, 
        temperature=0.5, 
        seed=0, 
        use_logging=True,
        batch_stats=False,
        debug_loss=True, # get as much insight on the loss as possible
        num_train_samples_per_cls=10,
        num_val_samples_per_cls=10,
        num_warmup_epochs=0,
        run_name=run_name,
        )


if __name__ == '__main__':

    model = ResnetSimClr(input_shape=(3,) + OUTPUT_SHAPE,  
                         output_dim=_OUTPUT_DIM, 
                         num_fc_layers=4, 
                         freeze=True, 
                         architecture=50)
    train_main(model)
