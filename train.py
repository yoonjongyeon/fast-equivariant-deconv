import argparse
import os
import numpy as np
import json
import time
import yaml
from joblib import Parallel, delayed
import h5py

from utils.loss import Losses
from utils.subject import SubjectdMRI
from utils.dataset import MultiSubjectdMRI
from utils.sampling import HealpixSampling
from model.model import ModelMultiSubject as Model

import torch
from torch.utils.data.dataloader import DataLoader
import wandb

def main(config, save_path):
    # set GPU
    DEVICE = torch.device(f"cuda:{config['training']['gpu_num']}" if torch.cuda.is_available() else "cpu")
    """Train a model
    """
    # Load the dataset
    assert config['model']['tissues']['wm']
    start = time.time()
    rf_isotropic_names = []
    fodf_isotropic_names = []
    if config['model']['tissues']['gm']:
        rf_isotropic_names.append('gm_response')
        fodf_isotropic_names.append('fodf_gm')
    if config['model']['tissues']['csf']:
        rf_isotropic_names.append('csf_response')
        fodf_isotropic_names.append('fodf_csf')
    if os.path.exists(f"{config['data']['data_path']}/train_subjects.txt"):
        subject_list_path = np.loadtxt(f"{config['data']['data_path']}/train_subjects.txt", dtype=str, ndmin=1)
    elif os.path.exists(f"{config['data']['data_path']}/features.nii.gz"):
        subject_list_path = [config['data']['data_path']]
    else:
        raise ValueError(f"Data path {config['data']['data_path']} does not contain any subject")
    # subset train data
    subject_list_path = subject_list_path[:config['data']['train_num']]
    print('-'*50)
    print('-'*6, ' Start Training ', '-'*6)
    print(f'Load {len(subject_list_path)} subjects: {subject_list_path}')
    if config['data']['loading_method']=='h5':
        print(f"Create/Get h5 file at {config['data']['data_path']}/subjects.hdf5")
        f = h5py.File(f"{config['data']['data_path']}/subjects.hdf5", 'a')
    else:
        f = None
    subject_list = Parallel(n_jobs=config['data']['cpu_subject_loader'])(delayed(SubjectdMRI)(subject_path, response_function_name=config['data']['rf_name'], verbose=True,
                    features_name='features', mask_name='T1_WM', bvecs_name='bvecs.bvecs', bvals_name='bvals.bvals', gradient_mask_input_name=config['data']['gradient_mask'],
                    rf_isotropic_names=rf_isotropic_names, fodf_path=config['data']['fodf_path'], fodf_isotropic_names=fodf_isotropic_names, normalize_per_shell=config['data']['normalize_per_shell'], normalize_in_mask=config['data']['normalize_in_mask'], sh_degree=config['model']['sh_degree'], loading_method=config['data']['loading_method'], h5_file=f) for subject_path in subject_list_path)
    dataset = MultiSubjectdMRI(subject_list, patch_size=config['model']['patch_size'], concatenate=config['model']['concatenate'], verbose=True)
    dataloader_train = DataLoader(dataset=dataset, batch_size=config['training']['batch_size'], shuffle=True, num_workers=config['data']['cpu_dataloader'])
    n_batch = min(config['data']['max_n_batch'], len(dataloader_train))

    has_validation = False
    if not config['data']['data_path_validation'] is None:
        subject_list_path_validation = np.loadtxt(f"{config['data']['data_path_validation']}/val_subjects.txt", dtype=str, ndmin=1)
        # subset validation data
        subject_list_path_validation = subject_list_path_validation[:config['data']['val_num']]
        print(f'Load {len(subject_list_path_validation)} Validation subjects: {subject_list_path_validation}')
        if config['data']['loading_method']=='h5':
            print(f"Create/Get h5 file at {config['data']['data_path_validation']}/subjects.hdf5")
            f_val = h5py.File(f"{config['data']['data_path_validation']}/subjects.hdf5", 'a')
        else:
            f_val = None
        subject_list_val = Parallel(n_jobs=config['data']['cpu_subject_loader'])(delayed(SubjectdMRI)(subject_path, response_function_name=config['data']['rf_name'], verbose=True,
                        features_name='features', mask_name='T1_WM', bvecs_name='bvecs.bvecs', bvals_name='bvals.bvals', gradient_mask_input_name=config['data']['gradient_mask'],
                        rf_isotropic_names=rf_isotropic_names, fodf_path=config['data']['fodf_path'], fodf_isotropic_names=fodf_isotropic_names, normalize_per_shell=config['data']['normalize_per_shell'], normalize_in_mask=config['data']['normalize_in_mask'], sh_degree=config['model']['sh_degree'], loading_method=config['data']['loading_method'], h5_file=f_val) for subject_path in subject_list_path_validation)
        dataset_val = MultiSubjectdMRI(subject_list_val, patch_size=config['model']['patch_size'], concatenate=config['model']['concatenate'], verbose=True)
        dataloader_val = DataLoader(dataset=dataset_val, batch_size=config['training']['batch_size_val'], shuffle=True, num_workers=config['data']['cpu_dataloader'])
        n_batch_val = min(config['data']['max_n_batch_val'], len(dataloader_val))
        has_validation = True
    time_dataset = time.time()

    # Update config file with unique input and output bvals
    #dataset.bvals_input
    #dataset.bvals_output
    config['data']['bvals_input'] = dataset.bvals_input
    config['data']['bvals_output'] = dataset.bvals_output

    # Create response function filter
    if dataset.group_response_functions.n_anisotropic>0:
        polar_filter_equi = torch.Tensor(dataset.group_response_functions.rf_anisotropic_mean)
        if config['model']['train_rf']:
            # add correct number of 0 to the shc of the response function
            n_shc = int(config['model']['sh_degree'] / 2 + 1)
            if polar_filter_equi.shape[2] < n_shc:
                add =  torch.zeros((*polar_filter_equi.shape[:2], n_shc - polar_filter_equi.shape[2]), dtype=polar_filter_equi.dtype)
                polar_filter_equi = torch.cat((polar_filter_equi, add), dim=2)
    else:
        polar_filter_equi = None
    if dataset.group_response_functions.n_isotropic>0:
        polar_filter_inva = torch.Tensor(dataset.group_response_functions.rf_isotropic_mean)
    else:
        polar_filter_inva = None

    
    # Create the deconvolution model
    # Get patch size
    patch_size = config['model']['patch_size']
    # Get input features and update if concatenate mode activated
    feature_in = dataset.max_n_shell_input
    if config['model']['concatenate']:
        feature_in = feature_in * (patch_size**3)
        patch_size = 1
    # Load the graph Sampling
    graphSampling = HealpixSampling(config['model']['n_side'], config['model']['depth'], patch_size, sh_degree=config['model']['sh_degree'], pooling_name=config['model']['conv_name'], pooling_mode='average', hemisphere=config['model']['use_hemisphere'], legacy=config['model']['use_legacy']) # I changed it to max pooling, switching back now (looking for performance drop since midl). Max pooling: bad
    # Update input feature depending on convolution mode
    if config['model']['conv_name'] in ['spatial', 'spatial_vec']:
        feature_in = feature_in * graphSampling.sampling.SH2S.shape[1]
    elif config['model']['conv_name'] in ['spatial_sh']:
        feature_in = feature_in * dataset.max_n_shc_input
    config['training']['feature_in'] = feature_in
    # Create the model
    model = Model(graphSampling, polar_filter_equi, polar_filter_inva, feature_in, config['model']['filter_start'], config['model']['kernel_sizeSph'], config['model']['kernel_sizeSpa'], config['model']['normalize'], config['model']['conv_name'], config['model']['isoSpa'], config['model']['train_rf'])
    n_params = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print(f'- Number of learnable params in Model: {n_params}')
    # If applicable, load pre trained network
    if not config['training']['load_state'] is None:
        print(f"Load pre-trained model at : {config['training']['load_state']}")
        model.load_state_dict(torch.load(config['training']['load_state']), strict=False)
    # Load model in GPU
    model = model.to(DEVICE)
    torch.save(model.state_dict(), os.path.join(save_path, 'history', 'epoch_0.pth'))
    # Send polar filter to GPU
    if not polar_filter_equi is None:
        polar_filter_equi = polar_filter_equi.to(DEVICE)
    if not polar_filter_inva is None:
        polar_filter_inva = polar_filter_inva.to(DEVICE)
        
     # extract loss used in training
    loss_types = {}
    for loss_type in list(config['loss'].keys()):
        for loss_type2 in list(config['loss'][loss_type].keys()):
            if config['loss'][loss_type][loss_type2]['weight'] != 0:
                loss_types[f'{loss_type}_{loss_type2}'] = config['loss'][loss_type][loss_type2]['weight']
    
    # wandb
    if config['wandb']['use_wandb']:
        wandb_config = {
            'gpu number': config['training']['gpu_num'],
            'number of train data': config['data']['train_num'],
            'number of validation data': config['data']['val_num'],
            'number of epochs': config['training']['n_epoch'],
            'learning rate': config['training']['lr']
            }
        # add loss used in training in wandb_config
        for key in list(loss_types.keys()):
            wandb_config[key] =  loss_types[key]
        exp_name = f'{config['training']['expname']}_{config['wandb']['exp_date']}'
        wandb.init(project = "macro_aware_fods", name = exp_name, config = wandb_config)

    # Loss
    has_equi = config['model']['tissues']['wm']
    has_inva = config['model']['tissues']['gm'] + config['model']['tissues']['csf']
    has_fodf = not config['data']['fodf_path'] is None
    losses = Losses(config['loss'], config['model']['sh_degree'], has_equi, has_inva, has_fodf, config['training']['compute_extra_loss'], config['training']['n_epoch'], n_batch, prefix_dataset='train', verbose=True)
    losses.to(DEVICE)
    if has_validation:
        losses_val = Losses(config['loss'], config['model']['sh_degree'], has_equi, has_inva, has_fodf, config['training']['compute_extra_loss'], config['training']['n_epoch'], n_batch_val, prefix_dataset='val', verbose=True)
        losses_val.to(DEVICE)

    # Optimizer/Scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70,80,90], gamma=0.1)
    time_model = time.time()
    print(f'Create model: {time_model - time_dataset:.4f}s, Dataset: {time_dataset - start:.4f}s')
    print('-'*50)
    # Training loop
    for epoch in range(config['training']['n_epoch']):
        print(f'----------Current Epoch: {epoch+1}----------')
        # TRAIN
        model.train()

        # Train on batch.
        start = time.time()
        for batch, data in enumerate(dataloader_train):
            if batch == n_batch:
                break
            
            # Delete all previous gradients
            optimizer.zero_grad()

            # Load the data in the DEVICE
            input_features = data['input_features'].to(DEVICE)
            output_features = data['output_features'].to(DEVICE)
            output_mask = data['output_mask'].to(DEVICE)
            output_b0 = data['output_b0'].to(DEVICE)
            input_signal_to_shc = data['input_signal_to_shc'].to(DEVICE)
            output_shc_to_signal = data['output_shc_to_signal'].to(DEVICE)

            # Run model
            output_reconstructed, deconvolved_equi_shc, deconvolved_inva_shc = model(input_features, output_b0, input_signal_to_shc, output_shc_to_signal)
            if len(rf_isotropic_names) == 0: 
                equi_polar_filter_shc = model.reconstruction.conv_equi.polar_filter
                inva_polar_filter_shc = None
            else:
                equi_polar_filter_shc, inva_polar_filter_shc = model.reconstruction.conv_equi.polar_filter, model.reconstruction.conv_inv.polar_filter

            # Compute loss
            loss_input = {'reconstruction': output_reconstructed, 'target': output_features, 'mask': output_mask,
              'equi_deconvolved_shc': deconvolved_equi_shc, 'inva_deconvolved_shc': deconvolved_inva_shc,
              'equi_polar_filter_shc': equi_polar_filter_shc, 'inva_polar_filter_shc': inva_polar_filter_shc,
              'equi_target_polar_filter_shc': polar_filter_equi, 'inva_target_polar_filter_shc': polar_filter_inva}
            if not config['data']['fodf_path'] is None:
                output_anisotropic_fodf = data['output_anisotropic_fodf'].to(DEVICE)
                loss_input['equi_deconvolved_shc_target'] = output_anisotropic_fodf
                if len(fodf_isotropic_names)>0:
                    output_isotropic_fodf = data['output_isotropic_fodf'].to(DEVICE)
                    loss_input['inva_deconvolved_shc_target'] = output_isotropic_fodf

            loss = losses(**loss_input)

            ###############################################################################################
            # Loss backward
            loss.backward()
            optimizer.step()
            ###############################################################################################
            
        ###############################################################################################
        # Save and print mean loss for the epoch
        mean_train_loss, train_loss_dict = losses.end_epoch()
        print(f'Epoch {epoch+1} Train mean loss: {mean_train_loss}')
        # log wandb
        if config['wandb']['use_wandb']:
            wandb.log(train_loss_dict, step=epoch)
        scheduler.step()

        ###############################################################################################
        # Save the loss and model
        torch.save(model.state_dict(), os.path.join(save_path, 'history', f'epoch_{epoch + 1}.pth'))
        config['training']['last_epoch'] = epoch + 1
        yaml.safe_dump(config, open(os.path.join(save_path, 'config.yml'), 'w'), default_flow_style=False)
        if config['training']['only_save_last']:
            os.remove(os.path.join(save_path, 'history', f'epoch_{epoch}.pth'))

        if has_validation:
            # VALIDATION
            model.eval()
            with torch.no_grad():
                # Train on batch.
                start = time.time()
                for batch, data in enumerate(dataloader_val):
                    if batch == n_batch_val:
                        break
                    
                    # Load the data in the DEVICE
                    input_features = data['input_features'].to(DEVICE)
                    output_features = data['output_features'].to(DEVICE)
                    output_mask = data['output_mask'].to(DEVICE)
                    output_b0 = data['output_b0'].to(DEVICE)
                    input_signal_to_shc = data['input_signal_to_shc'].to(DEVICE)
                    output_shc_to_signal = data['output_shc_to_signal'].to(DEVICE)

                    # Run model
                    output_reconstructed, deconvolved_equi_shc, deconvolved_inva_shc = model(input_features, output_b0, input_signal_to_shc, output_shc_to_signal)
                    if len(rf_isotropic_names) == 0: 
                        equi_polar_filter_shc = model.reconstruction.conv_equi.polar_filter
                        inva_polar_filter_shc = None
                    else:
                        equi_polar_filter_shc, inva_polar_filter_shc = model.reconstruction.conv_equi.polar_filter, model.reconstruction.conv_inv.polar_filter

                    # Compute loss
                    loss_input = {'reconstruction': output_reconstructed, 'target': output_features, 'mask': output_mask,
                    'equi_deconvolved_shc': deconvolved_equi_shc, 'inva_deconvolved_shc': deconvolved_inva_shc,
                    'equi_polar_filter_shc': equi_polar_filter_shc, 'inva_polar_filter_shc': inva_polar_filter_shc,
                    'equi_target_polar_filter_shc': polar_filter_equi, 'inva_target_polar_filter_shc': polar_filter_inva}
                    if not config['data']['fodf_path'] is None:
                        output_anisotropic_fodf = data['output_anisotropic_fodf'].to(DEVICE)
                        loss_input['equi_deconvolved_shc_target'] = output_anisotropic_fodf
                        if len(fodf_isotropic_names)>0:
                            output_isotropic_fodf = data['output_isotropic_fodf'].to(DEVICE)
                            loss_input['inva_deconvolved_shc_target'] = output_isotropic_fodf
                    loss = losses_val(**loss_input)

            # Print mean loss for the epoch
            mean_val_loss, val_loss_dict = losses_val.end_epoch()
            # log wandb
            if config['wandb']['use_wandb']:
                wandb.log(val_loss_dict, step=epoch)
            print(f'Epoch {epoch+1} Validation mean loss: {mean_val_loss}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help='Path of the config file (default: None)',
        type=str
    )
    args = parser.parse_args()
    config_path = args.config
    config = yaml.safe_load(open(config_path, 'r'))
    
    # Save directory
    # save_path = f"{config['data']['data_path']}/train_setting/{config['training']['expname']}"
    save_path = f"/home/jongyeon/train_setting/{config['training']['expname']}"
    if not os.path.exists(save_path):
        print(f'Create new directory: {save_path}')
        os.makedirs(save_path, exist_ok=True)

    # History directory
    history_path = os.path.join(save_path, 'history')
    if not os.path.exists(history_path):
        print(f'Create new directory: {history_path}')
        os.makedirs(history_path, exist_ok=True)
        
    # Save config file to txt file
    config_txt_path = os.path.join(save_path, 'config.txt')
    with open(config_txt_path, 'w') as file:
        json.dump(config, file, indent=2)
    
    # Save parameters
    with open(os.path.join(save_path, 'args.txt'), 'w') as file:
        json.dump(args.__dict__, file, indent=2)

    main(config, save_path)