import os
import platform
import shutil
import sys
import time
from collections import OrderedDict
from typing import Optional, Any

import numpy as np
import segmentation_models_pytorch as smp
import torch
import wandb
from torch.cuda.amp import autocast
from torch.utils.data import WeightedRandomSampler
from torchmetrics import MeanMetric
from torchvision.transforms import transforms
from tqdm.auto import tqdm

import visualization
from config import PreprocessedSatelliteDataset, FixValDataset
from config import means as meanDict
from config import percentiles as percentileDict
from config import stds as stdDict
from metrics import MetricsClass
from utilities import JointRandomRotationTransform
from utilities import SequentialSchedulers


class Runner:
    """Base class for all runners, defines the general functions"""

    def __init__(self, config: Any, tmp_dir: str, debug: bool):
        """
        Initialize useful variables using config.
        :param config: wandb run config
        :type config: wandb.config.Config
        :param debug: Whether we are in debug mode or not
        :type debug: bool
        """
        self.config = config
        self.debug = debug

        n_gpus = torch.cuda.device_count()
        if n_gpus > 0:
            config.update(dict(device='cuda:0'))
        else:
            config.update(dict(device='cpu'))

        self.dataParallel = (torch.cuda.device_count() > 1)
        if not self.dataParallel:
            self.device = torch.device(config.device)
            if 'gpu' in config.device:
                torch.cuda.set_device(self.device)
        else:
            # Use all visible GPUs
            self.device = torch.device("cuda:0")
            torch.cuda.device(self.device)
        torch.backends.cudnn.benchmark = True

        # Set a couple useful variables
        self.seed = int(self.config.seed)
        self.loss_name = self.config.loss_name or 'shift_l1'
        sys.stdout.write(f"Using loss: {self.loss_name}.\n")
        self.use_amp = self.config.fp16
        self.tmp_dir = tmp_dir
        print(f"Using temporary directory {self.tmp_dir}.")
        self.label_rescaling_factor = 1.
        if self.config.use_label_rescaling:
            self.label_rescaling_factor = 60.
            sys.stdout.write(f"Using label rescaling of {self.label_rescaling_factor} - this is hardcoded.\n")

        # Variables to be set
        self.loader = {loader_type: None for loader_type in ['train', 'val']}
        self.loss_criteria = {loss_name: self.get_loss(loss_name=loss_name) for loss_name in ['shift_l1', 'shift_l2', 'shift_huber', 'l1', 'l2', 'huber']}
        for threshold in [15, 20, 25, 30]:
            self.loss_criteria[f"l1_{threshold}"] = self.get_loss(loss_name=f"l1", threshold=threshold)

        self.optimizer = None
        self.scheduler = None
        self.model = None
        self.artifact = None
        self.model_paths = {model_type: None for model_type in ['initial', 'trained']}
        self.model_metrics = {  # Organized way of saving metrics needed for retraining etc.
        }

        self.metrics = {mode: {'loss': MeanMetric().to(device=self.device),
                               'shift_l1': MeanMetric().to(device=self.device),
                               'shift_l2': MeanMetric().to(device=self.device),
                               'shift_huber': MeanMetric().to(device=self.device),
                               'l1': MeanMetric().to(device=self.device),
                               'l2': MeanMetric().to(device=self.device),
                               'huber': MeanMetric().to(device=self.device),
                               }
                        for mode in ['train', 'val']}

        for mode in ['train', 'val']:
            for threshold in [15, 20, 25, 30]:
                self.metrics[mode][f"l1_{threshold}"] = MeanMetric().to(device=self.device)

        self.metrics['train']['ips_throughput'] = MeanMetric().to(device=self.device)

    @staticmethod
    def set_seed(seed: int):
        """
        Sets the seed for the current run.
        :param seed: seed to be used
        """
        # Set a unique random seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        # Remark: If you are working with a multi-GPU model, this function is insufficient to get determinism. To seed all GPUs, use manual_seed_all().
        torch.cuda.manual_seed(seed)  # This works if CUDA not available

    def reset_averaged_metrics(self):
        """Resets all metrics"""
        for mode in self.metrics.keys():
            for metric in self.metrics[mode].values():
                metric.reset()

    def get_metrics(self) -> dict:
        """
        Returns the metrics for the current epoch.
        :return: dict containing the metrics
        :rtype: dict
        """
        with torch.no_grad():
            loggingDict = dict(
                # Model metrics
                n_params=MetricsClass.get_parameter_count(model=self.model),

                # Optimizer metrics
                learning_rate=float(self.optimizer.param_groups[0]['lr']),
            )
            # Add metrics
            for split in ['train', 'val']:
                for metric_name, metric in self.metrics[split].items():
                    try:
                        # Catch case where MeanMetric mode not set yet
                        loggingDict[f"{split}/{metric_name}"] = metric.compute()
                    except Exception as e:
                        continue

        return loggingDict

    @staticmethod
    def get_dataset_root(dataset_name: str) -> str:
        """Copies the dataset and returns the rootpath."""
        # Determine where the data lies
        for root in ['/home/ubuntu/work/satellite_data/sentinel_pauls_paper/', '/home/htc/mzimmer/SCRATCH/', './datasets_pytorch/', '/home/jovyan/work/scratch/']:  # SCRATCHAIS2T, local, scratch_jan
            rootPath = f"{root}{dataset_name}"
            if os.path.isdir(rootPath):
                break
        """
        is_htc = (root == '/home/htc/mzimmer/SCRATCH/') and 'htc-' in platform.uname().node
        is_copyable = is_htc and ('_camera' in dataset_name or '_better_mountains' in dataset_name)
        sys.stdout.write(f"Dataset {dataset_name} is copyable: {is_copyable}.\n")

        if is_copyable:
            # We copy the data to have it on locally attached hardware
            local = '/scratch/local/'
            if not os.path.isdir(os.path.join(local, 'mzimmer')): os.mkdir(os.path.join(local, 'mzimmer'))
            local = local + 'mzimmer/'
            localPath = f"{local}{dataset_name}"
            inProcessFile = os.path.join(local, f"{dataset_name}-inprocess.lock")
            doneFile = os.path.join(local, f"{dataset_name}-donefile.lock")

            wait_it = 0
            while True:
                is_done = os.path.exists(doneFile) and os.path.isdir(f"{local}{dataset_name}")
                is_busy = os.path.exists(inProcessFile)
                if is_done:
                    # Dataset exists locally, continue with the training
                    rootPath = f"{local}{dataset_name}"
                    print("Local data storage: Done file exists.")
                    break
                elif is_busy:
                    # Wait for 10 seconds, then check again
                    time.sleep(10)
                    print("Local data storage: Is still busy - wait.")
                    continue
                else:
                    # Create the inProcessFile
                    open(inProcessFile, mode='a').close()

                    # Copy the dataset
                    print("Local data storage: Starts copying.")
                    shutil.copytree(src=rootPath, dst=localPath)
                    print("Local data storage: Copying done.")
                    # Create the doneFile
                    open(doneFile, mode='a').close()

                    # Remove the inProcessFile
                    os.remove(inProcessFile)

                wait_it += 1
                if wait_it == 360:
                    # Waited 1 hour, this should be done by now, check for errors
                    raise Exception("Waiting time too long.")
        """
        return rootPath

    def get_dataloaders(self):
        rootPath = self.get_dataset_root(dataset_name=self.config.dataset)
        print(f"Loading {self.config.dataset} dataset from {rootPath}.")

        data_path = rootPath
        train_dataframe = os.path.join(rootPath, 'train.csv')
        val_dataframe = os.path.join(rootPath, 'val.csv')
        fix_val_dataframe = os.path.join(rootPath, 'fix_val.csv')

        transformDict = {split: None for split in ['train', 'val']}
        base_transform = transforms.ToTensor()
        transforms_list = [base_transform]   # Convert to tensor (this changes the order of the channels)
        if self.config.use_standardization:
            assert self.config.use_input_clipping not in [False, None, 'None'], "Mutually exclusive options: use_standardization and use_input_clipping."
            assert self.config.dataset in meanDict.keys(), f"Mean of Dataset {self.config.dataset} not implemented."
            assert self.config.dataset in stdDict.keys(), f"Std of Dataset {self.config.dataset} not implemented."
            mean, std = meanDict[self.config.dataset], stdDict[self.config.dataset]
            normalize_transform = transforms.Normalize(mean=mean, std=std)
            transforms_list.append(normalize_transform)
        elif self.config.use_input_clipping not in [False, None, 'None']:
            assert not self.config.use_standardization, "Mutually exclusive options: use_standardization and use_input_clipping."
            use_input_clipping = int(self.config.use_input_clipping)
            assert use_input_clipping in [1, 2, 5], "use_input_clipping must be in [False, None, 1, 2, 5]."
            sys.stdout.write(f"Using input clipping of in the {use_input_clipping}-{100-use_input_clipping}-range.\n")
            input_clipping_lower_bound = percentileDict[self.config.dataset][use_input_clipping]
            input_clipping_upper_bound = percentileDict[self.config.dataset][100 - use_input_clipping]

            # Convert the bounds to tensors
            input_clipping_lower_bound = torch.tensor(input_clipping_lower_bound, dtype=torch.float).view(-1, 1, 1) # View to make it a 3D tensor
            input_clipping_upper_bound = torch.tensor(input_clipping_upper_bound, dtype=torch.float).view(-1, 1, 1) # View to make it a 3D tensor

            # Define the clipping transform over the channels, i.e. each channel is clipped individually using the bounds from percentileDict
            clipping_transform = transforms.Lambda(lambda x: torch.clamp(x, min=input_clipping_lower_bound, max=input_clipping_upper_bound))
            transforms_list.append(clipping_transform)


        transformDict['train'] = transforms.Compose(transforms_list)
        transformDict['val'] = transforms.Compose(transforms_list)

        # Create the label transform to rescale the labels
        label_transforms = transforms.Compose([base_transform, lambda x: x * (1./self.label_rescaling_factor)])

        joint_transforms = None # Train transforms that are both applied to the image and the label
        if self.config.use_augmentation:
            sys.stdout.write(f"Using JointRandomRotationTransform.\n")
            joint_transforms = JointRandomRotationTransform()

        use_weighted_sampler = self.config.use_weighted_sampler or False

        remove_corrupt = not self.debug
        trainData = PreprocessedSatelliteDataset(data_path=data_path, dataframe=train_dataframe, image_transforms=transformDict['train'], label_transforms=label_transforms, joint_transforms=joint_transforms,
                                                 use_weighted_sampler=use_weighted_sampler, use_weighting_quantile=self.config.use_weighting_quantile, use_memmap=self.config.use_memmap, remove_corrupt=remove_corrupt)
        valData = PreprocessedSatelliteDataset(data_path=data_path, dataframe=val_dataframe,
                                               image_transforms=transformDict['val'], label_transforms=label_transforms, use_memmap=self.config.use_memmap, remove_corrupt=remove_corrupt)
        fixvalData = FixValDataset(data_path=data_path, dataframe=fix_val_dataframe,
                                               image_transforms=transformDict['val'])

        sys.stdout.write(f"Length of train and val splits: {len(trainData)}, {len(valData)}.\n")
        cut_off = 3000
        if len(valData) >= cut_off:
            sys.stdout.write(f"Validation dataset is large, reducing to a maximum of {cut_off} samples.\n")
            # Reduce the size of the validation dataset using self.seed as the random seed
            # Perform a random split using a generator with self.seed as the seed
            valData, _ = torch.utils.data.random_split(valData, [cut_off, len(valData) - cut_off], generator=torch.Generator().manual_seed(self.seed))
        sys.stdout.write(f"New length of train and val splits: {len(trainData)}, {len(valData)}.\n")


        num_workers_default = self.config.num_workers_per_gpu if self.config.num_workers_per_gpu is not None else 8
        num_workers = num_workers_default * torch.cuda.device_count() * int(not self.debug)
        sys.stdout.write(f"Using {num_workers} workers.\n")
        train_sampler = None
        shuffle = True
        if use_weighted_sampler:
            train_sampler = WeightedRandomSampler(trainData.weights, len(trainData))
            shuffle = False

        trainLoader = torch.utils.data.DataLoader(trainData, batch_size=self.config.batch_size, sampler=train_sampler,
                                                  pin_memory=torch.cuda.is_available(), num_workers=num_workers,
                                                  shuffle=shuffle)
        valLoader = torch.utils.data.DataLoader(valData, batch_size=self.config.batch_size, shuffle=False,
                                                pin_memory=torch.cuda.is_available(), num_workers=num_workers)
        fixvalLoader = torch.utils.data.DataLoader(fixvalData, batch_size=2, shuffle=False, # This only works with batch_size=2
                                                pin_memory=torch.cuda.is_available(), num_workers=num_workers)

        return trainLoader, valLoader, fixvalLoader

    def get_model(self, reinit: bool, model_path: Optional[str] = None) -> torch.nn.Module:
        """
        Returns the model.
        :param reinit: If True, the model is reinitialized.
        :type reinit: bool
        :param model_path: Path to the model.
        :type model_path: Optional[str]
        :return: The model.
        :rtype: torch.nn.Module
        """
        print(
            f"Loading model - reinit: {reinit} | path: {model_path if model_path else 'None specified'}.")
        if reinit:
            # Define the model
            arch = self.config.arch or 'unet'
            backbone = self.config.backbone or 'resnet50'
            sys.stdout.write(f"Using architecture {arch}.\n")
            assert arch in ['unet', 'unetpp', 'manet', 'linknet', 'fpn', 'pspnet', 'pan', 'deeplabv3', 'deeplabv3p']
            network_config = {
                "encoder_name": backbone,
                "encoder_weights": None if not self.config.use_pretrained_model else 'imagenet',
                "in_channels": 14,
                "classes": 1,
            }
            if arch == 'unet':
                model = smp.Unet(**network_config)
            elif arch == 'unetpp':
                model = smp.UnetPlusPlus(**network_config)
            elif arch == 'manet':
                model = smp.MAnet(**network_config)
            elif arch == 'linknet':
                model = smp.Linknet(**network_config)
            elif arch == 'fpn':
                model = smp.FPN(**network_config)
            elif arch == 'pspnet':
                model = smp.PSPNet(**network_config)
            elif arch == 'pan':
                model = smp.PAN(**network_config)
            elif arch == 'deeplabv3':
                model = smp.DeepLabV3(**network_config)
            elif arch == 'deeplabv3p':
                model = smp.DeepLabV3Plus(**network_config)
        else:
            # The model has been initialized already
            model = self.model

        if model_path is not None:
            # Load the model
            state_dict = torch.load(model_path, map_location=self.device)

            new_state_dict = OrderedDict()
            require_DP_format = isinstance(model,
                                           torch.nn.DataParallel)  # If true, ensure all keys start with "module."
            for k, v in state_dict.items():
                is_in_DP_format = k.startswith("module.")
                if require_DP_format and is_in_DP_format:
                    name = k
                elif require_DP_format and not is_in_DP_format:
                    name = "module." + k  # Add 'module' prefix
                elif not require_DP_format and is_in_DP_format:
                    name = k[7:]  # Remove 'module.'
                elif not require_DP_format and not is_in_DP_format:
                    name = k
                new_state_dict[name] = v
            # Load the state_dict
            model.load_state_dict(new_state_dict)

        if self.dataParallel and reinit and not isinstance(model, torch.nn.DataParallel):
            # Only apply DataParallel when re-initializing the model!
            # We use DataParallelism
            model = torch.nn.DataParallel(model)
        model = model.to(device=self.device)
        return model

    def get_loss(self, loss_name: str, threshold: float = None):
        assert loss_name in ['shift_l1', 'shift_l2', 'shift_huber', 'l1', 'l2', 'huber'], f"Loss {loss_name} not implemented."
        if threshold is not None:
            assert loss_name == 'l1', f"Threshold only implemented for l1 loss, not {loss_name}."
        # Dim 1 is the channel dimension, 0 is batch.
        # Sums up to get average height, could be mean without zeros
        remove_sub_track = lambda out, target: (out, torch.sum(target, dim=1))

        if loss_name == 'shift_l1':
            from losses.shift_l1_loss import ShiftL1Loss
            loss = ShiftL1Loss(ignore_value=0)
        elif loss_name == 'shift_l2':
            from losses.shift_l2_loss import ShiftL2Loss
            loss = ShiftL2Loss(ignore_value=0)
        elif loss_name == 'shift_huber':
            from losses.shift_huber_loss import ShiftHuberLoss
            loss = ShiftHuberLoss(ignore_value=0)
        elif loss_name == 'l1':
            from losses.l1_loss import L1Loss
            # Rescale the threshold to account for the label rescaling
            if threshold is not None:
                threshold = threshold / self.label_rescaling_factor
            loss = L1Loss(ignore_value=0, pre_calculation_function=remove_sub_track, lower_threshold=threshold)
        elif loss_name == 'l2':
            from losses.l2_loss import L2Loss
            loss = L2Loss(ignore_value=0, pre_calculation_function=remove_sub_track)
        elif loss_name == 'huber':
            from losses.huber_loss import HuberLoss
            loss = HuberLoss(ignore_value=0, pre_calculation_function=remove_sub_track, delta=3.0)
        loss = loss.to(device=self.device)
        return loss

    def get_visualization(self, viz_name: str, inputs, labels, outputs):
        assert viz_name in ['input_output', 'density_scatter_plot',
                            'boxplot'], f"Visualization {viz_name} not implemented."

        # Detach and copy the labels and outputs, then undo the rescaling
        labels, outputs = labels.detach().clone(), outputs.detach().clone()

        # Undo the rescaling
        labels, outputs = labels * self.label_rescaling_factor, outputs * self.label_rescaling_factor


        def remove_sub_track_vis(inputs, labels, outputs):
            return inputs, labels.sum(
                axis=1), outputs  # Same as remove_sub_track, but for visualization (i.e. has outputs as well)

        if viz_name == 'input_output':
            viz_fn = visualization.get_input_output_visualization(rgb_channels=[6, 5, 4],
                                                                  process_variables=remove_sub_track_vis)
        elif viz_name == 'density_scatter_plot':
            viz_fn = visualization.get_density_scatter_plot_visualization(ignore_value=0,
                                                                          process_variables=remove_sub_track_vis)
        elif viz_name == 'boxplot':
            viz_fn = visualization.get_visualization_boxplots(ignore_value=0, process_variables=remove_sub_track_vis)
        return viz_fn(inputs=inputs, labels=labels, outputs=outputs)

    def get_optimizer(self, initial_lr: float) -> torch.optim.Optimizer:
        """
        Returns the optimizer.
        :param initial_lr: The initial learning rate
        :type initial_lr: float
        :return: The optimizer.
        :rtype: torch.optim.Optimizer
        """
        wd = self.config['weight_decay'] or 0.
        optim_name = self.config.optim or 'AdamW'
        if optim_name == 'SGD':
            optimizer = torch.optim.SGD(params=self.model.parameters(), lr=initial_lr,
                                        momentum=0.9,
                                        weight_decay=wd,
                                        nesterov=wd > 0.)
        elif optim_name == 'AdamW':
            optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=initial_lr,
                                          weight_decay=wd)
        else:
            raise NotImplementedError

        return optimizer

    def save_model(self, model_identifier: str, sync: bool = False) -> str:
        """
        Saves the model's state_dict to a file.
        :param model_identifier: Name of the file type.
        :type model_identifier: str
        :param sync: Whether to sync the file to wandb.
        :type sync: bool
        :return: Path to the saved model.
        :rtype: str
        """
        fName = f"{model_identifier}_model.pt"
        fPath = os.path.join(self.tmp_dir, fName)

        # Only save models in their non-module version, to avoid problems when loading
        try:
            model_state_dict = self.model.module.state_dict()
        except AttributeError:
            model_state_dict = self.model.state_dict()

        torch.save(model_state_dict, fPath)  # Save the state_dict

        if sync:
            wandb.save(fPath)
        return fPath

    def log(self, step: int, phase_runtime: float):
        """
        Logs the current training status.
        :param phase_runtime: The wall-clock time of the current phase.
        :type phase_runtime: float
        """
        loggingDict = self.get_metrics()
        loggingDict.update({
            'phase_runtime': phase_runtime,
            'iteration': step,
            'samples_seen': step * self.config.batch_size,
        })

        # Log and push to Wandb
        for metric_type, val in loggingDict.items():
            wandb.run.summary[f"{metric_type}"] = val

        wandb.log(loggingDict)

    def define_optimizer_scheduler(self):
        # Define the optimizer
        initial_lr = self.config.initial_lr
        self.optimizer = self.get_optimizer(initial_lr=initial_lr)

        # We define a scheduler. All schedulers work on a per-iteration basis
        n_total_iterations = self.config.n_iterations
        n_lr_cycles = self.config.n_lr_cycles or 0
        cyclic_mode = self.config.cyclic_mode or 'triangular2'
        n_warmup_iterations = int(0.1 * n_total_iterations) if n_lr_cycles == 0 else 0

        # Set the initial learning rate
        for param_group in self.optimizer.param_groups: param_group['lr'] = initial_lr

        # Define the warmup scheduler if needed
        warmup_scheduler, milestone = None, None
        if n_warmup_iterations > 0:
            # As a start factor we use 1e-20, to avoid division by zero when putting 0.
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                                 start_factor=1e-20, end_factor=1.,
                                                                 total_iters=n_warmup_iterations)
            milestone = n_warmup_iterations + 1

        n_remaining_iterations = n_total_iterations - n_warmup_iterations
        if n_lr_cycles > 0:
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer=self.optimizer, base_lr=0.,
                                                            max_lr=initial_lr, step_size_up=n_remaining_iterations // (2*n_lr_cycles),
                                                            mode=cyclic_mode,
                                                            cycle_momentum=False)
        else:
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer,
                                                      start_factor=1.0, end_factor=0.,
                                                      total_iters=n_remaining_iterations)

        # Reset base lrs to make this work
        scheduler.base_lrs = [initial_lr if warmup_scheduler else 0. for _ in self.optimizer.param_groups]

        # Define the Sequential Scheduler
        if warmup_scheduler is None:
            self.scheduler = scheduler
        else:
            self.scheduler = SequentialSchedulers(optimizer=self.optimizer, schedulers=[warmup_scheduler, scheduler],
                                                  milestones=[milestone])

    @torch.no_grad()
    def eval(self, data: str):
        """
        Evaluates the model on the given data set.
        :param data: string indicating the data set to evaluate on. Can be 'train' or 'val'.
        :type data: str
        """
        sys.stdout.write(f"Evaluating on {data} split.\n")
        for step, (x_input, y_target) in enumerate(tqdm(self.loader[data]), 1):
            x_input = x_input.to(self.device, non_blocking=True)
            y_target = y_target.to(self.device, non_blocking=True)


            with autocast(enabled=self.use_amp):
                output = self.model.eval()(x_input)
                loss = self.loss_criteria[self.loss_name](output, y_target)
                self.metrics[data]['loss'](value=loss, weight=len(y_target))
                for loss_type in self.loss_criteria.keys():
                    metric_loss = self.loss_criteria[loss_type](output, y_target)
                    # Check if the metric_loss is nan
                    if not torch.isnan(metric_loss):
                        self.metrics[data][loss_type](value=metric_loss, weight=len(y_target))

            if step <= 4:
                # Create the visualizations for the first 4 batches
                for viz_func in ['input_output', 'density_scatter_plot', 'boxplot']:
                    viz = self.get_visualization(viz_name=viz_func, inputs=x_input, labels=y_target, outputs=output)
                    wandb.log({data + '/' + viz_func + "_" + str(step): wandb.Image(viz)}, commit=False)

    @torch.no_grad()
    def eval_fixval(self):
        """Creates the fixval plots and logs them to wandb."""
        sys.stdout.write(f"Creating fixval plots.\n")
        def remove_sub_track_vis_wout_labels(inputs, labels, outputs):
            return inputs, None, outputs  # Same as remove_sub_track, but for visualization (i.e. has outputs as well)

        viz_fn = visualization.get_input_output_visualization(rgb_channels=[6, 5, 4],
                                                                  process_variables=remove_sub_track_vis_wout_labels)
        loggingDict = dict()
        for x_input, fileNames in tqdm(self.loader['fix_val']):
            x_input = x_input.to(self.device, non_blocking=True)
            with autocast(enabled=self.use_amp):
                output = self.model.eval()(x_input)
            viz = viz_fn(inputs=x_input, labels=None, outputs=output)
            jointName = "__".join(fileNames)
            wandb.log({'fixval' + '/' + 'input_output' + '/' + jointName: wandb.Image(viz)}, commit=False)

            # Get the min and max prediction for each image
            flattened_output = output.flatten(start_dim=1)
            min_values = flattened_output.min(dim=1).values
            max_values = flattened_output.max(dim=1).values

            for i, (min_val, max_val) in enumerate(zip(min_values, max_values)):
                # Log the min and max values as numeric values with the appropriate filename
                name = fileNames[i]
                loggingDict[f"fixval/min_{name}"] = min_val.item()
                loggingDict[f"fixval/max_{name}"] = max_val.item()
        wandb.log(loggingDict, commit=False)

    def train(self):
        log_freq, n_iterations = self.config.log_freq, self.config.n_iterations
        ampGradScaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        self.reset_averaged_metrics()
        phase_start = time.time()

        # Define the Stochastic Weight Averaging model
        swa_model = None
        if self.config.use_swa:
            swa_model = torch.optim.swa_utils.AveragedModel(self.model)
            swa_start = max(1, int(0.75 * n_iterations))
            sys.stdout.write(f"SWA model will be updated from iteration {swa_start} onwards.\n")

        # Define the distribution of mixup
        if self.config.use_mixup:
            alpha = 0.2
            beta_distribution = torch.distributions.beta.Beta(alpha, alpha)
            sys.stdout.write(f"Using mixup with alpha={alpha}. Example sample: {beta_distribution.sample()}\n")

        for step in tqdm(range(1, n_iterations + 1, 1)):
            # Reinitialize the train iterator if it reaches the end
            if step == 1 or (step - 1) % len(self.loader['train']) == 0:
                train_iterator = iter(self.loader['train'])

            # Move to CUDA if possible
            x_input, y_target = next(train_iterator)
            x_input = x_input.to(device=self.device, non_blocking=True)
            y_target = y_target.to(device=self.device, non_blocking=True)

            if self.config.use_mixup:
                # Sample from the beta distribution with alpha=0.2
                with torch.no_grad():
                    lam = beta_distribution.sample().to(self.device)
                    index = torch.randperm(x_input.size(0)).to(self.device)
                    x_input = lam * x_input + (1 - lam) * x_input[index, :]
                    y_target = lam * y_target + (1 - lam) * y_target[index, :]

            self.optimizer.zero_grad()

            itStartTime = time.time()
            with autocast(enabled=self.use_amp):
                output = self.model.train()(x_input)
                loss = self.loss_criteria[self.loss_name](output, y_target)
                ampGradScaler.scale(loss).backward()  # Scaling + Backpropagation
                # Unscale the weights manually, normally this would be done by ampGradScaler.step(), but since
                # we might use gradient clipping, this has to be split
                ampGradScaler.unscale_(self.optimizer)
                if self.config.use_grad_clipping:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                ampGradScaler.step(self.optimizer)
                ampGradScaler.update()  # This should happen only once per iteration
                self.scheduler.step()
                self.metrics['train']['loss'](value=loss, weight=len(y_target))

                with torch.no_grad():
                    for loss_type in self.loss_criteria.keys():
                        metric_loss = self.loss_criteria[loss_type](output, y_target)
                        # Check if the metric_loss is nan
                        if not torch.isnan(metric_loss):
                            self.metrics['train'][loss_type](value=metric_loss, weight=len(y_target))
                itEndTime = time.time()
                n_img_in_iteration = int(self.config.batch_size)
                ips = n_img_in_iteration / (itEndTime - itStartTime)  # Images processed per second
                self.metrics['train']['ips_throughput'](ips)

            if swa_model is not None:
                if step >= swa_start:
                    swa_model.update_parameters(self.model)
                    if step == n_iterations:
                        sys.stdout.write(f"Last step reached. Setting model to SWA model.\n")
                        # Set the model to the SWA model by copying the parameters
                        with torch.no_grad():
                            for param1, param2 in zip(self.model.parameters(), swa_model.parameters()):
                                param1.copy_(param2)
                            swa_model = None

            if step % log_freq == 0 or step == n_iterations:
                phase_runtime = time.time() - phase_start
                # Create the visualizations
                for viz_func in ['input_output', 'density_scatter_plot', 'boxplot']:
                    viz = self.get_visualization(viz_name=viz_func, inputs=x_input, labels=y_target, outputs=output)
                    wandb.log({'train/' + viz_func: wandb.Image(viz)}, commit=False)

                # Evaluate the validation dataset
                if not self.debug:
                    self.eval(data='val')

                # Create the fixval plots
                if not self.debug:
                    self.eval_fixval()

                self.log(step=step, phase_runtime=phase_runtime)
                self.reset_averaged_metrics()
                phase_start = time.time()

    def run(self):
        """Controls the execution of the script."""
        # We start training from scratch
        self.set_seed(seed=self.seed)  # Set the seed
        loaders = self.get_dataloaders()
        self.loader['train'], self.loader['val'], self.loader['fix_val'] = loaders
        self.model = self.get_model(reinit=True, model_path=self.model_paths['initial'])  # Load the model

        self.define_optimizer_scheduler()  # This was moved before define_strategy to have the optimizer available

        self.train()  # Train the model
        # Save the trained model and upload it to wandb
        self.save_model(model_identifier='trained', sync=True)
