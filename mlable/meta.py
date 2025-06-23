import argparse
import os
import random

# RANDOM #######################################################################

RANDOM_ARGS = {
    '--seed': {
        'type': int,
        'required': False,
        'default': random.randint(0, 2 ** 32),
        'help': 'A seed for reproducible training.',},}

# OUTPUT #######################################################################

OUTPUT_ARGS = {
    '--output_dir': {
        'type': str,
        'required': False,
        'default': 'lora-model',
        'help': 'The output directory where the model predictions and checkpoints will be written.'},
    '--cache_dir': {
        'type': str,
        'required': False,
        'default': None,
        'help': 'The directory where the downloaded models and datasets will be stored.'},
    '--logging_dir': {
        'type': str,
        'required': False,
        'default': 'logs',
        'help': '[TensorBoard](https://www.tensorflow.org/tensorboard) log directory.'},}

# MODEL ########################################################################

MODEL_ARGS = {
    '--model_name': {
        'type': str,
        'required': False,
        'default': 'stable-diffusion-v1-5/stable-diffusion-v1-5',
        'help': 'Path to pretrained model or model identifier from huggingface.co/models.',},
    '--revision': {
        'type': str,
        'required': False,
        'default': None,
        'help': 'Revision of pretrained model identifier from huggingface.co/models.',},
    '--variant': {
        'type': str,
        'required': False,
        'default': None,
        'help': 'Variant of the model files of the pretrained model identifier from huggingface.co/models, e.g. fp16',},
    '--lora_rank': {
        'type': int,
        'required': False,
        'default': 8,
        'help': 'The dimension of the LoRA update matrices.',},}

# DATASET ######################################################################

DATASET_ARGS = {
    '--dataset_name': {
        'type': str,
        'required': False,
        'default': 'apehex/ascii-art-datacompdr-12m',
        'help': 'The name of the Dataset (from the HuggingFace hub) to train on.'},
    '--dataset_config': {
        'type': str,
        'required': False,
        'default': 'default',
        'help': 'The config of the Dataset, leave as None if there\'s only one config.'},
    '--dataset_split': {
        'type': str,
        'required': False,
        'default': 'train',
        'help': 'The split of the Dataset.'},
    '--dataset_dir': {
        'type': str,
        'required': False,
        'default': None,
        'help': 'A folder containing the training data.'},
    '--image_column': {
        'type': str,
        'required': False,
        'default': 'content',
        'help': 'The column of the dataset containing an image.'},
    '--caption_column': {
        'type': str,
        'required': False,
        'default': 'caption',
        'help': 'The column of the dataset containing a caption or a list of captions.'},
    '--max_samples': {
        'type': int,
        'required': False,
        'default': 0,
        'help': 'Truncate the number of training examples to this value if set.'},}

# PREPROCESSING ################################################################

PREPROCESSING_ARGS = {
    '--resolution': {
        'type': int,
        'required': False,
        'default': 512,
        'help': 'The resolution for input images.'},
    '--center_crop': {
        'required': False,
        'default': False,
        'action': 'store_true',
        'help': 'Whether to center (instead of random) crop the input images to the resolution.'},
    '--random_flip': {
        'required': False,
        'default': False,
        'action': 'store_true',
        'help': 'whether to randomly flip images horizontally.'},
    '--image_interpolation_mode': {
        'type': str,
        'required': False,
        'default': 'lanczos',
        'choices': ['bicubic', 'bilinear', 'box', 'hamming', 'lanczos', 'nearest', 'nearest_exact'],
        'help': 'The image interpolation method to use for resizing images.'},}

# CHECKPOINT ###################################################################

CHECKPOINT_ARGS = {
    '--resume_from': {
        'type': str,
        'required': False,
        'default': '',
        'help': 'Use a path saved by `--checkpoint_steps`, or `"latest"` to automatically select the last available checkpoint.'},
    '--checkpoint_steps': {
        'type': int,
        'required': False,
        'default': 256,
        'help': 'Save a checkpoint of the training state every X updates, for resuming with `--resume_from`.'},
    '--checkpoint_limit': {
        'type': int,
        'required': False,
        'default': 0,
        'help': 'Max number of checkpoints to store.'},}

# VALIDATION ###################################################################

VALIDATION_ARGS = {
    '--validation_prompt': {
        'type': str,
        'default': '',
        'required': False,
        'help': 'A prompt that is sampled during training for inference.'},
    '--num_validation_images': {
        'type': int,
        'default': 4,
        'required': False,
        'help': 'Number of images that should be generated during validation with `validation_prompt`.'},
    '--validation_epochs': {
        'type': int,
        'default': 1,
        'required': False,
        'help': 'Run fine-tuning validation every X epochs.'},}

# ITERATION ####################################################################

ITERATION_ARGS = {
    '--batch_dim': {
        'type': int,
        'required': False,
        'default': 1,
        'help': 'Batch size (per device) for the training dataloader.'},
    '--epoch_num': {
        'type': int,
        'required': False,
        'default': 32,
        'help': 'Total number of epochs to perform.'},
    '--step_num': {
        'type': int,
        'required': False,
        'default': 0,
        'help': 'Total number of training steps to perform; overrides epoch_num.'},}

# GRADIENT #####################################################################

GRADIENT_ARGS = {
    '--gradient_accumulation_steps': {
        'type': int,
        'required': False,
        'default': 1,
        'help': 'Number of updates steps to accumulate before performing a backward/update pass.'},
    '--gradient_checkpointing': {
        'required': False,
        'default': False,
        'action': 'store_true',
        'help': 'Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.'},}

# LEARNING RATE ################################################################

LEARNING_ARGS = {
    '--learning_rate': {
        'type': float,
        'required': False,
        'default': 1e-4,
        'help': 'Initial learning rate (after the potential warmup period) to use.'},
    '--scale_lr': {
        'required': False,
        'default': False,
        'action': 'store_true',
        'help': 'Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.'},
    '--lr_scheduler': {
        'type': str,
        'required': False,
        'default': 'cosine',
        'help': 'The scheduler type to use, among ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"]'},
    '--lr_warmup_steps': {
        'type': int,
        'required': False,
        'default': 512,
        'help': 'Number of steps for the warmup in the lr scheduler.'},}

# LOSS #########################################################################

LOSS_ARGS = {
    '--snr_gamma': {
        'type': float,
        'required': False,
        'default': 0.0,
        'help': 'SNR weighting gamma to rebalance the loss; recommended value is 5.0. https://arxiv.org/pdf/2303.09556'},}

# PRECISION ####################################################################

PRECISION_ARGS = {
    '--mixed_precision': {
        'type': str,
        'required': False,
        'default': 'fp16',
        'choices': ['no', 'fp16', 'bf16'],
        'help': 'Choose between fp16 and bf16 (bfloat16).'},
    '--allow_tf32': {
        'required': False,
        'default': False,
        'action': 'store_true',
        'help': 'Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training.'},
    '--use_8bit_adam': {
        'required': False,
        'default': False,
        'action': 'store_true',
        'help': 'Whether or not to use 8-bit Adam from bitsandbytes.'},}

# DISTRIBUTION #################################################################

DISTRIBUTION_ARGS = {
    '--dataloader_num_workers': {
        'type': int,
        'required': False,
        'default': 0,
        'help': 'Number of subprocesses to use for data loading; 0 means that the data will be loaded in the main process.'},
    '--local_rank': {
        'type': int,
        'required': False,
        'default': int(os.environ.get('LOCAL_RANK', -1)),
        'help': 'For distributed training: local_rank'},}

# OPTIMIZER ####################################################################

OPTIMIZER_ARGS = {
    '--adam_beta1': {
        'type': float,
        'required': False,
        'default': 0.9,
        'help': 'The beta1 parameter for the Adam optimizer.'},
    '--adam_beta2': {
        'type': float,
        'required': False,
        'default': 0.999,
        'help': 'The beta2 parameter for the Adam optimizer.'},
    '--adam_weight_decay': {
        'type': float,
        'required': False,
        'default': 1e-2,
        'help': 'Weight decay to use.'},
    '--adam_epsilon': {
        'type': float,
        'required': False,
        'default': 1e-08,
        'help': 'Epsilon value for the Adam optimizer'},
    '--max_grad_norm': {
        'type': float,
        'required': False,
        'default': 1.0,
        'help': 'Max gradient norm.'},}

# FRAMEWORK ####################################################################

FRAMEWORK_ARGS = {
    '--enable_xformers': {
        'required': False,
        'default': False,
        'action': 'store_true',
        'help': 'Whether or not to use xformers.'},}

# DIFFUSION ####################################################################

DIFFUSION_ARGS = {
    '--prediction_type': {
        'type': str,
        'required': False,
        'default': 'epsilon',
        'help': 'The prediction type, among "epsilon", "v_prediction" or `None`.'},
    '--noise_offset': {
        'type': float,
        'required': False,
        'default': 0.0,
        'help': 'The scale of noise offset.'},}

# LORA #########################################################################

LORA_ARGS = {
    **RANDOM_ARGS,
    **OUTPUT_ARGS,
    **MODEL_ARGS,
    **DATASET_ARGS,
    **PREPROCESSING_ARGS,
    **CHECKPOINT_ARGS,
    **VALIDATION_ARGS,
    **ITERATION_ARGS,
    **GRADIENT_ARGS,
    **LEARNING_ARGS,
    **LOSS_ARGS,
    **PRECISION_ARGS,
    **DISTRIBUTION_ARGS,
    **OPTIMIZER_ARGS,
    **FRAMEWORK_ARGS,
    **DIFFUSION_ARGS,}

# VAE ##########################################################################

# PARSER #######################################################################

def create_parser(definitions: dict=LORA_ARGS, description: str='') -> argparse.ArgumentParser:
    __parser = argparse.ArgumentParser(description=description)
    # iterate on the argument definitions (str, dict)
    for __k, __d in definitions.items():
        __parser.add_argument(__k, **__d)
    # parser object
    return __parser
