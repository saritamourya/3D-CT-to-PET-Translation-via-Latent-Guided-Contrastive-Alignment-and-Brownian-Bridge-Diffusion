
class ModelConfig:
    def __init__(self):
    

        # List of districts with checkpoint paths
        self.district_checkpoints = {
            'brain': "./pretrained/checkpointssharedlatent_100petresidual/brain/autoencoder_epoch299.pth",
            'lung': "./pretrained/checkpointssharedlatent_100petresidual/lung/autoencoder_epoch299.pth",
            'liver': "./pretrained/checkpointssharedlatent_100petresidual/liver/autoencoder_epoch299.pth",
            'stomach': "./pretrained/checkpointssharedlatent_100petresidual/stomach/autoencoder_epoch299.pth",
            'kidney': "./pretrained/checkpointssharedlatent_100petresidual/kidney/autoencoder_epoch299.pth",
        }
        # self.district_diffusion_checkpoints={
        #     'brain': './BrownianBridgePretainCheckpoints/checkpointswithoutsharedlatent_gn32pet2pet/brain/ct2pet_diffusion_model_epoch299.pth',
        #     'lung': './BrownianBridgePretainCheckpoints/checkpointswithoutsharedlatent_gn32pet2pet/lung/ct2pet_diffusion_model_epoch299.pth',
        #     'liver': './BrownianBridgePretainCheckpoints/checkpointswithoutsharedlatent_gn32pet2pet/liver/ct2pet_diffusion_model_epoch299.pth',
        #     'kidney': './BrownianBridgePretainCheckpoints/checkpointswithoutsharedlatent_gn32pet2pet/kidney/ct2pet_diffusion_model_epoch299.pth',
        #     'stomach': './BrownianBridgePretainCheckpoints/checkpointswithoutsharedlatent_gn32pet2pet/stomach/ct2pet_diffusion_model_epoch299.pth'
        # }


        # Model hyperparameters
        self.BB = self.BrownianBridgeConfig()
        self.district_name = "lung"  # Default district for training
        self.dataroot= '/mimer/NOBACKUP/groups/naiss2023-6-336/dataset_shared/ct2pet/radiogenomics'
        self.phase = "train"  
        self.test_district = "lung"
        self.district = "lung"  # Default district for training
        self.batchSize = 32 #max(2, torch.cuda.device_count())  # Adjust batch size dynamically
        self.image_size = 32  # Image dimensions (D, H, W)
        self.in_channels = 1      # Number of input channels
        self.results_dir = 'ResultsDiffusionModel/sharedlatent_gn32_20residual_/'
        # Optimizer settings
        self.lr = 0.0001  
        self.n_epochs =  500
        self.val_interval = 10  # Reduce validation overhead
        self.model_name = "CPDM"
        self.model_type = "CPDM"  # Model type (e.g., 'unet', 'resnet')
        self.ema_decay = 0.995  # Exponential moving average decay
        self.start_ema_step = 30000  # EMA step size
        self.ema_update = 8  # EMA update rate
        self.use_ema = True  # Use EMA for model updates

        # Mixed Precision
        self.amp = True  

        # Loss Weights
        self.perceptual_weight = 0.3  
        self.kl_weight = 1e-7  
        self.adv_weight = 0.1 
        # for loading the checkpoints
        self.district_path = self.district_checkpoints[self.district_name]  # Path to the district checkpoint
        # Gradient Accumulation
        self.gradient_accumulation_steps = 2 if self.batchSize < 2 else 1  # Fix tuple issue
        
        # Set the paths for saving/loading based on district
        self.checkpoint_path = f"BrownianBridgeCheckpoints/100_petshared___latent_NewUnet/{self.district}"
        self.log_dir = f"BrownianBridgeCheckpoints/logs/{self.district}"
        self.model_save_path = f"BrownianBridgeCheckpoints/model/{self.district}"
        self.model_name = "CT2PETDiffusion_Model"  # Model name for saving and checkpointing


        # Hyperparameters for training (could be adjusted based on your setup)
        self.training_params = {
            'batch_size': 64,  # Batch size for training
            'learning_rate': 1e-4,  # Learning rate for the optimizer
            'epochs': 500,  # Number of epochs for training
            'optimizer': 'adam',  # Optimizer to use (Adam, SGD, etc.)
            'lr_scheduler': 'plateau',  # Learning rate scheduler (could also be 'linear', etc.)
            'weight_decay':0.0,  # Weight decay for regularization
            'beta1': 0.9,  # Adam optimizer beta1 parameter
            'patience': 3000,  # Early stopping patience

        }  # Removed the trailing comma

    class BrownianBridgeConfig:
        def __init__(self):
            # Diffusion parameters
            self.params = self.BBParams()

        class BBParams:
            def __init__(self):
                self.num_timesteps = 1000  # Total number of diffusion steps
                self.mt_type = "linear"  # Type of schedule for noise mixing
                self.max_var = 1.0  # Maximum variance for noise
                self.eta = 1.0  # Value for noise scaling
                self.skip_sample = False  # Whether to skip sample steps
                self.sample_type = "linear"  # Sampling strategy
                self.sample_step = 200  # Number of sampling steps
                self.loss_type = "l1"  # Loss type for training (e.g., 'l1', 'l2')
                self.objective = "grad"  # Objective for the model (e.g., 'grad', 'noise', 'ysubx')
                self.skip_sample=True  # Whether to skip sample steps
                # UNet parameters (for 3D UNet)
                self.UNetParams = self.UNetConfig()

            class UNetConfig:
                def __init__(self):
                    self.image_size = 8  # Image dimensions (D, H, W)
                    self.in_channels = 3  # Number of input channels
                    self.condition_key = "nocond"  # Type of conditioning model
                    self.model_channels = 128 # Number of output channels
                    self.out_channels = 3
                    self.num_res_blocks = 2  # Number of residual blocks in the UNet
                    self.attention_resolutions = [32, 16, 8]  # Resolutions for attention
                    self.num_head_channels = 64  # Number of channels per head in attention
                    self.num_heads= 8
                    self.use_scale_shift_norm = False  # Whether to use scale-shift normalization
                    self.use_spatial_transformer = False  # Whether to use spatial transformer
