import torch
from torch.optim import Adam, AdamW

from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.AvaGrad import AvaGrad


class nnUNetTrainerAdam(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
    def configure_optimizers(self):
        optimizer = AdamW(self.network.parameters(),
                          lr=self.initial_lr,
                          weight_decay=self.weight_decay,
                          amsgrad=True)
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


class nnUNetTrainerVanillaAdam(nnUNetTrainer):
    def configure_optimizers(self):
        optimizer = Adam(self.network.parameters(),
                         lr=self.initial_lr,
                         weight_decay=self.weight_decay)
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler


class nnUNetTrainerVanillaAdam1en3(nnUNetTrainerVanillaAdam):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3


class nnUNetTrainerVanillaAdam3en4(nnUNetTrainerVanillaAdam):
    # https://twitter.com/karpathy/status/801621764144971776?lang=en
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 3e-4


class nnUNetTrainerAdam1en3(nnUNetTrainerAdam):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3


class nnUNetTrainerAdam3en4(nnUNetTrainerAdam):
    # https://twitter.com/karpathy/status/801621764144971776?lang=en
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 3e-4

class nnUNetTrainerAvaGrad(nnUNetTrainer):
    def configure_optimizers(self):
        optimizer = AvaGrad(self.network.parameters(),
                          lr=self.initial_lr,
                          betas=(0.9, 0.999), # This parameter represents the coefficients used for computing running averages of gradient and its squared norm. It is a tuple of two values: betas[0] corresponds to the coefficient for the exponential moving average of the gradient, and betas[1] corresponds to the coefficient for the squared gradient norm. By default, the values are set to (0.9, 0.999).
                          
                          weight_decay=self.weight_decay, # It is a regularization technique that adds a penalty term to the loss function to encourage smaller weights in the model. By default, its value is set to 0, indicating no weight decay.
                          
                          eps=0.1) # This parameter represents a small constant added to the denominator for numerical stability. It prevents division by zero in cases where the denominator could be very small. By default, its value is set to 0.1.
                          
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler

class nnUNetTrainerAvaGrad_500epoch(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
    def configure_optimizers(self):
        optimizer = AvaGrad(self.network.parameters(),
                          lr=1e-1,
                          betas=(0.9, 0.999), # This parameter represents the coefficients used for computing running averages of gradient and its squared norm. It is a tuple of two values: betas[0] corresponds to the coefficient for the exponential moving average of the gradient, and betas[1] corresponds to the coefficient for the squared gradient norm. By default, the values are set to (0.9, 0.999).
                          
                          weight_decay=self.weight_decay, # It is a regularization technique that adds a penalty term to the loss function to encourage smaller weights in the model. By default, its value is set to 0, indicating no weight decay.
                          
                          eps=0.1) # This parameter represents a small constant added to the denominator for numerical stability. It prevents division by zero in cases where the denominator could be very small. By default, its value is set to 0.1.
                          
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, 1e-1, self.num_epochs)
        return optimizer, lr_scheduler

class nnUNetTrainerVanillaAdam_500epoch(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
    def configure_optimizers(self):
        optimizer = Adam(self.network.parameters(),
                         lr=1e-3,
                         weight_decay=self.weight_decay)
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, 1e-1, self.num_epochs)
        return optimizer, lr_scheduler 
        
        
        
class nnUNetTrainerAvaGrad_500epoch_lr2e(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
    def configure_optimizers(self):
        optimizer = AvaGrad(self.network.parameters(),
                          lr=2e-1,
                          betas=(0.9, 0.999), # This parameter represents the coefficients used for computing running averages of gradient and its squared norm. It is a tuple of two values: betas[0] corresponds to the coefficient for the exponential moving average of the gradient, and betas[1] corresponds to the coefficient for the squared gradient norm. By default, the values are set to (0.9, 0.999).
                          
                          weight_decay=self.weight_decay, # It is a regularization technique that adds a penalty term to the loss function to encourage smaller weights in the model. By default, its value is set to 0, indicating no weight decay.
                          
                          eps=0.1) # This parameter represents a small constant added to the denominator for numerical stability. It prevents division by zero in cases where the denominator could be very small. By default, its value is set to 0.1.
                          
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, 1e-1, self.num_epochs)
        return optimizer, lr_scheduler
        
        
class nnUNetTrainerAvaGrad_500epoch_lr1(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
    def configure_optimizers(self):
        optimizer = AvaGrad(self.network.parameters(),
                          lr=1,
                          betas=(0.9, 0.999), # This parameter represents the coefficients used for computing running averages of gradient and its squared norm. It is a tuple of two values: betas[0] corresponds to the coefficient for the exponential moving average of the gradient, and betas[1] corresponds to the coefficient for the squared gradient norm. By default, the values are set to (0.9, 0.999).
                          
                          weight_decay=self.weight_decay, # It is a regularization technique that adds a penalty term to the loss function to encourage smaller weights in the model. By default, its value is set to 0, indicating no weight decay.
                          
                          eps=0.1) # This parameter represents a small constant added to the denominator for numerical stability. It prevents division by zero in cases where the denominator could be very small. By default, its value is set to 0.1.
                          
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, 1e-1, self.num_epochs)
        return optimizer, lr_scheduler   
        
        
class nnUNetTrainerAdam3en4_500(nnUNetTrainerAdam):
    # https://twitter.com/karpathy/status/801621764144971776?lang=en
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 3e-4   
        
        
class nnUNetTrainerAdam1en3_500(nnUNetTrainerAdam):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-3  
        
        

class nnUNetTrainerAvaGrad_1000epoch(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 1000
    def configure_optimizers(self):
        optimizer = AvaGrad(self.network.parameters(),
                          lr=1e-1,
                          betas=(0.9, 0.999), # This parameter represents the coefficients used for computing running averages of gradient and its squared norm. It is a tuple of two values: betas[0] corresponds to the coefficient for the exponential moving average of the gradient, and betas[1] corresponds to the coefficient for the squared gradient norm. By default, the values are set to (0.9, 0.999).
                          
                          weight_decay=self.weight_decay, # It is a regularization technique that adds a penalty term to the loss function to encourage smaller weights in the model. By default, its value is set to 0, indicating no weight decay.
                          
                          eps=0.1) # This parameter represents a small constant added to the denominator for numerical stability. It prevents division by zero in cases where the denominator could be very small. By default, its value is set to 0.1.
                          
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, 1e-1, self.num_epochs)
        return optimizer, lr_scheduler  
        
class nnUNetTrainerAdamw(nnUNetTrainer):
    def configure_optimizers(self):
        optimizer = AdamW(self.network.parameters(),
                          lr=self.initial_lr,
                          weight_decay=self.weight_decay,
                          amsgrad=True)
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, self.initial_lr, self.num_epochs)
        return optimizer, lr_scheduler
        
class nnUNetTrainerAdamw_1en4(nnUNetTrainerAdamw):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-4  
        
class nnUNetTrainerAdamw_500_125_1_1en4(nnUNetTrainerAdamw):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-4
        self.num_epochs = 1000
        self.oversample_foreground_percent = 1.00
        self.num_iterations_per_epoch = 125
        
class nnUNetTrainerAdamw_500_500_80_1en4(nnUNetTrainerAdamw):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-4
        self.num_epochs = 500
        self.oversample_foreground_percent = 0.80
        self.num_iterations_per_epoch = 500        
        
class nnUNetTrainerAdamw_500_250_1en4(nnUNetTrainerAdamw):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = 1e-4
        self.num_epochs = 500
        self.oversample_foreground_percent = 0.80

        
class nnUNetTrainerSGD_500_125_1_1en4(nnUNetTrainerAdamw):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 500
        self.oversample_foreground_percent = 1.00
        self.num_iterations_per_epoch = 125 
        
class nnUNetTrainerAvaGrad_150epoch(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 150
    def configure_optimizers(self):
        optimizer = AvaGrad(self.network.parameters(),
                          lr=1,
                          betas=(0.9, 0.999), # This parameter represents the coefficients used for computing running averages of gradient and its squared norm. It is a tuple of two values: betas[0] corresponds to the coefficient for the exponential moving average of the gradient, and betas[1] corresponds to the coefficient for the squared gradient norm. By default, the values are set to (0.9, 0.999).
                          
                          weight_decay=self.weight_decay, # It is a regularization technique that adds a penalty term to the loss function to encourage smaller weights in the model. By default, its value is set to 0, indicating no weight decay.
                          
                          eps=0.1) # This parameter represents a small constant added to the denominator for numerical stability. It prevents division by zero in cases where the denominator could be very small. By default, its value is set to 0.1.
                          
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, 1, self.num_epochs)
        return optimizer, lr_scheduler
      
class nnUNetTrainerAvaGrad_200epoch(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True,
                 device: torch.device = torch.device('cuda')):
        """used for debugging plans etc"""
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.num_epochs = 200
    def configure_optimizers(self):
        optimizer = AvaGrad(self.network.parameters(),
                          lr=1,
                          betas=(0.9, 0.999), # This parameter represents the coefficients used for computing running averages of gradient and its squared norm. It is a tuple of two values: betas[0] corresponds to the coefficient for the exponential moving average of the gradient, and betas[1] corresponds to the coefficient for the squared gradient norm. By default, the values are set to (0.9, 0.999).
                          
                          weight_decay=self.weight_decay, # It is a regularization technique that adds a penalty term to the loss function to encourage smaller weights in the model. By default, its value is set to 0, indicating no weight decay.
                          
                          eps=0.1) # This parameter represents a small constant added to the denominator for numerical stability. It prevents division by zero in cases where the denominator could be very small. By default, its value is set to 0.1.
                          
        # optimizer = torch.optim.SGD(self.network.parameters(), self.initial_lr, weight_decay=self.weight_decay,
        #                             momentum=0.99, nesterov=True)
        lr_scheduler = PolyLRScheduler(optimizer, 1, self.num_epochs)
        return optimizer, lr_scheduler      

             
                                                              
