from codes.paradigm.dataset_2a_paradigm import Dataset_2a_Paradigm
from codes.models.ModelResgistry import MODEL_REGISTOR
import torch
import collections
import torch.nn as nn
import wandb
from utils import init_wandb_config
import os
import numpy as np
import codes.models.ModelResgistry
import codes.models.EEG_conformer # register the model
class Trainer():
    def __init__(self, data, label, args, logger):
        """
        Initialize the Trainer class.
        :param data: EEG data
        :param label: EEG labels
        :param args: Arguments for training
        :param logger: Logger for logging information
        """
        self.data = data
        self.label = label
        self.args = args
        self.logger = logger
        self.train_dataset, self.val_dataset, self.test_dataset = None, None, None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.result_dict = None

    def get_dataset(self, sub):
        self.logger.info(f"Loading {self.args.dataset_name} dataset...")
        if self.args.dataset_name == 'BCICIV2A':
            self.train_dataset, self.val_dataset, self.test_dataset = self.dataset.get_dataset_from_2a(sub)
        else:
            raise NotImplementedError(f"Unknown dataset: {self.args.dataset_name}")
        self.logger.info(f"Finished loading {self.args.dataset_name} dataset.")
    def init_dataset(self):
        if self.args.dataset_name == 'BCICIV2A':
            self.dataset = Dataset_2a_Paradigm(self.data, self.label, self.args)
        else:
            raise NotImplementedError(f"Unknown dataset: {self.args.dataset_name}")
    def get_available_model(self):
        """
        Get the available models for the specified paradigm
        :return:
        """
        return MODEL_REGISTOR.registered_names()

    def run(self):
        """
        Run the training process.
        """
        if self.args.paradigm == 'cross_session':
            self.__cross_session()
        elif self.args.paradigm == 'cross_subject':
            self.__cross_subject()
        else:
            raise ValueError(f"Unknown paradigm: {self.args.paradigm}")

    def __cross_subject(self):
        pass
    def __cross_session(self):
        """
        Cross-session for EEG data
        """
        models_name = self.get_available_model()
        self.logger.info(f"Available models: {models_name}")
        self.init_dataset()
        for model_name in models_name:
            if self.args.is_choose_best_hyper:
                # Initialize sweep configuration
                sweep_config = self.init_sweep_config()

                # Create the sweep
                sweep_id = wandb.sweep(sweep_config, project='EEG_hyperparameter_optimization')

                # Run the sweep
                self.logger.info(f"Starting hyperparameter sweep for model: {model_name}")
                wandb.agent(sweep_id, lambda: self.sweep_train(model_name), count=20)

                # Fetch best hyperparameters
                api = wandb.Api()
                sweep = api.sweep(f"{wandb.run.entity}/EEG_hyperparameter_optimization/{sweep_id}")
                best_run = sweep.best_run()

                # Update args with best hyperparameters
                self.args.learning_rate = best_run.config['learning_rate']
                self.args.batch_size = best_run.config['batch_size']
                self.args.optimizer = best_run.config['optimizer']

                self.logger.info(f"Best hyperparameters found: {best_run.config}")

            for sub in range(self.dataset.num_subject):
                wandb_logger = wandb.init(project='lesson_1_complex', name='Cross_session_' + model_name +'_sub_' + str(sub),
                                          resume='allow')
                wandb_logger = init_wandb_config(wandb_logger, args=self.args)
                self.wandb_logger = wandb_logger
                self.cur_sub = sub
                self.logger.info(f"Training model: {model_name} for subject: {sub}")
                # Split data for this subject into train, val and test sets
                self.get_dataset(sub)

                self.init_model(model_name)
                # watch the model
                self.wandb_logger.watch(self.model, log='all')
                # Train the model
                self.__train()

                # Test the model
                test_acc = self.__test()
                # Log the results
                self.wandb_logger.log({"Test_acc": test_acc})
                self.wandb_logger.finish()
                # Save the model
                save_dirs = os.path.join(self.args.save_models_dir, self.args.paradigm, self.args.model, str(sub))
                os.makedirs(save_dirs, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(save_dirs, 'final.pth'))
                self.logger.info(f'Model: {model_name}, Subject: {sub}, Test Accuracy: {test_acc:.2f}%')

    def __train(self):
        """
        Train the model.
        """
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        # Set the model to training mode
        self.model.train()
        for epoch in range(self.args.epochs):
            batch_loss = 0.0
            batch_acc = 0.0

            for batch_idx, (data, label) in enumerate(train_loader):
                # Move data to the device
                data, label = data.to(self.device), label.to(self.device)
                # Zero the parameter gradients
                self.optimizer.zero_grad()
                # Forward pass
                output = self.model(data)
                # Compute the loss
                loss = self.criterion(output, label)
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()

                batch_loss += loss.item()
                # Compute the accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(label.view_as(pred)).sum().item()
                batch_acc += correct / len(label)
            # Compute the average loss and accuracy for the epoch
            epoch_loss = batch_loss / len(train_loader)
            epoch_acc = batch_acc / len(train_loader)

            self.result_dict['Train_loss'].append(epoch_loss)
            self.result_dict['Train_acc'].append(epoch_acc)
            # Log the results
            self.wandb_logger.log({"Train_loss": epoch_loss, "Train_acc": epoch_acc})
            self.logger.info(f"Sub: {self.cur_sub}, Epoch: {epoch + 1}/{self.args.epochs}, Train Loss: {epoch_loss}, Train Accuracy: {epoch_acc}")
            # Validate the model
            if epoch % self.args.val_interval == 0:
                val_loss, val_acc = self.__val()

                # Save the model
                if self.args.save_model:
                    save_dirs = os.path.join(self.args.save_models_dir,self.args.paradigm, self.args.model, str(self.cur_sub))
                    os.makedirs(save_dirs, exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(save_dirs, 'epoch_' + str(epoch) + '.pth'))
                    self.logger.info(f"Model for subject {self.cur_sub} saved")

    def __val(self):
        """
        Validate the model.
        """
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False)
        self.model.eval()
        val_loss = 0.0
        val_acc = 0.0
        with torch.no_grad():
            for data, label in val_loader:
                # Move data to the device
                data, label = data.to(self.device), label.to(self.device)
                # Forward pass
                output = self.model(data)
                # Compute the loss
                loss = self.criterion(output, label)
                val_loss += loss.item()
                # Compute the accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(label.view_as(pred)).sum().item()
                val_acc += correct / len(label)

        epoch_loss = val_loss / len(val_loader)
        epoch_acc = val_acc / len(val_loader)
        self.result_dict['Val_loss'].append(epoch_loss)
        self.result_dict['Val_acc'].append(epoch_acc)
        # Log the results
        self.wandb_logger.log({"Val_loss": epoch_loss, "Val_acc": epoch_acc})
        self.logger.info(f"Sub: {self.cur_sub}, Val Loss: {epoch_loss}, Val Accuracy: {epoch_acc}")
        return epoch_loss, epoch_acc

    def __test(self):
        """
        Test the model.
        """
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=False)
        self.model.eval()
        test_loss = 0.0
        test_acc = 0.0
        with torch.no_grad():
            for data, label in test_loader:
                # Move data to the device
                data, label = data.to(self.device), label.to(self.device)
                # Forward pass
                output = self.model(data)
                # Compute the loss
                loss = self.criterion(output, label)
                test_loss += loss.item()
                # Compute the accuracy
                pred = output.argmax(dim=1, keepdim=True)
                correct = pred.eq(label.view_as(pred)).sum().item()
                test_acc += correct / len(label)
        epoch_loss = test_loss / len(test_loader)
        epoch_acc = test_acc / len(test_loader)

        self.result_dict['Test_loss'].append(epoch_loss)
        self.result_dict['Test_acc'].append(epoch_acc)
        # Log the results
        self.wandb_logger.log({"Test_loss": epoch_loss, "Test_acc": epoch_acc})
        return epoch_acc

    def init_model(self, model_name):
        self.model_init, self.criterion = MODEL_REGISTOR.get(model_name)
        self.model = self.model_init(input_shape = self.dataset.input_shape, output_shape = self.dataset.output_shape)
        if self.args.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        elif self.args.optimizer == 'SGD':

            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.learning_rate, momentum=0.9)
        elif self.args.optimizer == 'AdamW':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        else:
            raise NotImplementedError(f"Unknown optimizer: {self.args.optimizer}")
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                                    mode='min', factor=0.1, patience=8,
                                                                    verbose=False, min_lr=1e-6)
        self.result_dict = collections.OrderedDict({'Train_loss': [], 'Val_loss': [], 'Test_loss': [],
                                                    'Train_acc': [], 'Val_acc': [], 'Test_acc': []})
        if self.args.use_gpu:
            if not torch.cuda.is_available():
                raise ValueError("GPU is not available. Please check your CUDA installation.")
            self.device = torch.device(f'cuda:{self.args.gpu_id}')
        else:
            self.device = torch.device('cpu')
        self.model.to(self.device)
        self.criterion.to(self.device)
        self.initialize_parameters()
    def initialize_parameters(self):
        def weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.model.apply(weight_init)
        self.optimizer.state = collections.defaultdict(dict)

    def init_sweep_config(self):
        """Define the hyperparameter search space"""
        sweep_config = {
            'method': 'bayes',  # Bayesian optimization
            'metric': {
                'name': 'mean_val_loss',
                'goal': 'minimize'
            },
            'parameters': {
                'learning_rate': {
                    'distribution': 'log_uniform',
                    'min': 1e-6,
                    'max': 1e-1
                },
                'batch_size': {
                    'values': [32, 64]
                },
                'optimizer': {
                    'values': ['Adam', 'SGD']
                }
                # Add more parameters as needed
            }
        }
        return sweep_config

    def sweep_train(self, model_name):
        """Training function for sweep agent"""
        # Initialize wandb with sweep config
        with wandb.init() as run:
            self.wandb_logger = run
            # Get hyperparameters from wandb
            config = wandb.config

            # Update args with sweep parameters
            self.args.learning_rate = config.learning_rate
            self.args.batch_size = config.batch_size
            self.args.optimizer = config.optimizer

            # Track validation losses across subjects
            val_losses = []

            # Train for each subject
            for sub in range(self.dataset.num_subject):

                self.cur_sub = sub
                self.logger.info(f"Sweep trial for model: {model_name} subject: {sub}")

                # Get dataset for this subject
                self.get_dataset(sub)

                # Initialize model with current hyperparameters
                self.init_model(model_name)

                # Watch model
                self.wandb_logger.watch(self.model, log='all')

                # Train model for fewer epochs during sweep
                orig_epochs = self.args.epochs
                self.args.epochs = min(5, orig_epochs)  # Reduced epochs for faster sweeping

                # Train the model
                self.__train()

                # Validate model
                val_loss, val_acc = self.__val()
                val_losses.append(val_loss)

                # Restore original epochs
                self.args.epochs = orig_epochs

            # Calculate mean validation loss across all subjects
            mean_val_loss = np.mean(val_losses)

            # Log the mean loss (this is what wandb optimizes)
            wandb.log({"mean_val_loss": mean_val_loss})
