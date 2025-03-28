from codes.paradigm.dataset_2a_paradigm import Dataset_2a_Paradigm
from codes.models.ModelResgistry import MODEL_REGISTOR
import torch
import collections
import torch.nn as nn
import wandb
from utils import init_wandb_config, YamlHandler, get_cur_time, calMetrics
import os
import numpy as np
import codes.models.ModelResgistry
# import codes.models.EEG_conformer # register the model
import codes.models.EEG_conformer_CTN
# from torchinfo import summary
from torchsummary import summary
from codes.draw.draw_class import Draw
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
            self.model_name = model_name
            if self.args.is_choose_best_hyper:

                # Initialize sweep configuration
                sweep_config = self.init_sweep_config()

                # Create the sweep
                sweep_id = wandb.sweep(sweep_config,  entity='norman123', project='EEG_lesson1_complex')

                # Run the sweep
                self.logger.info(f"Starting hyperparameter sweep for model: {model_name}")
                wandb.agent(sweep_id, lambda: self.sweep_train(model_name), count=20, entity='norman123', project='EEG_lesson1_complex')

                # Fetch best hyperparameters
                api = wandb.Api()
                sweep = api.sweep(f"{wandb.run.entity}/EEG_lesson1_complex/{sweep_id}")
                best_run = sweep.best_run()

                # Update args with best hyperparameters
                self.args.learning_rate = best_run.config['learning_rate']
                self.args.batch_size = best_run.config['batch_size']
                self.args.optimizer = best_run.config['optimizer']
                self.args.epochs = self.model_config[self.args.paradigm]['num_epochs']
                self.logger.info(f"Best hyperparameters found: {best_run.config}")

            for sub in range(self.dataset.num_subject):
                self.init_model(self.model_name)
                if not self.args.is_choose_best_hyper:
                    self.args.learning_rate = self.model_config[self.args.paradigm]['learning_rate']
                    self.args.batch_size = self.model_config[self.args.paradigm]['batch_size']
                    self.args.optimizer = self.model_config[self.args.paradigm]['optimizer']
                    self.args.epochs = self.model_config[self.args.paradigm]['num_epochs']
                wandb_dir = os.path.join(self.args.save_models_dir, 'wandb', self.args.paradigm,self.model_name, str(sub))
                os.makedirs(wandb_dir, exist_ok=True)

                wandb_logger = wandb.init(entity='norman123', project='EEG_lesson1_complex',
                                          dir=str(wandb_dir),
                                          name='Cross_session_' + self.model_name +'_sub_' + str(sub) + '_' + get_cur_time(),
                                          resume='allow')
                wandb_logger = init_wandb_config(wandb_logger, args=self.args, model_name=self.model_name)
                self.wandb_logger = wandb_logger
                self.cur_sub = sub
                self.logger.info(f"Training model: {self.model_name} for subject: {sub}")
                # Split data for this subject into train, val and test sets
                self.get_dataset(sub)

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
                save_dirs = os.path.join(self.args.save_models_dir, self.args.paradigm, self.model_name, str(sub))
                os.makedirs(save_dirs, exist_ok=True)
                torch.save(self.model.state_dict(), os.path.join(save_dirs, 'final.pth'))
                self.logger.info(f'Model: {model_name}, Subject: {sub}, Test Accuracy: {test_acc:.2f}%')

    def __train(self):
        """
        Train the model.
        """
        if self.model_config[self.args.paradigm]['use_validation']:
            self.logger.info(f"Training with validation set")
            train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        else:
            self.logger.info(f"Training without validation set") # used in EEG_CONFORMER
            # 合并训练集和验证集
            train_data = np.concatenate([self.train_dataset.data.cpu().numpy(), self.val_dataset.data.cpu().numpy()], axis=0)
            train_label = np.concatenate([self.train_dataset.labels.cpu().numpy(), self.val_dataset.labels.cpu().numpy()], axis=0)

            # Create a dataset from the combined data
            from codes.paradigm.dataset_2a_paradigm import EEG_org_dataset  # Import the dataset class
            train_dataset = EEG_org_dataset(train_data, train_label, self.args)
            # Create the loader for the combined dataset
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
        # Set the model to training mode
        self.model.train()
        min_val_loss = 1e10
        save_dirs = os.path.join(self.args.save_models_dir, self.args.paradigm, self.model_name,
                                 str(self.cur_sub))
        os.makedirs(save_dirs, exist_ok=True)
        for epoch in range(self.model_config[self.args.paradigm]['num_epochs']):
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
            self.logger.info(f"Sub: {self.cur_sub}, Epoch: {epoch + 1}/{self.model_config[self.args.paradigm]['num_epochs']}, Train Loss: {epoch_loss}, Train Accuracy: {epoch_acc}")
            # Validate the model if using validation set


            if epoch % self.model_config[self.args.paradigm]['val_interval'] == 0 and self.model_config[self.args.paradigm]['use_validation']:
                val_loss, val_acc = self.__val()
                if val_loss < min_val_loss:
                    min_val_loss = min(min_val_loss, val_loss)
                    save_model_state_dict = self.model.state_dict()
                    torch.save(save_model_state_dict, os.path.join(save_dirs, 'best.pth'))

                # Save the model
                if self.args.save_model:
                    torch.save(self.model.state_dict(), os.path.join(save_dirs, 'epoch_' + str(epoch) + '.pth'))
                    self.logger.info(f"Model for subject {self.cur_sub} saved")
        if not self.model_config[self.args.paradigm]['use_validation']:
            torch.save(self.model.state_dict(), os.path.join(save_dirs, 'train_without_val.pth'))
            self.logger.info(f"Model for subject {self.cur_sub} saved")


    def __val(self):
        """
        Validate the model.
        """
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=False)
        # load the best model

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
        test_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.batch_size, shuffle=True)
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
        save_figure_dir = os.path.join(self.args.save_results_dir, 'figures', self.args.paradigm, self.model_name, str(self.cur_sub))
        draw_func = Draw(self.args, self.logger, self.cur_sub, self.model, test_loader, save_figure_dir)
        draw_func.run()
        return epoch_acc

    def init_model(self, model_name):
        self.model_init, self.criterion = MODEL_REGISTOR.get(model_name)
        self.model = self.model_init(input_shape = self.dataset.input_shape, output_shape = self.dataset.output_shape)
        self.model_config = YamlHandler(os.path.join(self.args.config_dir, model_name + '.yaml')).read_yaml()

        if self.model_config[self.args.paradigm]['optimizer'].lower() == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.model_config[self.args.paradigm]['learning_rate'], betas=(0.5, 0.999))
        elif self.model_config[self.args.paradigm]['optimizer'].lower() == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.model_config[self.args.paradigm]['learning_rate'], momentum=0.9)
        elif self.model_config[self.args.paradigm]['optimizer'].lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.model_config[self.args.paradigm]['learning_rate'])
        else:
            raise NotImplementedError(f"Unknown optimizer: {self.model_config[self.args.paradigm]['optimizer']}")
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
        input_size = (1, self.dataset.input_shape[0], self.dataset.input_shape[1])
        model_stats = summary(self.model, input_size)
        self.logger.info(f"Model: {model_name}, Model stats: {model_stats}")
        # self.logger.info(f"Model: {model_name}, Model params: {sum(p.numel() for p in self.model.parameters())}")
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
            'name': 'EEG_hyperparameter_optimization',
            'method': 'bayes',  # Bayesian optimization
            'metric': {
                'name': 'mean_val_loss', # The metric to optimize
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
        # Initialize wandb with sweep configs
        wandb_dir = os.path.join(self.args.save_models_dir, 'wandb', self.args.paradigm, self.model_name, 'wandb_sweep')
        os.makedirs(wandb_dir, exist_ok=True)
        with wandb.init( dir=str(wandb_dir), name=f'sweep_{model_name}') as run:
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

                # Train the model
                self.__train()

                # Validate model
                if not self.model_config[self.args.paradigm]['use_validation']:
                    raise ValueError("Validation set is not used. Please check your configuration.")
                val_loss, val_acc = self.__val()
                # Log validation loss
                val_losses.append(val_loss)


            # Calculate mean validation loss across all subjects
            mean_val_loss = np.mean(val_losses)

            # Log the mean loss (this is what wandb optimizes)
            wandb.log({"mean_val_loss": mean_val_loss})
