# FROM MACTN
# CODE https://github.com/ThreePoundUniverse/MACTN
# license None
import torch
import torch.nn as nn
class ModelResgistry_class:
    '''
    A registry class for storing and retrieving model objects.
    This class allows for dynamic registration and retrieval of model classes
    using their names.

    '''

    def __init__(self, name):
        self._name = name
        self._obj_map = dict()
    def _do_register(self, name, obj, loss_fn):
        """
        Register an object with a given name.

        Args:
            name (str): The name to register the object under.
            obj (object): The object to register.
            loss_fn (object): The loss function associated with the object.

        Raises:
            KeyError: If the name is already registered.
        """
        if name in self._obj_map:
            raise KeyError(
                'An object named "{}" was already '
                'registered in "{}" registry'.format(name, self._name)
            )
        self._obj_map[name] = {}
        self._obj_map[name]['Model'] = obj
        self._obj_map[name]['Loss_Fn'] = loss_fn


    def register(self, loss_fn, obj=None):
        """
        Register an object with the registry.

        :param obj:
        :return:
        """
        if obj is None:
            # Used as a decorator
            def wrapper(fn_or_class):
                name = fn_or_class.__name__
                # print(name)
                self._do_register(name, fn_or_class, loss_fn)

                return fn_or_class

            return wrapper

        # Used as a function call
        name = obj.__name__
        self._do_register(name, obj, loss_fn)
    def get(self, name):
        """
        Retrieve an object by its name.
        :param name:
        :return:
        """
        if name not in self._obj_map:
            raise KeyError(
                'Object name "{}" does not exist '
                'in "{}" registry'.format(name, self._name)
            )
        model = self._obj_map[name]['Model']
        loss_fn = self._obj_map[name]['Loss_Fn']
        return model , loss_fn

    def registered_names(self):
        """
        Get the names of all registered objects.
        :return:
        """
        return list(self._obj_map.keys())

MODEL_REGISTOR = ModelResgistry_class('Models') # Model registry class for storing available models

def get_model(model_name, input_shape, output_shape, *args, **kwargs):
    if model_name in MODEL_REGISTOR.registered_names():
        return MODEL_REGISTOR.get(model_name)(input_shape, output_shape, *args, **kwargs)
    else:
        raise NotImplementedError

class BaseModel(nn.Module):
    def __init__(self, input_shape=None, output_shape=None):
        super(BaseModel, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape

    def __build_pseudo_input(self, input_shape=None):
        if input_shape is None:
            input_shape = self.input_shape
        temp_x_ = torch.rand(input_shape)
        temp_x = temp_x_.unsqueeze(0)
        return temp_x

    def get_tensor_shape(self, forward_func, input_shape=None):
        pseudo_x = self.__build_pseudo_input(input_shape)
        pseudo_y = forward_func(pseudo_x)
        return pseudo_y.shape
