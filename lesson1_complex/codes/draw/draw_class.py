from codes.draw.TSNE import tsne
from codes.draw.Confusion_matrix import draw_confusion_matrix
import inspect
class Draw():
    def __init__(self, args,logger, sub, model, dataloader, save_path):
        self.args = args
        self.classes_dict = {
            0: 'left_hand',
            1: 'right_hand',
            2: 'foot',
            3: 'tongue'
        }

        self.sub = sub
        self.model = model
        self.dataloader = dataloader
        self.save_path = save_path
        self.logger = logger
    def run(self):
        self.all_func = self._get_callable_methods()
        """
        Run the draw function
        :return:
        """
        # args.draw_list = [confusion_matrix, tsne]
        for draw_func in self.args.draw_list:
            if draw_func not in self.all_func:
                raise ValueError(f"Unknown draw function: {draw_func}")
            else:
                func = getattr(self, draw_func)
                func()
                self.logger.info(f"{draw_func} Done!")

    def _get_callable_methods(self):
        methods = []
        for name, member in inspect.getmembers(self):
            if inspect.ismethod(member) or inspect.isfunction(member):
                if not name.startswith('__'):
                    methods.append(name)
        return methods
    def tsne(self):
        """
        t-SNE visualization
        :return:
        """
        layer_name = list(dict([*self.model.named_modules()]))[-2]
        tsne_save_path = self.save_path + '/tsne'
        tsne(self.model, layer_name, self.classes_dict, self.sub, self.dataloader, tsne_save_path)
        self.logger.info(f"Layer name: {layer_name}, t-SNE visualization done!")

    def confusion_matrix(self):
        """
        Confusion matrix visualization
        :return:
        """
        confusion_matrix_save_path = self.save_path + '/confusion_matrix'
        draw_confusion_matrix(self.model, self.classes_dict, self.sub, self.dataloader, confusion_matrix_save_path)


