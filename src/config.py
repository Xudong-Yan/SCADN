import os
import yaml


class Config(dict):
    def __init__(self, config_path):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.load(self._yaml, Loader=yaml.FullLoader)
            self._dict['PATH'] = os.path.dirname(config_path)

    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]
        else:
            return None

    def print(self):
        with open(self.config_path, 'r') as f:
            _yaml = f.read()
            print('Model configurations:')
            print('---------------------------------')
            print(_yaml)
            print('')
            print('---------------------------------')
            print('')

    def append(self, name, value):
        self._dict[str(name)] = value

    def save(self):
        with open(self.config_path, 'w') as f:
            yaml.dump(self._dict, f)

