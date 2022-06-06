from sklearn.pipeline import Pipeline

from config.core import config
from processing import preprocessors as pp
import model


pipe = Pipeline([
                ('dataset', pp.CreateDataset(config.model_config.image_size)),
                ('cnn_model', model.cnn_clf)])
