import pandas as pd
import logging
from starter.ml.data import process_data

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def test_process_data(data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        dframe = process_data(data, categorical_features=cat_features, label='salary', training=True, encoder=None,
                              lb=None)
        assert dframe.shape[0] > 0
        assert dframe.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("ERROR Testing import_data: The file wasn't found: %s", err)
        raise err
    except AssertionError as err:
        logging.error("ERROR Testing import_data: empy file %s", err)
        raise err

    return None

