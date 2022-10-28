
from logging import getLogger, Formatter, StreamHandler, FileHandler, INFO

class BaseOptim():
    def __init__(self, optim_name=__name__):
        self.__logger = None
        self.__logfile_basename = optim_name
        self.__init_logger()
        self.database = {}
        self.current_iter = None
    
    #-- Logger handling
    @property
    def logger(self):
        """ Accessor on logger member variable
        """
        return self.__logger

    def __init_logger(self):
        """ Methods that initialize logging for the whole class

        Log are streamed  in two way:
        - on the standard output
        - into a log file
        """

        self.__logger = getLogger(__name__)

        # create console handler and set level to debug
        handler = StreamHandler()

        # create formatter
        formatter = Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # add formatter to ch
        handler.setFormatter(formatter)
        self.__logger.addHandler(handler)

        handler = FileHandler(filename=f'{self.__logfile_basename}.log',
                              mode='w', encoding='utf-8')
        handler.setFormatter(formatter)
        self.__logger.addHandler(handler)

        self.__logger.setLevel(INFO)
        self.__logger.info('Logger initialized')
        
    def _log_iteration_vector(self, xk):
        """ Callback method attach to fmin_l_bgfs that capture each vector use during optimization

        :params: xk, current optimization vector
        :type: list
        """
        self.current_iter += 1
        msg = "ite " + str(self.current_iter) + \
            " x_%i " % self.current_iter + str(xk)
        if tuple(xk) in self.database:
            inputs = self.database[tuple(xk)]
            for key, value in inputs.items():
                msg += ' ' + key + ' ' + str(value)
        self.__logger.info(msg)
        
    #-- method used in optimization
    def eval_all(self, x):
        """
        Base eval all 
        """
        pass