#!/usr/bin/env python
# Created by "Thieu" at 04:18, 28/09/2023 ----------%                                                                               
#       Email: nguyenthieu2102@gmail.com            %                                                    
#       Github: https://github.com/thieu1995        %                         
# --------------------------------------------------%

import numpy as np



class Agent:
    ID = 0

    def __init__(self, solution: np.ndarray = None, target: float = None, **kwargs) -> None:
        self.solution = solution
        self.fitness = target

