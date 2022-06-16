import numpy as np
from regrasp.utils.transform import Transform
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional
from regrasp.utils.robot import Panda

class Hand(Enum):
    OBJ_IN_LEFT = auto()
    OBJ_IN_RIGHT = auto()

# class Mode(Enum):
#     OBJ_IN_LEFT = auto()
#     OBJ_IN_RIGHT = auto()

@dataclass
class Grasp:
    T: Transform
    index: int = field(default_factory=lambda: -1) #not registered

@dataclass
class Mode:
    hand: Hand
    grasp: Grasp

    def __eq__(self, other):
        assert other.__class__ is self.__class__
        return (self.hand, self.grasp.index) == (other.hand, other.grasp.index)

# @dataclass
# class RobotConfig:
#     l: np.ndarray
#     r: np.ndarray

@dataclass
class Config1:
    """Configuration for single arm regrasping
    """
    q: np.ndarray
    T_obj: Optional[Transform] = field(default_factory=lambda: None)
    grasp: Optional[Grasp] = field(default_factory=lambda: None)
    index: int = field(default_factory=lambda: -1) #not registered

    
@dataclass
class Config:
    """Node for Global planner
    """
    mode: Mode
    q_l: np.ndarray
    q_r: np.ndarray
    T_l: Optional[Transform] = field(default_factory=lambda: None)
    T_r: Optional[Transform] = field(default_factory=lambda: None)
    index: int = field(default_factory=lambda: -1) #not registered

    # grasp_handover: Optional[Grasp] = field(default_factory=lambda: None)
    # config_handover: Optional[Config] = field(default_factory=lambda: None)
    # trajectory: List = field(default_factory=lambda: []) #not registered
    # index: int = field(default_factory=lambda: -1) #not registered

# @dataclass
# class Config:
#     q_l: np.ndarray
#     q_r: np.ndarray
#     T_obj: Transform



@dataclass
class DualArm:
    left: Panda
    right: Panda