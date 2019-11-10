# -*- coding: utf-8 -*-
"""Model module

This module is the definition of CNN model.
"""
import torch
from torch import nn
import torch.nn.functional as F

class Flatten(nn.Module):
    """Flatten Module(layer).
    
       This model flatten input to (batch size, -1)
    """
    def forward(self, input):
        return input.view(input.size(0), -1)

class CNN(nn.Module):
    """CNN modle.
    
       Refernce https://github.com/JasonLiTW/simple-railway-captcha-solver 
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.hidden1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1)
            ,nn.ReLU()
            ,nn.Conv2d(32, 32, 3)
            ,nn.ReLU()
            ,nn.BatchNorm2d(32)
            ,nn.MaxPool2d(2)
            ,nn.Dropout2d(p=0.3)
        )

        self.hidden2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1)
            ,nn.ReLU()
            ,nn.Conv2d(64, 64, 3)
            ,nn.ReLU()
            ,nn.BatchNorm2d(64)
            ,nn.MaxPool2d(2)
            ,nn.Dropout2d(p=0.3)
        )

        self.hidden3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1)
            ,nn.ReLU()
            ,nn.Conv2d(128, 128, 3)
            ,nn.ReLU()
            ,nn.BatchNorm2d(128)
            ,nn.MaxPool2d(2)
            ,nn.Dropout2d(p=0.3)
        )

        self.hidden4 = nn.Sequential(
            nn.Conv2d(128, 256, 3)
            ,nn.ReLU()
            ,nn.BatchNorm2d(256)
            ,nn.MaxPool2d(2)
            ,Flatten()
            ,nn.Dropout(p=0.3)
        )
        
        self.digit1 = nn.Linear(9216, 36)
        self.digit2 = nn.Linear(9216, 36)
        self.digit3 = nn.Linear(9216, 36)
        self.digit4 = nn.Linear(9216, 36)

    def forward(self, inputs):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.hidden4(x)
        digit1 = torch.nn.functional.softmax(self.digit1(x), dim=1)
        digit2 = torch.nn.functional.softmax(self.digit2(x), dim=1)
        digit3 = torch.nn.functional.softmax(self.digit3(x), dim=1)
        digit4 = torch.nn.functional.softmax(self.digit4(x), dim=1)
       
        return digit1, digit2, digit3, digit4

    def save(self, path):
        """Save parameters of model.

        Args:
            path(str): parameters file path.

        """
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """Load parameters of model.

        Args:
            path(str): parameters file path.

        """
        self.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
