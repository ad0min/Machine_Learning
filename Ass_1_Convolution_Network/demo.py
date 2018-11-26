from loss_function import SquareLoss, CrossEntropy
import numpy as np
loss = SquareLoss()

print(loss.loss(np.random.normal(1,0.2, size=(256,10)), np.ones((256,10))))
print(type(SquareLoss),type(CrossEntropy))