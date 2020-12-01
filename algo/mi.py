import numpy as np
import torch
import time
a = 0
print(time.strftime('%m_%d_%H_%M',time.localtime(time.time())))
# print(time.strftime('%m-%d %H:%M:%S',time.localtime(time.time()))
# from tensorboardX import SummaryWriter
# writer = SummaryWriter()
# for j in range(100):
#     writer.add_scalar("scalar/test", np.random.rand(),j)
#     # writer.add_scalars("scalar/scalars_test")
# writer.close()
# c = torch.tensor([[1,2,3],[5,7,9],[8,1,2]])
# a = torch.tensor([[1],[0],[2]])
# b =torch.lt(c,7)
#
# print(a)
#
# print(torch.select(c,1,a))
import ai2thor.controller
import cv2
c = ai2thor.controller.Controller()
c.start()
c.reset("FloorPlan425")
c.step(dict(action = "Initialize",renderImage = True,renderDepthImage =True,renderObjectImage =True))
envent0 = c.step(dict(action = "AddThirdPartyCamera", rotation=dict(x=90,y=0,z=0),position=dict(x=-0.5,y=2,z=4)))
envent1= c.step(dict(action = "AddThirdPartyCamera", rotation=dict(x=90,y=0,z=0),position=dict(x=-0.5,y=2.5,z=4)))

b,g,r = cv2.split(envent0.third_party_camera_frames[0])
cv2.imwrite("image0.jpg",cv2.merge([r,g,b]))

b,g,r = cv2.split(envent1.third_party_camera_frames[0])
cv2.imwrite("image1.jpg",cv2.merge([r,g,b]))