import numpy as np
import cv2
import h5py
import os
import random


class MultiSceneEnv():
    """
    Wrapper base class
    """

    def __init__(self, scene, config, arguments=dict(), seed=None):
        """
        :param seed: (int)   Random seed
        :param config: (str)   Dictionary file storing cofigurations
        :param: scene: (list)  Scene to train
        :param: objects: (list)  Target objects to train
        """

        if seed is not None:
            np.random.seed(seed)

        self.config = config
        self.arguments = arguments
        self.scene = scene

        self.history_size = arguments.history_size
        self.action_size = arguments.action_size

        scene_id = int(scene.split("FloorPlan")[1])
        if scene_id > 0 and scene_id < 31:
            room_type = "Kitchens"
        elif scene_id > 200 and scene_id < 231:
            room_type = 'Living Rooms'
        elif scene_id > 300 and scene_id < 331:
            room_type = 'Bedrooms'
        elif scene_id > 400 and scene_id < 431:
            room_type = 'Bathrooms'
        else:
            raise KeyError

        self.h5_file = h5py.File("{}.hdf5".format(os.path.join(config['dump_path'], self.scene)), 'r')
        # print("e", os.path.join(config['dump_path'], self.scene))
        try:
            self.states = self.h5_file['locations'][()]
        # except
        except:
            print("error",scene)



        self.graph = self.h5_file['graph'][()]
        # self.scores = self.h5_file['resnet_scores'][()]
        self.all_visible_objects = self.h5_file['all_visible_objects'][()].tolist()
        self.visible_objects = self.h5_file['visible_objects'][()]
        self.observations = self.h5_file['observations'][()]

        self.resnet_features = self.h5_file['resnet_features'][()]
        # self.dump_features = self.h5_file['dump_features'][()]
        if arguments.test == 1:
            self.targets = [x for x in
                             config["rooms"][room_type]['test_objects'] if
                            x in self.all_visible_objects]


        else:
            self.targets = [x for x in config["rooms"][room_type]['train_objects'] if x in self.all_visible_objects]
        if len(self.targets) == 0:
            print("error")
        # if arguments['onehot']:
        #     self.features = self.dump_features
        # else:
        self.features = self.resnet_features
        if not self.arguments.hard:
            assert self.action_size <= self.graph.shape[1], "The number of actions exceeds the limit of environment."
        else:
            assert self.action_size <= self.graph.shape[1]+1, "The number of actions exceeds the limit of environment."


        if "shortest" in self.h5_file.keys():
            self.shortest = self.h5_file['shortest'][()]

        if self.arguments.hard:
            # agent has to reach the correct position and has right rotation
            self.offset = 3
        else:
        #     # agent only has to reach the correct position
             self.offset = 2
        # self.offset = 3

        self.target = np.random.choice(self.targets)
        self.target_ids = [idx for idx in range(len(self.states)) if
                           self.target in self.visible_objects[idx].split(",")]
        self.target_locs = set([tuple(self.states[idx][:self.offset]) for idx in self.target_ids])

        self.action_space = self.action_size
        self.cv_action_onehot = np.identity(self.action_space)

        self.history_states = np.zeros((self.history_size, self.features.shape[1]))
        self.history_objs = []
        # self.observations_stack = [np.zeros((3, 2048)) for _ in range(self.history_size)]

    def step(self,action):
        '''
        0: move ahead
        1: move back
        2: rotate right
        3: rotate left
        # 4: look down
        # 5: look up
        '''
        # print("action:",action, "action_space:", self.action_space)
        if action >= self.action_space:
            raise error.InvalidAction('Action must be an integer between '
                                      '0 and {}!'.format(self.action_space - 1))
        k = self.current_state_id
        # print("current_id", k,action)
        if not self.arguments.hard:
            if self.graph[k][action] != -1:
                # if action == 2 or action == 3:
                #     # for _ in range(int(self.arguments.angle / 90)):
                #     self.current_state_id = int(self.graph[k][action])
                # elif action == 1:
                #     self.current_state_id = int(self.graph[k][action])
                # else:
                self.current_state_id = int(self.graph[k][action])

                # print(self.current_state_id)

                if tuple(self.states[self.current_state_id][:self.offset]) in self.target_locs\
                        and self.current_state_id in self.target_ids:
                    # print(tuple(self.states[self.current_state_id][:self.offset]))
                    # print(self.target_locs)
                    self.terminal = True
                    self.collided = False
                else:
                    self.terminal = False
                    self.collided = False
            else:
                self.terminal = False
                self.collided = True
        else:
            if action != (self.action_size-1):

                if self.graph[k][action] != -1:

                    # if action == 1 or action == 2:
                    #     for _ in range(int(self.arguments.angle / 45)):
                    #         self.current_state_id = int(self.graph[k][action + 1])
                    # else:
                    self.current_state_id = int(self.graph[k][action])
                    if self.current_state_id == k:
                        print("stop")
                    self.terminal = False
                    self.collided = False
                else:
                    self.terminal = False
                    self.collided = True
            else:
                if tuple(self.states[self.current_state_id][:self.offset]) in self.target_locs:
                    self.terminal = True
                    self.collided = False
                else:
                    self.terminal = False
                    self.collided = False
        reward, done = self.transition_reward()
        self.update_states()

        state_cc = self.states[self.current_state_id]
        if self.arguments.run_type == 1:
            print(state_cc)
        return self.history_states, self.objs_stack[-1], reward, done

    def transition_reward(self):
        reward = self.config['default_reward']
        done = 0
        if self.terminal:
            reward = self.config['success_reward']
            done = 1

        # elif self.arguments.anti_col and self.collided:
        elif self.collided:
            reward = self.config['collide_reward']

        return reward, done

    def reset(self):
        self.target = np.random.choice(self.targets)
        # print(" Now find {} in {}! ".format(self.target, self.scene))


        # self.target = self.targets[0]
        self.target_ids = [idx for idx in range(len(self.states)) if
                           self.target in self.visible_objects[idx].split(",")]
        self.target_locs = set([tuple(self.states[idx][:self.offset]) for idx in self.target_ids])

        k = random.randrange(self.states.shape[0])

        while self.states[k][2] % self.arguments.angle != 0.0:
            k = random.randrange(self.states.shape[0])

        # reset parameters
        self.current_state_id = k

        self.update_states(reset=True)
        self.terminal = False
        self.collided = False
        return self.history_states, self.objs_stack[-1], self.target

    def update_states(self, reset=False):
        if reset:
            self.history_states = np.zeros((self.history_size, self.features.shape[1]))
            # self.observations_stack = [np.zeros((3, 128, 128)) for _ in range(self.history_size)]
            self.objs_stack = []
            self.history_objs = []

        cur_objs_list = [x for x in self.visible_objects[self.current_state_id].split(",") if
                         x is not ""]
        self.history_objs.append(cur_objs_list)
        state = self.states[self.current_state_id]
        if (state[2] - self.arguments.angle) >= 0:
            current_left = [state[0], state[1], state[2] - self.arguments.angle]
        else:
            current_left = [state[0], state[1], 360 + state[2] - self.arguments.angle]
        if (state[2] + self.arguments.angle) >= 360:
            current_right = [state[0], state[1], state[2] + self.arguments.angle - 360]
        else:
            current_right = [state[0], state[1], state[2] + self.arguments.angle]
        if (state[2] + 2*self.arguments.angle) >= 360:
            current_back = [state[0], state[1], state[2] + 2*self.arguments.angle - 360]
        else:
            current_back = [state[0], state[1], state[2] + 2*self.arguments.angle]

        for i in range(len(self.states)):
            # print(self.states[i])
            if current_right == self.states[i].tolist():
                right_id = i
                r_obj_ids = [x for x in self.visible_objects[i].split(",") if x is not ""]
                current_right_feature = self.features[i]
            if current_left == self.states[i].tolist():
                left_id = i
                l_obj_ids = [x for x in self.visible_objects[i].split(",") if x is not ""]
                # if current_left and current_right:

                current_left_feature = self.features[i]
            if current_back == self.states[i].tolist():
                back_id = i
                b_obj_ids = [x for x in self.visible_objects[i].split(",") if x is not ""]
                # if current_left and current_right:

                current_back_feature = self.features[i]


        try:
            cur_states = [self.current_state_id, back_id, right_id,left_id]
        except:
            print("error",current_back)
        # image = self.observations[self.current_state_id]
        # cv2.imshow("state", image)
        # cv2.imwrite("state.png", image)
        # cv2.waitKey(20)
        # r_image = self.observations[right_id]
        # cv2.imshow("r_state", r_image)
        # cv2.imwrite("r_state.png", r_image)
        # cv2.waitKey(20)
        # l_image = self.observations[left_id]
        # cv2.imshow("l_state", l_image)
        # cv2.imwrite("l_state.png", l_image)
        # cv2.waitKey(20)

        image_features = (self.features[self.current_state_id],current_back_feature,current_right_feature, current_left_feature )
        image_features = np.hstack(image_features)
        self.history_states = image_features

        # cur_feature = torch.stack((current_left_feature, self.features[self.current_state_id],current_right_feature),4).float() #current resnet feature
        objs_list = [cur_objs_list,b_obj_ids,  r_obj_ids,l_obj_ids]
        self.objs_stack.append(objs_list)
        # obj_state = torch.from_numpy(self.get_obj_state(len(self.all_objects), objs_index)).unsqueeze(0).float()
        # self.observations_stack.append(image_features)
        # self.observations_stack = self.observations_stack[1:]

    def state(self):
        return self.features[self.current_state_id]

    def observation(self):
        ob = self.observations[self.current_state_id]
        resized_ob = cv2.resize(ob, (128, 128))
        return np.transpose(resized_ob, (2, 0, 1))