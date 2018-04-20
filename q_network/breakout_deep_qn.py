import gym

#self-evident, average across z-dim (RGB vals)
def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

#downsamples by 2x
def downsample(img):
    return img[::2,::2]

def preprocess(img):
    return to_grayscale(downsample(img))

#fix all rewards to be -1, 0, or 1
def transform_reward(reward):
    return np.sign(reward)

env = gym.make('BreakoutDeterministic-v4')

frame = env.reset()
env.render()

is_done = False

while not is_done:
    frame, reward, is_done, _ = env.step(env.action_space.sample())

    env.render()
