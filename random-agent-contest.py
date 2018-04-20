from retro_contest.local import make



def main():
    #pick the game and level
    env = make(game='SonicTheHedgehog-Genesis',state='LabyrinthZone.Act1')
    obs = env.reset()
    while True:
        action = env.action_space.sample()
        action[7] = 1 #always walk right
        obs, rew, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()
            #env.close()


if __name__ == '__main__':
    main()
