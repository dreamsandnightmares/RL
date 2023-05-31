import argparse
import copy
import mujoco_py
import gym
import numpy as np
import torch
import  torch.nn as nn
import torch.nn.functional as F




class Actor(nn.Module):
    def __init__(self,state_dim,action_dim,max_action):
        super(Actor,self).__init__()

        self.l1 =nn.Linear(state_dim,400)
        self.l2 = nn.Linear(400,300)
        self.l3 = nn.Linear(300,action_dim)

        self.max_action =max_action

    def forward(self,state):
        a =F.relu(self.l1(state))
        b =F.relu(self.l2(a))
        return self.max_action*torch.tanh(self.l3(b))


class Critic(nn.Module):

    def __init__(self,state_dim,action_dim):
        super(Critic,self).__init__()

        self.l1 = nn.Linear(state_dim,400)
        self.l2 = nn.Linear(400+action_dim,300)
        self.l3 = nn.Linear(300,1)
    def forward(self,state,action):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(torch.cat([q+action],1)))
        return self.l3(q)


class DDPG(object):
    def __init__(self,state_dim,action_dim,max_action,discount=0.99,tau =0.001):
        self.device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.actor = Actor(state_dim=state_dim,action_dim=action_dim,max_action=max_action).to(self.device)
        self.actor_target =copy.deepcopy(self.actor)
        self.actor_optimzer = torch.optim.Adam(self.actor.parameters(),lr=1e-4)

        self.critic = Critic(state_dim=state_dim, action_dim=action_dim).to(self.device)
        self.Critic_target =copy.deepcopy(self.critic)
        self.critic_optimzer = torch.optim.Adam(self.critic.parameters(),weight_decay=1e-2)

        self.discount = discount
        self.tau =tau

    def take_action(self,state):

        state = torch.FloatTensor(state.shape(-1,1)).to(self.device)
        return self.actor(state).cpu().data.numpy().flatten()

    def train(self,replay_buffer,batch_size = 64):

        state,action,next_state,reward,done = replay_buffer.sample(batch_size)
        target_Q = reward+self.discount*self.Critic_target(next_state,self.actor_target(next_state))*done

        current_Q = self.critic(state,action)

        critic_loss =F.mse_loss(current_Q,target_Q)

        self.critic_optimzer.zero_grad()
        critic_loss.backward()
        self.critic_optimzer.step()

        actor_loss = -self.critic(state,self.actor(state)).meam()

        self.actor_optimzer.zero_grad()
        actor_loss.backward()
        self.actor_optimzer.step()


        for param,target_param in zip(self.actor.parameters(),self.actor_target.parameters()):
            target_param.data.deepcopy(self.tau*param.data+(1-self.tau)*target_param.data)

        for param,target_param in zip(self.critic.parameters(),self.Critic_target.parameters()):
            target_param.data.deepcopy(self.tau*param.data+(1-self.tau)*target_param.data)

    def save(self,filename):
        torch.save(self.critic.state_dict(),filename+'_critic')
        torch.save(self.critic_optimzer.state_dict(),filename+'_critci_optimizer')

        torch.save(self.actor.state_dict(),filename+"_actor")
        torch.save(self.actor_optimzer.state_dict(),filename+"_actor_optimizer")


    def load(self,filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimzer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimzer.load_state_dict(torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)


def eval_policy(policy,env_name,seed,eval_episodes =10):
    eval_env = gym.make(env_name)
    eval_env.seed(seed+100)

    avg_reward =0

    for _ in range(eval_episodes):
        state,done = eval_env.reset()
        while not  done:
            action = policy.take_action(state)
            state,reward,done,_ =eval_env.step(action)

            avg_reward +=reward
    avg_reward /=eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    env_name = "HalfCheetah-v2"
    env = gym.make(env_name)
    env.seed(1)
    env.action_space.seed(1)
    np.random.seed(1)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": 0.99,
        "tau": 0.005,
    }
    print(env.action_space)























