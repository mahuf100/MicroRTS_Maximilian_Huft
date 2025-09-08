import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import argparse
from distutils.util import strtobool
import numpy as np
import gym
import gym_microrts
from gym_microrts.envs.vec_env2 import MicroRTSGridModeVecEnv
from gym_microrts import microrts_ai
from gym.wrappers import TimeLimit, Monitor

from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicrortsDefeatCoacAIShaped-v3",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=100000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod-mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument('--capture-video', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='weather to capture videos of the agent performances (check out `videos` folder)')
    parser.add_argument('--wandb-project-name', type=str, default="cleanRL",
                        help="the wandb's project name")
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help="the entity (team) of wandb's project")

    # Algorithm specific arguments
    parser.add_argument('--n-minibatch', type=int, default=4,
                        help='the number of mini batch')
    parser.add_argument('--num-bot-envs', type=int, default=24,
                        help='the number of bot game environment; 16 bot envs measn 16 games')
    parser.add_argument('--num-selfplay-envs', type=int, default=0,
                        help='the number of self play envs; 16 self play envs means 8 games')
    parser.add_argument('--num-steps', type=int, default=256,
                        help='the number of steps per game environment')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--gae-lambda', type=float, default=0.95,
                        help='the lambda for the general advantage estimation')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help="coefficient of the entropy")
    parser.add_argument('--vf-coef', type=float, default=0.5,
                        help="coefficient of the value function")
    parser.add_argument('--max-grad-norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--clip-coef', type=float, default=0.1,
                        help="the surrogate clipping coefficient")
    parser.add_argument('--update-epochs', type=int, default=4,
                         help="the K epochs to update the policy")
    parser.add_argument('--kle-stop', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will be early stopped w.r.t target-kl')
    parser.add_argument('--kle-rollback', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                         help='If toggled, the policy updates will roll back to previous policy if KL exceeds target-kl')
    parser.add_argument('--target-kl', type=float, default=0.03,
                         help='the target-kl variable that is referred by --kl')
    parser.add_argument('--norm-adv', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggles advantages normalization")
    parser.add_argument('--anneal-lr', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument('--clip-vloss', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                          help='Toggles wheter or not to use a clipped loss for the value function, as per the paper.')

    args = parser.parse_args()
    if not args.seed:
        args.seed = int(time.time())


args.num_envs = args.num_selfplay_envs + args.num_bot_envs
args.batch_size = int(args.num_envs * args.num_steps)
args.minibatch_size = int(args.batch_size // args.n_minibatch)

class VecstatsMonitor(VecEnvWrapper):
    def __init__(self, venv, gamma=None):
        super().__init__(venv)
        self.eprets = None
        self.eplens = None
        self.epcount = 0
        self.tstart = time.time()
        self.gamma = gamma
        self.raw_rewards = None

    def reset(self):
        obs = self.venv.reset()
        n = self.num_envs
        self.eprets = np.zeros(n, dtype=float)
        self.eplens = np.zeros(n, dtype=int)
        self.raw_rewards = [[] for _ in range(n)]
        self.tstart = time.time()
        return obs

    def step_wait(self):
        obs, denserews,winlossrews, scorerews , dones, infos = self.venv.step_wait()

        self.eprets += denserews +winlossrews +scorerews
        self.eplens += 1

        for i, info in enumerate(infos):
            if 'raw_rewards' in info:
                self.raw_rewards[i].append(info['raw_rewards'])

        newinfos = list(infos)

        for i, done in enumerate(dones):
            if done:
                info = infos[i].copy()
                ep_ret = float(self.eprets[i])
                ep_len = int(self.eplens[i])
                ep_time = round(time.time() - self.tstart, 6)
                info['episode'] = {'r': ep_ret, 'l': ep_len, 't': ep_time}


                self.epcount += 1

                if self.raw_rewards[i]:
                    agg = np.sum(np.array(self.raw_rewards[i]), axis=0)
                    raw_names = [str(rf) for rf in self.rfs]
                    info['microrts_stats'] = dict(zip(raw_names, agg.tolist()))
                else:
                    info['microrts_stats'] = {}

                if winlossrews[i] == 0:
                    info['microrts_stats']['draw'] = True
                else:
                    info['microrts_stats']['draw'] = False

                self.eprets[i] = 0.0
                self.eplens[i] = 0
                self.raw_rewards[i] = []
                newinfos[i] = info

        return obs, denserews,winlossrews, scorerews, dones, newinfos

    def step(self, actions):
        self.venv.step_async(actions)
        return self.step_wait()

experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
# = "MicrortsDefeatCoacAIShaped-v3__lstm__1__1747682181"
writer = SummaryWriter(f"runs/{experiment_name}")
writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

RUN_ID_PATH = f"models/{experiment_name}/wandb_run_id.txt"
if args.prod_mode:
    import  wandb
    if os.path.exists(RUN_ID_PATH):
        # Resume: read the previous run ID
        with open(RUN_ID_PATH, "r") as f:
            run_id = f.read().strip()
        resume_mode = "must"
    else:
        # First time: no file exists
        run_id = None
        resume_mode = "allow"
    run = wandb.init(
        project=args.wandb_project_name, entity=args.wandb_entity,
        # sync_tensorboard=True,
        config=vars(args), name=experiment_name, monitor_gym=True,resume=resume_mode,id=run_id, save_code=True)

    if resume_mode == "allow":
        os.makedirs(os.path.dirname(RUN_ID_PATH), exist_ok=True)
        with open(RUN_ID_PATH, "w") as f:
            f.write(run.id)
    wandb.tensorboard.patch(save=False)
    writer = SummaryWriter(f"/tmp/{experiment_name}")
    CHECKPOINT_FREQUENCY = 10


device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')



random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=args.num_selfplay_envs,
    num_bot_envs=args.num_bot_envs,
    max_steps=2000,
    render_theme=1,
    ai2s=[microrts_ai.coacAI for _ in range(args.num_bot_envs-6)] + \
         #[microrts_ai.droplet for _ in range(2)] + \
         #[microrts_ai.naiveMCTSAI for _ in range(2)] + \
         [microrts_ai.randomBiasedAI for _ in range(2)] + \
         [microrts_ai.lightRushAI for _ in range(2)] + \
         [microrts_ai.workerRushAI for _ in range(2)],
    map_path="maps/16x16/basesWorkers16x16.xml",
    reward_weight=np.array([10.0,1.0, 1.0, 0.2, 1.0, 4.0, 5.25, 6.0, 0])
)

envs = VecstatsMonitor(envs, args.gamma)

if args.capture_video:
    envs = VecVideoRecorder(envs, f'videos/{experiment_name}',
                            record_video_trigger=lambda x: x % 1000000 == 0, video_length=2000)


# ALGO LOGIC: initialize agent here:
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)



T = 20
frame_buffer = deque(maxlen=T)


def get_sequence():
    # returns [B, T, H, W, C]

    buf = list(frame_buffer)
    if len(buf) < T:
        pad = [torch.zeros_like(buf[0])]*(T - len(buf))
        buf = pad + buf
    return torch.stack(buf, dim=1)
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = layer_init(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.conv2 = layer_init(nn.Conv2d(channels, channels, kernel_size=3, padding=1))

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        return F.relu(out + x)

class Agent(nn.Module):
    def __init__(self, mapsize=16 * 16, lstm_hidden=384, lstm_layers=3):
        super(Agent, self).__init__()
        self.mapsize = mapsize
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(27, 32, kernel_size=3, stride=2, padding=1)),
            nn.ReLU(),
            ResBlock(32),
            ResBlock(32),
            nn.Flatten(),
            layer_init(nn.Linear(32 * 8 * 8, 256)),
            nn.ReLU()
        )

        # LSTM core
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )
        # Actor heads per branch
        self.actor = nn.Linear(lstm_hidden, self.mapsize*envs.action_space.nvec[1:].sum())
        self.critic = layer_init(nn.Linear(lstm_hidden, 1), std=1)

    def forward(self, x, hx=None, first_step=False):
        if x.ndim == 4:
            x = x.unsqueeze(1)
        x = x.permute(0, 1, 4, 2, 3)
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        print(x_flat.shape)
        feats = self.encoder(x_flat)  # [B*T, 256]
        print(feats.shape)
        feats = feats.view(B, T, -1)  # [B, T, 256]
        print(feats.shape)
        # init hidden state if needed
        if first_step or hx is None:
            device = feats.device
            h0 = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=device)
            c0 = torch.zeros(self.lstm.num_layers, B, self.lstm.hidden_size, device=device)
            hx = (h0, c0)
        # run LSTM
        print(feats.shape)
        lstm_out, hx = self.lstm(feats, hx)  # [B, T, hidden]
        return lstm_out, hx

    def get_action(self, x, action=None, invalid_action_masks=None, envs=None, hx=None, first_step=False):
        # x: [B, 1, H, W, C]
        lstm_out, hx = self.forward(x, hx, first_step)
        h = lstm_out[:, -1, :]  # [B, hidden]
        flat_logits = self.actor(h)
        # reshape to [B * mapsize, Σn_i]
        B = flat_logits.size(0)
        flat_logits = flat_logits.view(B * self.mapsize, -1)
        split_logits = torch.split(flat_logits, envs.action_space.nvec[1:].tolist(), dim=1)
        if action is None:
            invalid_action_masks = torch.tensor(np.array(envs.vec_client.getMasks(0))).to(device)
            invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
            split_invalid_action_masks = torch.split(invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(),
                                                     dim=1)

            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                  zip(split_logits, split_invalid_action_masks)]
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        else:
            invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
            action = action.view(-1, action.shape[-1]).T
            split_invalid_action_masks = torch.split(invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(),
                                                     dim=1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in
                                  zip(split_logits, split_invalid_action_masks)]
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        num_predicted_parameters = len(envs.action_space.nvec) - 1
        logprob = logprob.T.view(-1, 256, num_predicted_parameters)
        entropy = entropy.T.view(-1, 256, num_predicted_parameters)
        action = action.T.view(-1, 256, num_predicted_parameters)
        invalid_action_masks = invalid_action_masks.view(-1, 256, envs.action_space.nvec[1:].sum() + 1)

        return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks, hx

    def get_initial_hidden_state(self, batch_size):
        return (
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device),
            torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(device)
        )

    def get_value(self, x, hx=None, first_step=False):
        lstm_out, _ = self.forward(x, hx, first_step)
        h = lstm_out[:, -1, :]
        return self.critic(h).squeeze(-1)


agent = Agent().to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
if args.anneal_lr:
    # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
    lr = lambda f: f * args.learning_rate

# ALGO Logic: Storage for epoch data
mapsize = 16 * 16
action_space_shape = (mapsize, envs.action_space.shape[0] - 1)
invalid_action_shape = (mapsize, envs.action_space.nvec[1:].sum() + 1)

obs = torch.zeros((args.num_steps, args.num_envs) + envs.observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + action_space_shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)

rewards_dense = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards_winloss = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards_score = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)
invalid_action_masks = torch.zeros((args.num_steps, args.num_envs) + invalid_action_shape).to(device)
# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()
# Note how `next_obs` and `next_done` are used; their usage is equivalent to
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/84a7582477fb0d5c82ad6d850fe476829dddd2e1/a2c_ppo_acktr/storage.py#L60
next_obs = torch.Tensor(envs.reset()).to(device)
next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size
## CRASH AND RESUME LOGIC:
starting_update = 1


obs_buffer    = torch.zeros(args.num_steps, args.num_envs, 16, 16, 27, device=device)
actions_buffer   = torch.zeros(args.num_steps, args.num_envs, device=device, dtype=torch.long)
logprobs_buffer  = torch.zeros(args.num_steps, args.num_envs, device=device)
values_buffer    = torch.zeros(args.num_steps, args.num_envs, device=device)
rewards_buffer   = torch.zeros(args.num_steps, args.num_envs, device=device)
dones_buffer     = torch.zeros(args.num_steps, args.num_envs, device=device)
# Mask = 1 for ongoing episodes, 0 if done (to reset LSTM in next step, used later in advantage computation)
masks_buffer     = torch.ones(args.num_steps, args.num_envs, device=device)


from jpype.types import JArray, JInt

if args.prod_mode and wandb.run.resumed:
    starting_update = run.summary.get('charts/update') + 1
    global_step = starting_update * args.batch_size
    ckpt_path = f"models/{experiment_name}/agent.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    agent.load_state_dict(torch.load(ckpt_path, map_location=device))
    agent.eval()
    print(f"resumed at update {starting_update}")

for update in range(starting_update, num_updates + 1):
    hx = None
    first_step = True
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow
    frame_buffer.clear()
    # TRY NOT TO MODIFY: prepare the execution of the game.
    for step in range(0, args.num_steps):
        envs.render()

        global_step += 1 * args.num_envs
        obs[step] = next_obs
        obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)
        print(obs.shape)
        print(next_obs.shape)

        dones[step] = next_done
        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            value = agent.get_value(next_obs.unsqueeze(1), hx, first_step)  # shape [1, B, ...]
            action, logprob, entropy,invalid_action_masks[step] , hx = agent.get_action(next_obs.unsqueeze(1),hx= hx,first_step=first_step, envs=envs)


        actions[step] = action
        logprobs[step] = logprob

        first_step = False

        real_action = torch.cat([
            torch.stack(
                [torch.arange(0, mapsize, device=device) for i in range(envs.num_envs)
                 ]).unsqueeze(2), action], 2)

        # at this point, the `real_action` has shape (num_envs, map_height*map_width, 8)
        # so as to predict an action for each cell in the map; this obviously include a
        # lot of invalid actions at cells for which no source units exist, so the rest of
        # the code removes these invalid actions to speed things up
        real_action = real_action.cpu().numpy()

        valid_actions = real_action[invalid_action_masks[step][:, :, 0].bool().cpu().numpy()]
        valid_actions_counts = invalid_action_masks[step][:, :, 0].sum(1).long().cpu().numpy()


        java_valid_actions = []
        valid_action_idx = 0
        for env_idx, valid_action_count in enumerate(valid_actions_counts):
            java_valid_action = []
            for c in range(valid_action_count):
                java_valid_action += [JArray(JInt)(valid_actions[valid_action_idx])]
                valid_action_idx += 1
            java_valid_actions += [JArray(JArray(JInt))(java_valid_action)]
        java_valid_actions = JArray(JArray(JArray(JInt)))(java_valid_actions)

        try:

            next_obs, denserew, winlossrew,scorerew , ds, infos = envs.step(java_valid_actions)
            next_obs = torch.Tensor(next_obs).to(device)

        except Exception as e:
            e.printStackTrace()
            raise

        winloss = min(0.01, 2.72222222e-9 * global_step)
        densereward = max(0, 0.8 + (-4.44444444e-9 * global_step))
        if global_step < 170000000:
            scorew = 0.19 + 1.754e-8 * global_step
        else:
            scorew = 0.5 - 1.33e-8 * global_step


        rewards_dense[step] = torch.Tensor(denserew* densereward).to(device)
        rewards_winloss[step] = torch.Tensor(winlossrew*winloss).to(device)
        rewards_score[step] = torch.Tensor(scorerew*scorew).to(device)
        next_done = torch.Tensor(ds).to(device)
        done_mask = next_done.view(1, -1, 1).bool()
        done_mask = done_mask.expand_as(hx[0])
        print(done_mask.shape)
        print(hx[0].shape)

        hx = (
            hx[0] * (~done_mask).float(),
            hx[1] * (~done_mask).float()
        )

        for info in infos:
           if 'episode' in info.keys():
                print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                for key in info['microrts_stats']:
                    writer.add_scalar(f"charts/episode_reward/{key}", info['microrts_stats'][key], global_step)

                break


    with torch.no_grad():
        last_value = agent.get_value(next_obs.unsqueeze(1).to(device), hx).flatten().reshape(1, -1)
        advantages = torch.zeros_like(rewards_winloss).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards_dense[t] + rewards_winloss[t] + rewards_score[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

    print(obs.shape)
    b_obs = obs.reshape((-1,) + envs.observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + action_space_shape)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    b_invalid_action_masks = invalid_action_masks.reshape((-1,) + invalid_action_shape)

    # Optimizaing the policy and value network
    inds = np.arange(args.batch_size, )
    for epoch in range(args.update_epochs):
        # shuffle environment indices
        perm_envs = torch.randperm(24)

        # break into env-minibatches of size M (e.g. 8)
        M = args.minibatch_size  # must divide B
        for i in range(3):
            # Select a batch of environment indices
            env_idx = perm_envs[i * 8: (i + 1) * 8]

            # Slice sequences for these environments
            # obs_buffer: [T, 24, H, W, C] -> select envs -> [T, B, H, W, C]
            obs_batch = obs[:, env_idx]  # [20, 8, H, W, C]
            print(obs_batch.shape)
            actions_batch = actions[:, env_idx]  # [20, 8]
            mask_batch = invalid_action_masks[:, env_idx]
            old_logprobs_batch = logprobs[:, env_idx]
            old_values_batch = values[:, env_idx]
            returns_batch = returns[:, env_idx]
            adv_batch = advantages[:, env_idx]

            # Permute to [B, T, C, H, W] for network input
            obs_batch = obs_batch.permute(1, 0, 4, 2, 3)  # [8, 20, C, H, W]
            # Detach old values/logprobs from graph
            actions_batch = actions_batch.permute(1, 0)  # [8, 20]
            old_logprobs_batch = old_logprobs_batch.permute(1, 0)  # [8, 20]
            returns_batch = returns_batch.permute(1, 0)  # [8, 20]
            adv_batch = adv_batch.permute(1, 0)  # [8, 20]

            # Initialize hidden states for this mini-batch (zero for a new forward pass)
            hx_batch = torch.zeros(1, 8, 384, device=device)
            cx_batch = torch.zeros(1, 8, 384, device=device)

            # Forward pass through policy (to get new logprobs) and value network
            # Option A: If agent has a method to get logprobs of specific actions:
            new_logprobs, values_pred = agent.get_action(obs_batch, actions_batch, (hx_batch, cx_batch))

            # Option B: Sample from policy and ignore actions (not recommended as actions differ)
            # new_actions, new_logprobs, (hx_batch, cx_batch) = agent.get_action(obs_batch, (hx_batch, cx_batch))
            # values_pred, _ = agent.get_value(obs_batch, (hx_batch, cx_batch))

            # For clarity, assume agent can compute log-prob of given actions:
            new_logprobs, (hx_batch, cx_batch) = agent.get_log_prob(obs_batch, actions_batch, (hx_batch, cx_batch))
            values_pred, _ = agent.get_value(obs_batch, (hx_batch, cx_batch))
            values_pred = values_pred.squeeze(-1)  # [8, 20]

            # Compute PPO losses
            clip_eps = 0.2
            # Ratio for policy gradient
            ratio = (new_logprobs - old_logprobs_batch).exp()  # [8,20]
            # Policy loss (clipped surrogate)
            adv_batch_detach = adv_batch.detach()
            pg_loss = -torch.min(
                ratio * adv_batch_detach,
                torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_batch_detach
            ).mean()
            # Value loss
            value_loss = 0.5 * (values_pred - returns_batch).pow(2).mean()
            # (Entropy bonus could be added if get_log_prob returns distribution info)

            loss = pg_loss + value_loss  # + (optional) entropy bonus

            # Backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    if args.prod_mode and update % CHECKPOINT_FREQUENCY == 0:
        print("checkpoint")
        os.makedirs(f"models/{experiment_name}", exist_ok=True)
        torch.save(agent.state_dict(), f"models/{experiment_name}/agent.pt")

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
    writer.add_scalar("charts/update", update, global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy.mean().item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    if args.kle_stop or args.kle_rollback:
        writer.add_scalar("debug/pg_stop_iter", i_epoch_pi, global_step)
    writer.add_scalar("charts/sps", int(global_step / (time.time() - start_time)), global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))

envs.close()
writer.close()
