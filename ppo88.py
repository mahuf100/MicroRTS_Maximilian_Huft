from time import process_time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import IterableDataset
from sympy.codegen import Print
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import argparse
from distutils.util import strtobool
import numpy as np
import gym
import gym_microrts
#from gym_microrts.envs.vec_env2 import MicroRTSGridModeVecEnv
from gym_microrts.envs.microrts_vec_env import  MicroRTSGridModeVecEnv
from gym_microrts.envs.microrts_bot_vec_env import MicroRTSBotGridVecEnv
from gym_microrts import microrts_ai
from gym.wrappers import TimeLimit, Monitor
from typing import Any, Dict, List, Optional, TypeVar
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import json
import random
import os
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder
from microrts_space_transform import MicroRTSSpaceTransform
from microrts_space_transformbots import MicroRTSSpaceTransformbot
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PPO agent')
    # Common arguments
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument('--gym-id', type=str, default="MicrortsDefeatCoacAIShaped-v3",
                        help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, default=2.5e-5,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=1,
                        help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, default=10000000000,
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
    parser.add_argument('--num-steps', type=int, default=512,
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
        obs, denserews,winlossrews, scorerews , dones, infos,res = self.venv.step_wait()

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

        return obs, denserews,winlossrews, scorerews, dones, newinfos,res

    def step(self, actions):
        self.venv.step_async(actions)
        return self.step_wait()

experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
#experiment_name = "MicrortsDefeatCoacAIShaped-v3__int8__1__1752473888"
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
        config=vars(args), name=experiment_name, monitor_gym=True,resume=resume_mode,id=run_id, save_code=False)

    if resume_mode == "allow":
        os.makedirs(os.path.dirname(RUN_ID_PATH), exist_ok=True)
        with open(RUN_ID_PATH, "w") as f:
            f.write(run.id)
    wandb.tensorboard.patch(save=False)
    writer = SummaryWriter(f"/tmp/{experiment_name}")
    CHECKPOINT_FREQUENCY = 10


device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
print(device)


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=args.num_selfplay_envs,
    num_bot_envs=args.num_bot_envs,
    max_steps=2000,
    render_theme=1,
    ai2s=[microrts_ai.coacAI for _ in range(args.num_bot_envs-10)] +[microrts_ai.mayari for _ in range(4)] + [microrts_ai.lightRushAI for _ in range(2)]+[microrts_ai.passiveAI for _ in range(2)]+[microrts_ai.workerRushAI for _ in range(2)],
    map_paths=["maps/16x16/basesWorkers16x16.xml"],
    reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0, 5.25, 6.0, 0])
)
envsT = MicroRTSSpaceTransform(envs)

envsT = VecstatsMonitor(envsT, args.gamma)
if args.capture_video:
    envs = VecVideoRecorder(envs, f'videos/{experiment_name}',
                            record_video_trigger=lambda x: x % 1000000 == 0, video_length=2000)




from torch.utils.data import DataLoader, TensorDataset, random_split
def compute_edit_distance(seq1: torch.Tensor, seq2: torch.Tensor) -> int:

    seq1 = seq1.tolist()
    seq2 = seq2.tolist()
    len1, len2 = len(seq1), len(seq2)

    # Initialize DP table
    dp = [[0] * (len2 + 1) for _ in range(len1 + 1)]

    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    # Fill DP table
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # Deletion
                dp[i][j - 1] + 1,      # Insertion
                dp[i - 1][j - 1] + cost  # Substitution
            )

    return dp[len1][len2]
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

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



def getScalarFeatures(obs,res, numenvs):
    ScFeatures = torch.zeros(numenvs, 11)

    for i in range(numenvs):

        res_plane = (obs[i, :, :, 1]*obs[i, :, :, 7])
        lightunit_plane = (obs[i, :, :, 11])
        heavyunit_plane = (obs[0, :, :, 12])
        rangedunit_plane = (obs[0, :, :, 13])
        total_res = res_plane.sum().item()


        worker_plane = obs[i, :, :, 10]
        building_plane = obs[i, :, :, 9]
        player0_plane = obs[i, :, :, 4]
        player1_plane = obs[i, :, :, 5]

        ScFeatures[i,0]  = res[i][0] #Player0 res
        ScFeatures[i, 1] =  res[i][1] #Player1 res
        ScFeatures[i, 2] =  total_res #vorhandene res
        ScFeatures[i, 3] = (worker_plane * player0_plane).sum().item()  # Player0 worker
        ScFeatures[i, 4] = (lightunit_plane * player0_plane).sum().item()  # Player0 light
        ScFeatures[i, 5] = (heavyunit_plane * player0_plane).sum().item()  # Player0 heavy
        ScFeatures[i, 6] = (rangedunit_plane * player0_plane).sum().item()  # Player0 ranged
        ScFeatures[i, 7] = (worker_plane * player1_plane).sum().item()  # Player1 worker
        ScFeatures[i, 8] = (lightunit_plane * player1_plane).sum().item()  # Player1 light
        ScFeatures[i, 9] = (heavyunit_plane * player1_plane).sum().item()  # Player1 heavy
        ScFeatures[i, 10] = (rangedunit_plane * player1_plane).sum().item()  # Player1 ranged

    #Time step in the game
    return ScFeatures
class ScalarFeatureEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim = 32, output_dim = 32, num_layers=2):
        super(ScalarFeatureEncoder, self).__init__()

        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = layer_init(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        self.conv2 = layer_init(nn.Conv2d(channels, channels, kernel_size=3, padding=1))
        reduced_channels = max(1, channels // 16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            layer_init(nn.Conv2d(channels, reduced_channels, kernel_size=1)),
            nn.GELU(),
            layer_init(nn.Conv2d(reduced_channels, channels, kernel_size=1)),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = F.gelu(self.conv1(x))
        out = self.conv2(out)
        w = self.se(out)
        out = out * w
        return F.gelu(out + x)

class Agent(nn.Module):
    def __init__(self, mapsize=16 * 16, lstm_hidden=384, lstm_layers=3):
        super(Agent, self).__init__()
        self.mapsize = mapsize
        self.network = nn.Sequential(
            layer_init(nn.Conv2d(73, 64, kernel_size=3, stride=2, padding=1)),
            nn.GELU(),
            ResBlock(64),
            layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)),
            nn.GELU(),
            ResBlock(64),
            layer_init(nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)),
            nn.GELU(),
            ResBlock(64),
            nn.Flatten(),
            layer_init(nn.Linear(64 * 8 * 8, 256)),
            nn.ReLU()
        )
        self.z_embedding = nn.Embedding(num_embeddings=7, embedding_dim=8)
        self.scalar_encoder  = ScalarFeatureEncoder(11)
        #Scalarencoder()
        self.actor = layer_init(nn.Linear(256+32+ 85, self.mapsize * envsT.action_plane_space.nvec.sum()), std=0.01)
        self.critic = layer_init(nn.Linear(256+32+ 85, 1), std=1)

    def forward(self, x,sc,z):
        sc_feat = self.scalar_encoder(sc)

        z_embed = self.z_embedding(z[:, :10])

        z_feat =  torch.cat((z_embed.view(z.size(0), -1),z[:, 10:]), dim=1)

        obs_feat = self.network(x.permute((0, 3, 1, 2)))
        feat = torch.cat([obs_feat, sc_feat, z_feat], dim=-1)
        return feat


    def bc_loss_fn(self, obs,sc, expert_actions,z):

        B, H, W, C = obs.shape
        features = self.forward(obs,sc,z)
        flat = self.actor(features)
        grid_logits = flat.view(-1, envsT.action_plane_space.nvec.sum())
        split_logits = torch.split(grid_logits, envsT.action_plane_space.nvec.tolist(), dim=1)
        invalid_action_masks = torch.ones((B * H*W, envsT.action_plane_space.nvec.sum()+1), dtype=torch.bool).to(device)
        invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
        split_invalid_action_masks = torch.split(
            invalid_action_masks[:, 1:], envsT.action_plane_space.nvec.tolist(), dim=1
        )

        multi_categoricals = [
            CategoricalMasked(logits=l, masks=m)
            for (l, m) in zip(split_logits, split_invalid_action_masks)
        ]

        expert_actions = expert_actions.view(-1, expert_actions.shape[-1]).T



        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(expert_actions, multi_categoricals)])

        bc_loss = - logprob.sum() / (B * H * W)
        return bc_loss



    def get_action(self, x,Sc,z ,action=None, invalid_action_masks=None, envs=None):
        logits = self.actor(self.forward(x,Sc,z))
        grid_logits = logits.view(-1, envsT.action_plane_space.nvec.sum())
        split_logits = torch.split(grid_logits, envsT.action_plane_space.nvec.tolist(), dim=1)
        if action is None:
            all_arrays = []
            for i in range(envsT.num_envs):
                arr = np.array(envsT.debug_matrix_mask(i))
                all_arrays.append(arr)
            mask = np.stack(all_arrays)
            invalid_action_masks = torch.tensor(mask).to(device)
            invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
            split_invalid_action_masks = torch.split(
                invalid_action_masks[:, 1:], envsT.action_plane_space.nvec.tolist(), dim=1
            )
            multi_categoricals = [
                CategoricalMasked(logits=l, masks=m)
                for (l, m) in zip(split_logits, split_invalid_action_masks)
            ]
            action = torch.stack([c.sample() for c in multi_categoricals])
        else:
            # … expert‐action logprob branch (same as before) …
            invalid_action_masks = invalid_action_masks.reshape(-1, invalid_action_masks.shape[-1])
            action = action.view(-1, action.shape[-1]).T
            split_invalid_action_masks = torch.split(
                invalid_action_masks[:, 1:], envsT.action_plane_space.nvec.tolist(), dim=1
            )
            multi_categoricals = [
                CategoricalMasked(logits=l, masks=m)
                for (l, m) in zip(split_logits, split_invalid_action_masks)
            ]

        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        num_predicted_parameters = len(envsT.action_plane_space.nvec)


        logprob = logprob.T.reshape(-1, 256, num_predicted_parameters)
        entropy = entropy.T.reshape(-1, 256, num_predicted_parameters)

        action = action.T.reshape(-1, 256, num_predicted_parameters)

        invalid_action_masks = invalid_action_masks.view(-1, 256, envsT.action_plane_space.nvec.sum() + 1)

        return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks

    def get_value(self, x,Sc,z):
        return self.critic(self.forward(x,Sc,z))


class ReplayDataset(IterableDataset):
    def __init__(self, data_dir, shuffle=True):
        self.data_dir = data_dir
        self.files = sorted([f for f in os.listdir(data_dir) if f.endswith(".pt")])
        self.shuffle = shuffle

    def __iter__(self):
        files = self.files.copy()
        if self.shuffle:
            random.shuffle(files)

        for fname in files:
            try:
                data = torch.load(os.path.join(self.data_dir, fname), map_location="cpu")
                obs = data["obs"]
                act = data["act"]
                sc = data["sc"]
                z = data["z"]
                for i in range(len(obs)):
                    yield obs[i], act[i], sc[i], z[i]
            except Exception as e:
                print(f"Error loading {fname}: {e}")

agent = Agent().to(device)

BCtraining = True
newdata = False
start_epoch = 1
if args.prod_mode and wandb.run.resumed:
    start_epoch = run.summary.get('charts/BCepoch') + 1
    ckpt_path = f"models/{experiment_name}/agent.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    agent.load_state_dict(torch.load(ckpt_path, map_location=device))
    agent.eval()
    print(f"resumed at epoch {start_epoch}")
z= []
if BCtraining:
    if newdata:

        T_max = 2048
        N = 1

        opponents =[[microrts_ai.workerRushAI,microrts_ai.coacAI],[microrts_ai.coacAI,microrts_ai.workerRushAI], [microrts_ai.coacAI,microrts_ai.passiveAI], [microrts_ai.coacAI,microrts_ai.lightRushAI], [microrts_ai.coacAI,microrts_ai.coacAI], [microrts_ai.coacAI,microrts_ai.mayari], [microrts_ai.coacAI,microrts_ai.randomAI], [microrts_ai.coacAI,microrts_ai.randomBiasedAI],[microrts_ai.mayari,microrts_ai.workerRushAI], [microrts_ai.mayari,microrts_ai.passiveAI], [microrts_ai.mayari,microrts_ai.lightRushAI], [microrts_ai.mayari,microrts_ai.coacAI], [microrts_ai.mayari,microrts_ai.mayari], [microrts_ai.mayari,microrts_ai.randomAI], [microrts_ai.mayari,microrts_ai.randomBiasedAI]]
        num_runs = [200,200,50,300,150,300,200,200,300,50,300,300,150,200,200]


        for l,x in enumerate(opponents):
            # ai_list = [microrts_ai.coacAI if i % 2 == 0 else opponents[(i // 2) % len(opponents)] for i in range(24)]

            ai_list = x

            env = MicroRTSBotGridVecEnv(
                max_steps=2048,
                ais=ai_list,  # [microrts_ai.coacAI for _ in range(2)],  # CoacAI plays as second player
                map_paths=["maps/16x16/basesWorkers16x16.xml"],
                reference_indexes=[0]  # ,1,2,3,4,5,6,7,8,9,10,11]
            )

            envT = MicroRTSSpaceTransformbot(env)


            obs_batch, _, res = envT.reset()

            Latenttorch = torch.zeros((0, 15), dtype=torch.int32)
            obsten = torch.zeros((0, 16, 16, 73), dtype=torch.int32)
            actten = torch.zeros((0, 256, 7), dtype=torch.int32)
            scten = torch.zeros((0, 11), dtype=torch.int32)

            for ep in range(num_runs[l]):
                buildorder = torch.zeros(10, dtype=torch.int32) # 0 empty,1 base, 2 barrack, 3 worker, 4 light,5  heavy,6 ranged
                custat = torch.zeros(5, dtype=torch.int32) # 0 barrack, 1 worker, 2 light,3  heavy,4 ranged
                dones = np.array([False] * N)
                step = 0

                obs_arr = []
                act_arr = []



                while not dones.all():
                    acti = []


                    obs_arr.append(obs_batch)

                    scten = torch.cat([ scten, getScalarFeatures(obs_batch, res,1)], dim=0)

                    obs_batch, mask, dones, action,res = envT.step("")

                    #time.sleep(0.1)



                    arr = np.zeros((256, 7), dtype=np.int64)


                    for j in range(len(action[0])):
                        if action[0][j][1] == 4:

                            zer = (buildorder == 0).nonzero()
                            custat[action[0][j][6] -2] += 1

                            if len(zer) >0:
                                buildorder[zer[0].item()] = action[0][j][6]

                        arr[action[0][j][0]] = action[0][j][1:]





                    act_arr.append(arr)



                    step += 1



                print("Collecting Data ep: ",ep)
                obsten = torch.cat((obsten,torch.tensor(np.array(obs_arr)).squeeze(1)), dim=0)
                actten = torch.cat((actten,torch.tensor(np.array(act_arr))), dim=0)
                Latenttorch = torch.cat((Latenttorch,torch.cat((buildorder.repeat(len(obs_arr),1),custat.repeat(len(obs_arr),1)), dim=1)), dim=0)
                if (ep + 1) % 50 == 0:
                    torch.save({
                        'obs': obsten,
                        'act': actten,
                        'sc': scten,
                        'z': Latenttorch,
                    }, f"replays/replay_{l}_up_to_ep{ep + 1}.pt")


                    Latenttorch = torch.zeros((0, 15), dtype=torch.int32)
                    obsten = torch.zeros((0, 16, 16, 73), dtype=torch.int32)
                    actten = torch.zeros((0, 256, 7), dtype=torch.int32)
                    scten = torch.zeros((0, 11), dtype=torch.int32)

                #z.append(torch.cat((buildorder,custat), dim=0))
        envT.close()
        env.close()
        #z_np = torch.stack(z).numpy()

        #np.savetxt("zFeatures.txt", z_np, fmt="%.6f")

    replay_dataset = ReplayDataset("replays/")
    train_loader = DataLoader(replay_dataset, batch_size=1024)


    optimizer = optim.Adam(agent.parameters(), lr=1e-4, eps=1e-5)

    num_epochs = start_epoch+1000

    for epoch in range(start_epoch ,num_epochs):

        train_loss_sum = 0.0
        train_count = 0
        lr = 1e-3 * (0.8 ** ((epoch - start_epoch) / 10))
        optimizer.param_groups[0]['lr'] = lr
        for obs_batch, expert_actions, sc, latentv in train_loader:
            sc_t = sc.to(device)
            obs_t = obs_batch.to(device)
            expert_t = expert_actions.to(device)
            latentv_t = latentv.to(device)

            loss = agent.bc_loss_fn(obs_t, sc_t, expert_t, latentv_t)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()

            train_loss_sum += loss.item() * obs_t.size(0)
            train_count += obs_t.size(0)

        avg_train_loss = train_loss_sum / train_count

        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.16f}")

        writer.add_scalar("charts/BCepoch", epoch)
        writer.add_scalar("charts/BCLossTrain",avg_train_loss , epoch)

        if args.prod_mode and epoch % CHECKPOINT_FREQUENCY == 0:
            print("checkpoint")
            os.makedirs(f"models/{experiment_name}", exist_ok=True)
            torch.save(agent.state_dict(), f"models/{experiment_name}/agent.pt")

        # Early stopping check



optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
if args.anneal_lr:
    # https://github.com/openai/baselines/blob/ea25b9e8b234e6ee1bca43083f8f3cf974143998/baselines/ppo2/defaults.py#L20
    lr = lambda f: f * args.learning_rate

# ALGO Logic: Storage for epoch data
mapsize = 16 * 16

action_space_shape = (mapsize, envsT.action_plane_space.shape[0])
invalid_action_shape = (mapsize, envsT.action_plane_space.nvec.sum() + 1)


obs = torch.zeros((args.num_steps, args.num_envs) + envsT.single_observation_space.shape).to(device)
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
ob,mas,res = envsT.reset()

next_obs = torch.Tensor(ob).to(device)
next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size
ScFeatures = torch.zeros((args.num_steps, args.num_envs, 11)).to(device)
zFeatures = torch.zeros((args.num_steps, args.num_envs, 15), dtype=torch.long).to(device)
## CRASH AND RESUME LOGIC:
starting_update = 1
import inspect
from jpype.types import JArray, JInt

'''if args.prod_mode and wandb.run.resumed:
    starting_update = run.summary.get('charts/update') + 1
    global_step = starting_update * args.batch_size
    ckpt_path = f"models/{experiment_name}/agent.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    agent.load_state_dict(torch.load(ckpt_path, map_location=device))
    agent.eval()
    print(f"resumed at update {starting_update}")'''
import time

with open("zFeatures.txt", "r") as f:
    lines = f.readlines()

z = [torch.tensor(list(map(float, line.strip().split()))) for line in lines]

buildorder = torch.zeros(args.num_envs,10, dtype=torch.long).to(device)  # 0 empty,1 base, 2 barrack, 3 worker, 4 light,5  heavy,6 ranged
custat = torch.zeros(args.num_envs,5, dtype=torch.long).to(device)  # 0 barrack, 1 worker, 2 light,3  heavy,4 ranged
new_z = random.sample(z, k=args.num_envs)
new_z = torch.stack(new_z).to(device)
for update in range(starting_update, num_updates + 1):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow

    for step in range(0, args.num_steps):



        """indices = (next_done == 1).nonzero(as_tuple=True)[0]
        for i in range(indices.size(0)):
            print("new for:",indices[i])
            custat[indices[i].item()].zero_()
            buildorder[indices[i].item()].zero_()
            zz = random.sample(z, k=1)
            new_z[indices[i].item()] = zz[0]"""

        envsT.render("human")


        global_step += 1 * args.num_envs

        obs[step] = next_obs
        ScFeatures[step] = getScalarFeatures(next_obs,res,args.num_envs)
        zFeatures[step] = new_z
        dones[step] = next_done

        # ALGO LOGIC: put action logic here
        with torch.no_grad():

            values[step] = agent.get_value(obs[step],ScFeatures[step],zFeatures[step]).flatten()
            action, logproba, _, invalid_action_masks[step] = agent.get_action(obs[step],ScFeatures[step],zFeatures[step], envs=envsT)




        actions[step] = action

        logprobs[step] = logproba


        real_action = torch.cat([
            torch.stack(
                [torch.arange(0, mapsize).to(device) for i in range(envsT.num_envs)
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

            next_obs, denserew, winlossrew,scorerew , ds, infos,res = envsT.step(java_valid_actions)
            next_obs = envsT._from_microrts_obs(next_obs)
            next_obs = torch.Tensor(next_obs).to(device)




        except Exception as e:
            e.printStackTrace()
            raise

        '''winloss = min(0.01, 6.72222222e-9 * global_step)
        densereward = max(0, 0.8 + (-4.44444444e-9 * global_step))

        if global_step < 100000000:
            scorew = 0.19 + 1.754e-8 * global_step
        else:
            scorew = 0.5 - 1.33e-8 * global_step'''
        densereward = 0.3
        winloss = 1
        scorew = 0.2

        """for i in range(args.num_envs):

            matching_indices = (actions[step][i][:, 0] == 4).nonzero(as_tuple=True)[0]

            for j in range(len(java_valid_actions[i])):


                if java_valid_actions[i][j][1] == 4:

                    zer = (buildorder[i] == 0).nonzero()
                    custat[i][java_valid_actions[i][j][6] - 2] += 1

                    if len(zer) > 0:
                        buildorder[i][zer[0].item()] = java_valid_actions[i][j][6]

            if random.random() < 0.25:
                edit_dist = compute_edit_distance(
                    zFeatures[step][i][:10], buildorder[i]
                )
                pseudo_r1 = -edit_dist
            else:
                pseudo_r1 = 0.0

            # Pseudo-reward 2: Hamming on cumulative stats
            if random.random() < 0.25:
                ham = (zFeatures[step][i][10:] != custat[i]).float().mean().item()
                pseudo_r2 = -ham
            else:
                pseudo_r2 = 0.0

            rewards_winloss[step][i] = rewards_winloss[step][i] + pseudo_r2 + pseudo_r1 *0.1"""

        rewards_dense[step] = torch.Tensor(denserew* densereward).to(device)
        rewards_winloss[step] = torch.Tensor(winlossrew*winloss).to(device)
        rewards_score[step] = torch.Tensor(scorerew*scorew).to(device)
        next_done = torch.Tensor(ds).to(device)

        for info in infos:

            if 'episode' in info.keys():

                #print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                for key in info['microrts_stats']:
                    writer.add_scalar(f"charts/episode_reward/{key}", info['microrts_stats'][key], global_step)

                break


    with torch.no_grad():

        last_value = agent.get_value(next_obs.to(device),ScFeatures[step],zFeatures[step]).reshape(1, -1)
        advantages = torch.zeros_like(rewards_winloss).to(device)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = last_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards_dense[t] + rewards_winloss[t] + rewards_score[t]   + args.gamma * nextvalues * nextnonterminal - values[t] #
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values
    b_z = zFeatures.reshape(-1,15 )
    b_Sc = ScFeatures.reshape(-1, 11)
    b_obs = obs.reshape((-1,) + envsT.single_observation_space.shape)
    b_actions = actions.reshape((-1,)+action_space_shape)
    b_logprobs = logprobs.reshape(-1)
    b_advantages = advantages.reshape(-1)
    b_returns = returns.reshape(-1)
    b_values = values.reshape(-1)
    b_invalid_action_masks = invalid_action_masks.reshape((-1,) + invalid_action_shape)


    inds = np.arange(args.batch_size, )
    for i_epoch_pi in range(args.update_epochs):
        np.random.shuffle(inds)

        for start in range(0, args.batch_size, args.minibatch_size):
            end = start + args.minibatch_size
            minibatch_ind = inds[start:end]
            mb_advantages = b_advantages[minibatch_ind]


            if args.norm_adv:
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
            _, newlogproba, entropy, _ = agent.get_action(
                b_obs[minibatch_ind],
                b_Sc[minibatch_ind],
                b_z[minibatch_ind],
                b_actions.long()[minibatch_ind],
                b_invalid_action_masks[minibatch_ind],
                envsT)
            ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

            approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            new_values = agent.get_value(b_obs[minibatch_ind],b_Sc[minibatch_ind],b_z[minibatch_ind]).view(-1)
            if args.clip_vloss:
                v_loss_unclipped = ((new_values - b_returns[minibatch_ind]) ** 2)
                v_clipped = b_values[minibatch_ind] + torch.clamp(new_values - b_values[minibatch_ind], -args.clip_coef,
                                                                  args.clip_coef)
                v_loss_clipped = (v_clipped - b_returns[minibatch_ind]) ** 2
                v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                v_loss = 0.5 * v_loss_max.mean()
            else:
                v_loss = 0.5 * ((new_values - b_returns[minibatch_ind]) ** 2)

            loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
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

envsT.close()
writer.close()
