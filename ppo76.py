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
#from gym_microrts.envs.vec_env2 import MicroRTSGridModeVecEnv
from gym_microrts.envs.microrts_vec_env import  MicroRTSGridModeVecEnv
from gym_microrts.envs.microrts_bot_vec_env import MicroRTSBotGridVecEnv
from gym_microrts import microrts_ai
from gym.wrappers import TimeLimit, Monitor

from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
import time
import random
import os
from stable_baselines3.common.vec_env import VecEnvWrapper, VecVideoRecorder
from microrts_space_transform import MicroRTSSpaceTransform
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

#experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
experiment_name = "MicrortsDefeatCoacAIShaped-v3__ppo3__1__1748315682"
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
opponents = [microrts_ai.coacAI, microrts_ai.coacAI, microrts_ai.lightRushAI, microrts_ai.workerRushAI, microrts_ai.mayari]
ai_list = [microrts_ai.coacAI if i % 2 == 0 else opponents[(i // 2) % len(opponents)] for i in range(12)]
env = MicroRTSBotGridVecEnv(
    max_steps=2000,
    ais=[microrts_ai.coacAI for _ in range(2)],  # CoacAI plays as second player
    map_paths=["maps/16x16/basesWorkers16x16.xml"],
    reference_indexes = [0]#,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
)

envT = MicroRTSSpaceTransform(env)


random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic
envs = MicroRTSGridModeVecEnv(
    num_selfplay_envs=args.num_selfplay_envs,
    num_bot_envs=args.num_bot_envs,
    max_steps=2000,
    render_theme=1,
    ai2s=[microrts_ai.coacAI for _ in range(args.num_bot_envs)],
         #[microrts_ai.mayari for _ in range(2)] + \
         #[microrts_ai.tiamat for _ in range(2)] + \
         #[microrts_ai.lightRushAI for _ in range(2)]+\
         #[microrts_ai.workerRushAI for _ in range(2)],
    map_paths=["maps/16x16/basesWorkers16x16.xml"],
    reward_weight=np.array([10.0,1.0, 1.0, 0.2, 1.0,4.0, 5.25, 6.0, 0])
)
envsT = MicroRTSSpaceTransform(envs)

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




def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        x = self.gelu(self.conv1(x))
        x = self.conv2(x)
        return self.gelu(x + residual)


class Agent(nn.Module):
    def __init__(self, in_channels=22, base_channels=64, num_actions=3):
        super().__init__()

        # Initial conv: [B, in_channels=22, 16, 16] → [B, 64, 16, 16]
        self.in_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        # ↓ “16×16 → 8×8”
        self.enc1 = ResBlock(base_channels)
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=2, stride=2),  # 16→8
            nn.GELU()
        )

        # ↓ “8×8 → 4×4”
        self.enc2 = ResBlock(base_channels)
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=2, stride=2),  # 8→4
            nn.GELU()
        )

        # ↓ “4×4 → 2×2”
        self.enc3 = ResBlock(base_channels)
        self.down3 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=2, stride=2),  # 4→2
            nn.GELU()
        )

        # Bottom of U‐Net: 2×2
        self.enc4 = ResBlock(base_channels)

        # ↑ “2×2 → 4×4”
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2),  # 2→4
            nn.GELU()
        )
        self.dec3 = ResBlock(base_channels)

        # ↑ “4×4 → 8×8”
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2),  # 4→8
            nn.GELU()
        )
        self.dec2 = ResBlock(base_channels)

        # ↑ “8×8 → 16×16”
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels, base_channels, kernel_size=2, stride=2),  # 8→16
            nn.GELU()
        )
        self.dec1 = ResBlock(base_channels)

        self.actor_head = nn.Conv2d(base_channels, envs.action_space.nvec[1:].sum(), kernel_size=1)
        self.value_heads = nn.Conv2d(base_channels, 3, kernel_size=1)

    def forward(self, x):
        # x is [B, 22, 16, 16]
        x1 = self.in_conv(x)           # → [B, 64, 16, 16]
        x1_res = self.enc1(x1)         # → [B, 64, 16, 16]

        x2 = self.down1(x1_res)        # → [B, 64,  8,  8]
        x2_res = self.enc2(x2)         # → [B, 64,  8,  8]

        x3 = self.down2(x2_res)        # → [B, 64,  4,  4]
        x3_res = self.enc3(x3)         # → [B, 64,  4,  4]

        x4 = self.down3(x3_res)        # → [B, 64,  2,  2]
        x4_res = self.enc4(x4)         # → [B, 64,  2,  2]

        u3 = self.up3(x4_res) + x3_res  # → [B, 64,  4,  4]
        u3 = self.dec3(u3)             # → [B, 64,  4,  4]

        u2 = self.up2(u3) + x2_res      # → [B, 64,  8,  8]
        u2 = self.dec2(u2)             # → [B, 64,  8,  8]

        u1 = self.up1(u2) + x1_res      # → [B, 64, 16, 16]
        u1 = self.dec1(u1)             # → [B, 64, 16, 16]

        actor_out = self.actor_head(u1)    # → [B, num_actions, 16, 16]
        value_out = self.value_heads(u1)   # → [B,        3, 16, 16]

        return actor_out, value_out

    def bc_loss_fn(self, obs, expert_actions, invalid_action_masks=None):
        B, C, H, W = obs.shape
        HW = H * W

        # --- 1. Forward pass → [B, sum_nvec, H, W] ---
        logits_all, _ = self.forward(obs)
        sum_nvec = logits_all.size(1)  # = envs.action_space.nvec[1:].sum()

        # --- 2. Flatten spatial dims → [B*H*W, sum_nvec] ---
        # Reshape [B, sum_nvec, H, W] → [B, sum_nvec, H*W], then permute → [sum_nvec, B*H*W], finally transpose → [B*H*W, sum_nvec].
        flat_logits = logits_all.reshape(B, sum_nvec, HW).permute(1, 0, 2).reshape(sum_nvec, B * HW).permute(1,
                                                                                                             0).contiguous()
        # Now flat_logits.shape == [B*H*W, sum_nvec]

        # --- 3. Flatten invalid_action_masks: [B, H*W, action_dim+1] → [B*H*W, action_dim+1], then drop first column → [B*H*W, sum_nvec] ---
        flat_mask = invalid_action_masks.reshape(B * HW, invalid_action_masks.size(-1))  # [B*H*W, action_dim+1]
        flat_mask = flat_mask[:, 1:]  # drop index 0; now [B*H*W, sum_nvec]

        # --- 4. Flatten expert_actions: [B, H*W, action_dim] → [B*H*W, action_dim] ---
        flat_expert = expert_actions.reshape(B * HW, -1)  # [B*H*W, action_dim]

        # --- 5. Get split sizes from envs.action_space.nvec[1:] and verify sum matches ---
        nvec_list = [int(x) for x in envs.action_space.nvec[1:]]
        assert sum(nvec_list) == sum_nvec, (
            f"Sum of envs.action_space.nvec[1:] = {sum(nvec_list)} must equal sum_nvec = {sum_nvec}"
        )

        # --- 6. Split flat_logits and flat_mask along dim=1 into chunks of sizes nvec_list ---
        # Each logits_chunks[j].shape == [B*H*W, n_j]
        logits_chunks = torch.split(flat_logits, nvec_list, dim=1)
        mask_chunks = torch.split(flat_mask, nvec_list, dim=1)

        total_logprob = 0.0
        total_count = 0
        action_dim = flat_expert.size(1)
        assert action_dim == len(nvec_list), (
            f"expert_actions last dim ({action_dim}) != len(nvec_list) ({len(nvec_list)})"
        )

        # --- 7. For each sub-action j, mask out invalid logits (out-of-place) and compute log_prob ---
        for j, n_j in enumerate(nvec_list):
            logits_j = logits_chunks[j]  # [B*H*W, n_j]
            mask_j = mask_chunks[j].bool()  # [B*H*W, n_j]
            labels_j = flat_expert[:, j]  # [B*H*W]

            # Out-of-place masking: replace invalid logits with -inf
            logits_j = logits_j.masked_fill(~mask_j, float('-inf'))

            # CategoricalMasked will re‐mask internally, but we already set invalid logits to -inf
            dist_j = CategoricalMasked(logits=logits_j, masks=mask_j)
            logprob_j = dist_j.log_prob(labels_j)  # → [B*H*W]

            total_logprob += logprob_j.sum()
            total_count += (B * HW)

        # --- 8. BC loss = average negative log_protover all sub‐actions & all cells ---
        bc_loss = - (total_logprob / total_count)
        return bc_loss

    def get_action(self, obs_seq, action=None, invalid_action_masks=None, envs=None):
        logits,_ = self.forward(obs_seq)  # [B, T, H]
        # [B, T, mapsize * action_dim]
        grid_logits = logits.view(-1, envs.action_space.nvec[1:].sum())
        split_logits = torch.split(grid_logits, envs.action_space.nvec[1:].tolist(), dim=1)

        # Build invalid_action_masks if needed (same as before)…
        if action is None:

            all_arrays = []
            for i in range(envs.num_envs):
                arr = np.array(envs.debug_matrix_mask(i))
                all_arrays.append(arr)
            mask = np.stack(all_arrays)
            invalid_action_masks = torch.tensor(mask).to(device)
            invalid_action_masks = invalid_action_masks.view(-1, invalid_action_masks.shape[-1])
            split_invalid_action_masks = torch.split(
                invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(), dim=1
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
                invalid_action_masks[:, 1:], envs.action_space.nvec[1:].tolist(), dim=1
            )
            multi_categoricals = [
                CategoricalMasked(logits=l, masks=m)
                for (l, m) in zip(split_logits, split_invalid_action_masks)
            ]

        # Compute log‐prob and entropy exactly as before
        logprob = torch.stack([cat.log_prob(a) for a, cat in zip(action, multi_categoricals)])
        entropy = torch.stack([cat.entropy() for cat in multi_categoricals])
        num_params = len(envs.action_space.nvec) - 1

        logprob = logprob.T.view(-1, 256, num_params)  # -> [N, 256, #params]
        entropy = entropy.T.view(-1, 256, num_params)
        action = action.T.view(-1, 256, num_params)

        invalid_action_masks = invalid_action_masks.view(-1, 256, envs.action_space.nvec[1:].sum() + 1)

        return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks

    def get_value(self, x):
        _, value_out = self.forward(x)

        return value_out.mean(dim=[1, 2, 3])


from torch.utils.data import DataLoader, TensorDataset
T_max = 2000
N = 1
envT.reset()
obs_batch = env.reset()
all_obs = []
all_act= []
all_mask = []
for ep in range(1):

    dones = np.array([False] * N)
    step = 0
    dones_list = [[] for _ in range(N)]
    obs_arr = []
    act_arr = []
    mask_arr = []
    while not dones.all() and step < T_max:
        other_masks = []
        acti = []

        for i in range(N):
            other_masks.append(env.debug_matrix_mask(i).reshape(256, 79))

        obs_batch, mask, rew, rew2, dones, infos, action = env.step("")
        obss_transformed = envT._from_microrts_obs(obs_batch)

        for i in range(N):
            arr = np.zeros((256, 7), dtype=float)
            if len(action[i*2]) >= 0:
                for j in range(len(action[i * 2])):
                    arr[action[i * 2][j][0]] = action[i * 2][j][1:]
                acti.append(arr)
                dones_list[i].append(dones[i])

#wenn N>1 dann aus for schleife
                obs_arr.append(obss_transformed)
                act_arr.append(acti)
                mask_arr.append(other_masks)

        step += 1

    all_obs.append(obs_arr)
    all_act.append(act_arr)
    all_mask.append(mask_arr)
    print("Collecting Data ep: ",ep)

obs_tensor = torch.tensor(np.concatenate(all_obs,axis=0), device=device).squeeze(1)
action_tensor = torch.tensor(np.concatenate(all_act,axis=0), device=device).squeeze(1)
invalid_mask_flat = torch.tensor(np.concatenate(all_mask,axis=0), dtype=torch.float32, device=device).squeeze(1)
obs_tensor = obs_tensor.reshape(-1,16,16,22).to(device)
action_tensor = action_tensor.reshape(-1,256,7).to(device)    # [B, 64, 64]
invalid_mask_flat = invalid_mask_flat.reshape(-1,256,79).to(device)
#invalid_mask_flat[:,:,:] = 1

dataset = TensorDataset(obs_tensor, action_tensor, invalid_mask_flat)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True, drop_last=True)

agent = Agent().to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)


for epoch in range(10):
    for obs, expert_actions, invalid_action_masks in dataloader:

        obs = obs.to(device)                          # [B, 74, 64, 64]
        expert_actions = expert_actions.to(device)    # [B, 64, 64]
        invalid_action_masks = invalid_action_masks.to(device)  # [B, 64, 64, A]

        loss = agent.bc_loss_fn(obs.permute(0,3,1,2), expert_actions, invalid_action_masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"[BC epoch {epoch}]  BC_loss = {loss.item():.6f}")


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
ob,mas = envs.reset()


obss_batch = envsT._from_microrts_obs(ob)

next_obs = torch.Tensor(obss_batch).to(device)
next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size



## CRASH AND RESUME LOGIC:
starting_update = 1
import inspect
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
import time

for update in range(starting_update, num_updates + 1):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = lr(frac)
        optimizer.param_groups[0]['lr'] = lrnow

    for step in range(0, args.num_steps):

        envs.render()

        global_step += 1 * args.num_envs

        obs[step] = next_obs

        dones[step] = next_done

        # ALGO LOGIC: put action logic here
        with torch.no_grad():
            obsT = obs[step].permute(0, 3, 1, 2)
            values[step] = agent.get_value(obsT).flatten()
            action, logproba, _, invalid_action_masks[step] = agent.get_action(obsT, envs=envs)

        actions[step] = action


        logprobs[step] = logproba


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

            nobs, denserew, winlossrew,scorerew , ds, infos = envs.step(java_valid_actions)
            next_obs = envsT._from_microrts_obs(nobs)
            next_obs = torch.Tensor(next_obs).to(device)




        except Exception as e:
            e.printStackTrace()
            raise

        winloss = min(0.01, 6.72222222e-9 * global_step)
        densereward = max(0, 0.8 + (-4.44444444e-9 * global_step))

        if global_step < 170000000:
            scorew = 0.19 + 1.754e-8 * global_step
        else:
            scorew = 0.5 - 1.33e-8 * global_step


        rewards_dense[step] = torch.Tensor(denserew* densereward).to(device)
        rewards_winloss[step] = torch.Tensor(winlossrew*winloss).to(device)
        rewards_score[step] = torch.Tensor(scorerew*scorew).to(device)
        next_done = torch.Tensor(ds).to(device)


        for info in infos:
           if 'episode' in info.keys():
                print(f"global_step={global_step}, episode_reward={info['episode']['r']}")
                writer.add_scalar("charts/episode_reward", info['episode']['r'], global_step)
                for key in info['microrts_stats']:
                    writer.add_scalar(f"charts/episode_reward/{key}", info['microrts_stats'][key], global_step)

                break


    with torch.no_grad():

        last_value = agent.get_value(next_obs.to(device).permute(0, 3, 1, 2)).reshape(1, -1)
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


    b_obs = obs.reshape((-1,) + envs.observation_space.shape)
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
            obsT = b_obs[minibatch_ind].permute(0, 3, 1, 2)
            _, newlogproba, entropy, _ = agent.get_action(
                obsT,

                b_actions.long()[minibatch_ind],
                b_invalid_action_masks[minibatch_ind],
                envs)
            ratio = (newlogproba - b_logprobs[minibatch_ind]).exp()

            approx_kl = (b_logprobs[minibatch_ind] - newlogproba).mean()

            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
            pg_loss = torch.max(pg_loss1, pg_loss2).mean()
            entropy_loss = entropy.mean()

            new_values = agent.get_value(obsT).view(-1)
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

envs.close()
writer.close()
