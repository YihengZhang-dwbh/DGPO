from __future__ import annotations

from functools import partial
from typing import Literal

import dataclasses
import jax
import jax_dataclasses as jdc
import mujoco_playground as mjp
import optax
from jax import Array
from jax import numpy as jnp

from flow_policy.networks import MlpWeights
from . import math_utils, networks, rollouts


@jdc.pytree_dataclass
class DGPOFMConfig:
    # --- 全新 Q-Guided 生成控制核心 ---
    independent_noise_sampling: jdc.Static[bool] = False  # 👑 新增：是否让 8 个噪声独立竞争
    use_global_variance: jdc.Static[bool] = False
    temp_func_type: jdc.Static[Literal["log", "cbrt", "std"]] = "log"
    resampling_alpha_k: float = 0.1
    resampling_alpha_min: float = 0.3
    num_generated_actions: jdc.Static[int] = 1  # 👑 现在的 K 固定了，不会再有形状Bug
    num_epsilon_samples: jdc.Static[int] = 8


    # --- 👑 接受率 p_accept 控制 ---
    p_accept_mode: jdc.Static[Literal["fixed", "linear", "cyclical"]] = "fixed"
    p_fixed_val: float = 0.5  # fixed 模式下的固定值
    p_min: float = 0.0  # 最小值 (用于 linear/cyclical)
    p_max: float = 1.0  # 最大值 (用于 linear/cyclical)
    p_cycles: jdc.Static[int] = 5  # cyclical 模式的循环次数

    # 原有退火参数保留作为 linear 模式参考
    start_decay: float = 0.33
    end_decay: float = 0.8

    use_hard_resampling: jdc.Static[bool] = True

    w_v_loss: float = 1.0
    learning_rate_p: float = 3e-4
    learning_rate_v: float = 3e-4
    loop_v: jdc.Static[int] = 1

    use_hinge_cql: jdc.Static[bool] = True
    cql_decay_mode: jdc.Static[Literal["none", "linear", "cosine", "exponential", "inverse", "auto"]] = "auto"
    cql_init_weight: float = 0.1
    cql_final_weight: float = 0.0001
    cql_decay_ratio: float = 0.5

    cql_target_margin: float = 10
    cql_alpha_lr: float = 3e-4
    cql_alpha_kp: float = 0.05
    cql_clip_alpha: jdc.Static[bool] = False

    flow_steps: jdc.Static[int] = 10
    output_mode: jdc.Static[Literal["u", "u_but_supervise_as_eps"]] = "u_but_supervise_as_eps"
    timestep_embed_dim: jdc.Static[int] = 8
    n_samples_per_action: jdc.Static[int] = 8
    average_losses_before_exp: jdc.Static[bool] = False
    discretize_t_for_training: jdc.Static[bool] = True
    feather_std: float = 0.0
    policy_mlp_output_scale: float = 0.25

    loss_mode: jdc.Static[Literal["dgpo", "denoising_mdp"]] = "dgpo"
    final_steps_only: jdc.Static[bool] = False
    sde_sigma: float = 0.0
    clipping_epsilon: float = 0.05

    batch_size: jdc.Static[int] = 1024
    discounting: float = 0.995
    episode_length: int = 1000
    normalize_observations: jdc.Static[bool] = True
    num_envs: jdc.Static[int] = 2048
    num_evals: jdc.Static[int] = 30
    num_minibatches: jdc.Static[int] = 32
    num_timesteps: jdc.Static[int] = 180000000
    num_updates_per_batch: jdc.Static[int] = 16
    reward_scaling: float = 10.0
    unroll_length: jdc.Static[int] = 30

    gae_lambda: float = 0.95
    normalize_advantage: jdc.Static[bool] = True
    value_loss_coeff: float = 0.25

    def __post_init__(self) -> None:
        assert self.timestep_embed_dim % 2 == 0

    @property
    def iterations_per_env(self) -> int:
        return (self.num_minibatches * self.batch_size * self.unroll_length) // self.num_envs


@jdc.pytree_dataclass
class DGPOFMParams:
    policy: MlpWeights
    value: MlpWeights


@jdc.pytree_dataclass
class DGPOFMActionInfo:
    # 👑 使用 default_factory 绕过 Python 的默认值检查
    target_qs: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))


@jdc.pytree_dataclass
class FlowSchedule:
    t_current: Array
    t_next: Array


DGPOFMTransition = rollouts.TransitionStruct[DGPOFMActionInfo]


@jdc.pytree_dataclass
class DGPOFMState:
    env: jdc.Static[mjp.MjxEnv]
    config: DGPOFMConfig
    params: DGPOFMParams
    obs_stats: math_utils.RunningStats
    opt_policy: jdc.Static[optax.GradientTransformation]
    opt_value: jdc.Static[optax.GradientTransformation]
    opt_state_policy: optax.OptState
    opt_state_value: optax.OptState
    log_cql_weight: Array
    prng: Array
    steps: Array

    @staticmethod
    def init(prng: Array, env: jdc.Static[mjp.MjxEnv], config: DGPOFMConfig) -> DGPOFMState:
        obs_size = env.observation_size
        action_size = env.action_size
        assert isinstance(obs_size, int)

        prng0, prng1, prng2 = jax.random.split(prng, num=3)
        actor_net = networks.mlp_init(prng0,
                                      (obs_size + action_size + config.timestep_embed_dim, 32, 32, 32, 32, action_size))
        critic_net = networks.mlp_init(prng1, (obs_size + action_size, 256, 256, 256, 256, 256, 1))
        network_params = DGPOFMParams(actor_net, critic_net)
        opt_policy = optax.adam(config.learning_rate_p)
        opt_value = optax.adam(config.learning_rate_v)
        return DGPOFMState(
            env=env, config=config, params=network_params,
            obs_stats=math_utils.RunningStats.init((obs_size,)),
            opt_policy=opt_policy, opt_value=opt_value,
            opt_state_policy=opt_policy.init(network_params.policy),
            opt_state_value=opt_value.init(network_params.value),
            prng=prng2, steps=jnp.zeros((), dtype=jnp.int32),
            log_cql_weight=jnp.log(jnp.array(config.cql_init_weight, dtype=jnp.float32)),
        )

    def _step_minibatch(self, transitions: DGPOFMTransition, prng: Array) -> tuple[DGPOFMState, dict[str, Array]]:
        prng_gen, prng_pol = jax.random.split(prng)
        cfg, sch = self.config, self.get_schedule()

        # 1. 维度与数据对齐
        N = transitions.obs.shape[0] * transitions.obs.shape[1]
        obs_dim, act_dim, t_dim = self.env.observation_size, self.env.action_size, cfg.timestep_embed_dim

        obs_flat = ((
                                transitions.obs - self.obs_stats.mean) / self.obs_stats.std if cfg.normalize_observations else transitions.obs).reshape(
            (N, obs_dim))
        target_qs = transitions.action_info.target_qs.reshape((N, 1))

        # 2. p_accept 计算
        total_updates = (cfg.num_timesteps // (
                    cfg.iterations_per_env * cfg.num_envs)) * cfg.num_updates_per_batch * cfg.num_minibatches
        progress = jnp.minimum(1.0, self.steps / jnp.maximum(1.0, total_updates))

        if cfg.p_accept_mode == "fixed":
            p_accept = cfg.p_fixed_val
        elif cfg.p_accept_mode == "linear":
            p_accept = cfg.p_min + progress * (cfg.p_max - cfg.p_min)
        else:  # cyclical
            cycle_val = jnp.cos(2.0 * jnp.pi * cfg.p_cycles * progress)
            p_accept = cfg.p_min + 0.5 * (cfg.p_max - cfg.p_min) * (1.0 - cycle_val)

        # 3. 动作池生成 (大池子: K可以放心设为 3, 7, 15 等)
        # 3. 动作池生成 (K=8，但我们让它坐火箭)
        K = cfg.num_generated_actions
        obs_b_gen = jnp.broadcast_to(obs_flat[:, None, :], (N, K, obs_dim))

        # 👑 性能黑客：单独定义一个只有 3 步（甚至 2 步）的粗糙时间表
        # 这会让 K=8 的生成速度直接飙升 3 倍以上！
        fast_flow_steps = 3
        fast_full_t = jnp.linspace(1.0, 0.0, fast_flow_steps + 1)
        fast_t_current, fast_t_next = fast_full_t[:-1], fast_full_t[1:]

        def gen_step(x, t_tup):
            t_curr, t_next = t_tup
            t_embed_raw = self.embed_timestep(jnp.array([t_curr])[..., None])
            t_embed = jnp.broadcast_to(t_embed_raw[:, None, :], (N, K, t_dim))
            vel = networks.flow_mlp_fwd(jax.lax.stop_gradient(self.params.policy), obs_b_gen, x,
                                        t_embed) * cfg.policy_mlp_output_scale
            return x + (t_next - t_curr) * vel, None

        # 用快速时间表进行生成
        gen_acts, _ = jax.lax.scan(gen_step, jax.random.normal(prng_gen, (N, K, act_dim)),
                                   (fast_t_current, fast_t_next))
        pool_actions = jnp.concatenate([transitions.action.reshape((N, 1, act_dim)), gen_acts], axis=1)

        # 4. Q 网络更新 (Critic 对大池子进行全局打分)
        def value_inner_step(carry, _):
            v_p, v_opt, log_a = carry
            alpha = jax.lax.stop_gradient(jnp.exp(log_a))

            def v_loss_fn(p):
                total, penalty = self._compute_value_loss(p, obs_flat, transitions.action.reshape((N, act_dim)),
                                                          transitions.truncation.reshape((N, 1)), target_qs,
                                                          pool_actions, alpha)
                return total, penalty

            (v_loss_val, current_penalty), v_grads = jax.value_and_grad(v_loss_fn, has_aux=True)(v_p)
            v_updates, next_v_opt = self.opt_value.update(v_grads, v_opt, v_p)
            next_log_alpha = jnp.clip(log_a + cfg.cql_alpha_lr * (current_penalty - cfg.cql_target_margin),
                                      jnp.log(cfg.cql_final_weight),
                                      jnp.log(cfg.cql_init_weight)) if cfg.cql_decay_mode == "auto" else log_a
            return (optax.apply_updates(v_p, v_updates), next_v_opt, next_log_alpha), {
                "v_loss/total": v_loss_val, "v_loss/cql_penalty": current_penalty, "v_loss/current_cql_weight": alpha
            }

        (new_v_params, new_v_opt_state, new_la), extra_v_metrics = jax.lax.scan(value_inner_step, (self.params.value,
                                                                                                   self.opt_state_value,
                                                                                                   self.log_cql_weight),
                                                                                None, length=cfg.loop_v)

        # 5. 策略更新 👑 (核心：解耦评估与 1v1 目标采样)
        probs, _ = self._compute_fresh_weights(new_v_params, obs_flat, pool_actions,
                                               extra_v_metrics["v_loss/total"][-1])

        # 此时你可以放心地把 K 开到 4 或者 8
        def policy_loss_fn(p_params):
            M = cfg.num_epsilon_samples
            p_idx, p_eps, p_t, p_acc = jax.random.split(prng_pol, 4)
            logits = jnp.log(probs + 1e-8)

            # 👑 普适化核心：在 K 个假动作中（索引 1 到 K），找出概率最大的那 1 个作为挑战者
            # probs[:, 1:] 是所有假动作的概率，argmax 找到最大值的相对索引，+1 映射回真实索引
            challenger_idx = jnp.argmax(probs[:, 1:], axis=-1) + 1  # 形状: (N,)
            # # 👑 你的神来之笔：在 1 到 K 的假动作中完全随机抽取 1 个作为合法挑战者
            # challenger_idx = jax.random.randint(p_idx, (N,), minval=1, maxval=K + 1)

            if cfg.independent_noise_sampling:
                # 噪声在 K+1 个动作中自由抽签
                sampled_indices = jax.random.categorical(p_idx, jnp.broadcast_to(logits[:, None, :], (N, M, K + 1)),
                                                         axis=-1)
                a_target = jnp.take_along_axis(pool_actions[:, None, :, :], sampled_indices[..., None, None],
                                               axis=2).squeeze(2)
                rand_vals = jax.random.uniform(p_acc, (N, M))

                # 只有真动作 (0) 和最强假动作 (challenger_idx) 才有资格参与更新
                is_real = (sampled_indices == 0)
                is_active_fake = (sampled_indices == challenger_idx[:, None]) & (rand_vals < p_accept)
                valid_mask = (is_real | is_active_fake).astype(jnp.float32)

                # 剩下的 K-1 个假动作全部沦为吸收态 (Dummy)
                is_dummy = (sampled_indices > 0) & (sampled_indices != challenger_idx[:, None])
                dummy_absorption_ratio = jnp.mean(is_dummy.astype(jnp.float32))

            else:
                # 一发入魂模式
                sampled_indices = jax.random.categorical(p_idx, logits, axis=-1)[:, None]
                a_target = jnp.broadcast_to(pool_actions[jnp.arange(N), sampled_indices[:, 0]][:, None, :],
                                            (N, M, act_dim))
                rand_vals = jnp.broadcast_to(jax.random.uniform(p_acc, (N, 1)), (N, M))

                is_real = (sampled_indices == 0)
                is_active_fake = (sampled_indices == challenger_idx[:, None]) & (rand_vals < p_accept)
                valid_mask_single = (is_real | is_active_fake).astype(jnp.float32)
                valid_mask = jnp.broadcast_to(valid_mask_single[:, None], (N, M))

                is_dummy = (sampled_indices > 0) & (sampled_indices != challenger_idx[:, None])
                dummy_absorption_ratio = jnp.mean(is_dummy.astype(jnp.float32))

            # --- Flow Matching 拟合逻辑保持不变 ---
            eps = jax.random.normal(p_eps, (N, M, act_dim))
            t_idx = jax.random.randint(p_t, (N, M, 1), 0, cfg.flow_steps)
            t = sch.t_current[t_idx]
            x_t = t * eps + (1.0 - t) * a_target

            obs_b_p = jnp.broadcast_to(obs_flat[:, None, :], (N, M, obs_dim))
            t_embed = self.embed_timestep(t)
            vel = networks.flow_mlp_fwd(p_params, obs_b_p, x_t, t_embed) * cfg.policy_mlp_output_scale

            if cfg.output_mode == "u_but_supervise_as_eps":
                err = jnp.sum((eps - ((x_t - t * vel) + vel)) ** 2, axis=-1)
            else:
                err = jnp.sum((vel - (eps - a_target)) ** 2, axis=-1)

            loss = jnp.sum(err * valid_mask) / jnp.maximum(1.0, jnp.sum(valid_mask))

            return loss, {
                "policy_loss": loss,
                "q_guided/p_accept": p_accept,
                "q_guided/real_win_ratio": jnp.mean((sampled_indices == 0).astype(jnp.float32)),
                "q_guided/dummy_absorption_ratio": dummy_absorption_ratio  # 监控 K-1 个占位符吸走了多少火力
            }

        (p_loss, p_metrics), p_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(self.params.policy)
        p_updates, new_p_opt = self.opt_policy.update(p_grads, self.opt_state_policy, self.params.policy)

        # 6. 状态组装
        new_state = jdc.replace(self,
                                params=DGPOFMParams(optax.apply_updates(self.params.policy, p_updates), new_v_params),
                                opt_state_policy=new_p_opt, opt_state_value=new_v_opt_state, log_cql_weight=new_la,
                                steps=self.steps + 1
                                )
        final_metrics = {**{k: v[-1] for k, v in extra_v_metrics.items()}, **p_metrics}
        return new_state, final_metrics

    def _compute_fresh_weights(self, value_params, obs_norm, pool_actions, final_v_loss) -> tuple[
        Array, dict[str, Array]]:
        N, K_plus_1, act_dim = pool_actions.shape
        flat_obs = obs_norm.reshape((N, self.env.observation_size))
        obs_pool_b = jnp.broadcast_to(flat_obs[:, None, :], (N, K_plus_1, flat_obs.shape[-1]))
        q_pool, _ = networks.value_mlp_fwd_with_features(value_params,
                                                         jnp.concatenate([obs_pool_b, pool_actions], axis=-1))
        q_pool = jax.lax.stop_gradient(q_pool)

        # q_pool 的形状是 (N, 3)，包含了真动作、Max假动作、Min假动作的得分

        # 👑 1. 核心开关：全局宏观方差 vs 局部状态级方差
        if self.config.use_global_variance:
            # 全局方差：把整个 q_pool 当作一个整体，算出一个纯标量 (Scalar)
            # 它代表当前整个 Batch 在宏观环境下的 Q 值跨度
            x_var = jax.lax.stop_gradient(jnp.var(q_pool))
        else:
            # 局部方差：沿着动作维度 (axis=-1) 计算，形状变为 (N, 1)
            # 它为当前 Batch 里的每一个独立状态 (State) 量身定制方差
            x_var = jax.lax.stop_gradient(jnp.var(q_pool, axis=-1, keepdims=True))

        # 👑 2. 你的核心映射函数 f(x)
        # config.temp_func_type 可以是 "log", "cbrt", 或者 "sqrt"
        if self.config.temp_func_type == "log":
            # 终极防爆盾：f(x) = ln(1+x)
            f_x = jnp.log1p(x_var)
        elif self.config.temp_func_type == "cbrt":
            # 次线性退火：f(x) = x^(1/3)
            f_x = jnp.power(x_var + 1e-8, 1.0 / 3.0)
        else:
            # 尺度对齐 (标准差)：f(x) = x^(1/2)
            f_x = jnp.sqrt(x_var + 1e-8)

        # 👑 3. 算出动态 alpha (这里的 x_var 无论是标量还是 (N, 1) 向量，JAX 都会自动完美广播)
        alpha = jnp.maximum(self.config.resampling_alpha_min, f_x * self.config.resampling_alpha_k)

        # 👑 4. 计算最终的 Softmax 概率
        logits = (q_pool - jnp.max(q_pool, axis=-1, keepdims=True)) / alpha
        pool_probs = jax.nn.softmax(logits, axis=-1)

        return jax.lax.stop_gradient(pool_probs), {"q_guided/q_real_mean": jnp.mean(q_pool[:, 0]),
                                                   "q_guided/prob_real_mean": jnp.mean(pool_probs[:, 0]),
                                                   "q_guided/alpha_mean": jnp.mean(alpha)}

    def _compute_targets(self, transitions: DGPOFMTransition, obs_norm: Array, prng: Array) -> tuple[
        Array, Array, Array, dict[str, Array]]:
        metrics = dict[str, Array]()
        T, B, obs_dim = obs_norm.shape
        act_dim = self.env.action_size
        N = T * B

        prng_boot, prng_gen, prng_eval = jax.random.split(prng, 3)

        concat_inputs = jnp.concatenate([obs_norm, transitions.action], axis=-1)
        q_pred, h_s = networks.value_mlp_fwd_with_features(self.params.value, concat_inputs)
        q_pred = jax.lax.stop_gradient(q_pred)

        bootstrap_obs = transitions.next_obs[-1:, :, :]
        if self.config.normalize_observations:
            bootstrap_obs = (bootstrap_obs - self.obs_stats.mean) / self.obs_stats.std

        def boot_step_fn(x, t_tuple):
            t_curr, t_next = t_tuple
            t_embed_raw = self.embed_timestep(jnp.array([t_curr])[..., None])  # (1, 8)
            t_embed = jnp.broadcast_to(t_embed_raw[:, None, :], (1, B, self.config.timestep_embed_dim))
            vel = networks.flow_mlp_fwd(self.params.policy, bootstrap_obs, x,
                                        t_embed) * self.config.policy_mlp_output_scale
            return x + (t_next - t_curr) * vel, None

        boot_noise = jax.random.normal(prng_boot, (1, B, act_dim))
        schedule = self.get_schedule()
        bootstrap_act, _ = jax.lax.scan(boot_step_fn, boot_noise, (schedule.t_current, schedule.t_next))
        bootstrap_concat = jnp.concatenate([bootstrap_obs, bootstrap_act], axis=-1)
        bootstrap_q, _ = networks.value_mlp_fwd_with_features(self.params.value, bootstrap_concat)
        bootstrap_q = jax.lax.stop_gradient(bootstrap_q)

        gae_qs, _ = jax.lax.stop_gradient(
            rollouts.compute_gae(
                truncation=transitions.truncation,
                discount=transitions.discount * self.config.discounting,
                rewards=transitions.reward * self.config.reward_scaling,
                values=q_pred,
                bootstrap_value=bootstrap_q,
                gae_lambda=self.config.gae_lambda,
            )
        )

        K = self.config.num_generated_actions
        flat_obs = obs_norm.reshape((N, obs_dim))
        flat_acts_real = transitions.action.reshape((N, 1, act_dim))

        x_t = jax.random.normal(prng_gen, (N, K, act_dim))
        obs_b = jnp.broadcast_to(flat_obs[:, None, :], (N, K, obs_dim))

        def gen_step_fn(x, t_tuple):
            t_curr, t_next = t_tuple
            t_embed_raw = self.embed_timestep(jnp.array([t_curr])[..., None])
            t_embed = jnp.broadcast_to(t_embed_raw[:, None, :], (N, K, self.config.timestep_embed_dim))
            p_params = jax.lax.stop_gradient(self.params.policy)
            vel = networks.flow_mlp_fwd(p_params, obs_b, x, t_embed) * self.config.policy_mlp_output_scale
            return x + (t_next - t_curr) * vel, None

        generated_acts, _ = jax.lax.scan(gen_step_fn, x_t, (schedule.t_current, schedule.t_next))
        pool_actions = jnp.concatenate([flat_acts_real, generated_acts], axis=1)

        obs_pool_b = jnp.broadcast_to(flat_obs[:, None, :], (N, K + 1, obs_dim))
        concat_pool = jnp.concatenate([obs_pool_b, pool_actions], axis=-1)

        q_pool, _ = networks.value_mlp_fwd_with_features(self.params.value, concat_pool)
        q_pool = jax.lax.stop_gradient(q_pool)

        alpha = self.config.resampling_alpha_min
        logits = (q_pool - jnp.max(q_pool, axis=-1, keepdims=True)) / alpha
        pool_probs = jax.nn.softmax(logits, axis=-1)

        metrics["q_guided/q_real_mean"] = jnp.mean(q_pool[:, 0])
        metrics["q_guided/q_generated_mean"] = jnp.mean(q_pool[:, 1:])
        metrics["q_guided/prob_real_mean"] = jnp.mean(pool_probs[:, 0])

        return jax.lax.stop_gradient(pool_actions), jax.lax.stop_gradient(pool_probs), gae_qs, metrics

    def _compute_policy_loss(self, policy_params, obs_norm, actions_pool, weights_pool, prng, p_accept):
        N, K_plus_1, act_dim = actions_pool.shape
        M = self.config.num_epsilon_samples
        flat_obs = obs_norm.reshape((N, obs_norm.shape[-1]))

        if self.config.use_hard_resampling:
            prng_idx, prng_eps, prng_t, prng_accept = jax.random.split(prng, 4)
            logits = jnp.log(weights_pool + 1e-8)

            if self.config.independent_noise_sampling:
                # ==========================================
                # 👑 模式 A：独立采样 (8 个噪声分别竞争)
                # ==========================================
                # 将概率分布广播到 (N, M, K+1)
                logits_b = jnp.broadcast_to(logits[:, None, :], (N, M, K_plus_1))
                # 采样得到 (N, M) 的索引
                sampled_indices = jax.random.categorical(prng_idx, logits_b, axis=-1)

                # 取出对应的动作 (N, M, act_dim)
                a_target = jnp.take_along_axis(
                    actions_pool[:, None, :, :],
                    sampled_indices[:, :, None, None],
                    axis=2
                ).squeeze(2)

                # 退火拒绝逻辑 (逐个噪声判定)
                rand_vals = jax.random.uniform(prng_accept, (N, M))
                is_real = (sampled_indices == 0)
                is_fake_accepted = (sampled_indices > 0) & (rand_vals < p_accept)
                valid_mask = (is_real | is_fake_accepted).astype(jnp.float32)  # (N, M)

            else:
                # ==========================================
                # 👑 模式 B：一发入魂 (代码 4 的稳定模式)
                # ==========================================
                sampled_indices = jax.random.categorical(prng_idx, logits, axis=-1)  # (N,)
                sampled_actions = actions_pool[jnp.arange(N), sampled_indices]
                a_target = jnp.broadcast_to(sampled_actions[:, None, :], (N, M, act_dim))

                # 组级退火拒绝 (一个状态 8 个噪声共生死)
                rand_vals = jax.random.uniform(prng_accept, (N,))
                is_real = (sampled_indices == 0)
                is_fake_accepted = (sampled_indices > 0) & (rand_vals < p_accept)

                valid_mask_single = (is_real | is_fake_accepted).astype(jnp.float32)
                valid_mask = jnp.broadcast_to(valid_mask_single[:, None], (N, M))

            # --- 公共计算部分 ---
            eps = jax.random.normal(prng_eps, (N, M, act_dim))
            t_idx = jax.random.randint(prng_t, (N, M, 1), 0, self.config.flow_steps)
            t = self.get_schedule().t_current[t_idx]

            obs_b = jnp.broadcast_to(flat_obs[:, None, :], (N, M, flat_obs.shape[-1]))
            x_t = t * eps + (1.0 - t) * a_target
            t_embed = self.embed_timestep(t)

            vel_pred = networks.flow_mlp_fwd(policy_params, obs_b, x_t, t_embed) * self.config.policy_mlp_output_scale

            if self.config.output_mode == "u_but_supervise_as_eps":
                x1_pred = (x_t - t * vel_pred) + vel_pred
                error_sq = jnp.sum((eps - x1_pred) ** 2, axis=-1)
            else:
                error_sq = jnp.sum((vel_pred - (eps - a_target)) ** 2, axis=-1)

            # 应用静音掩码
            masked_error = error_sq * valid_mask
            policy_loss = jnp.sum(masked_error) / jnp.maximum(1.0, jnp.sum(valid_mask))

            p_metrics = {
                "policy_loss": policy_loss,
                "q_guided/p_accept": p_accept,
                "q_guided/valid_noise_ratio": jnp.mean(valid_mask),
                "q_guided/real_win_ratio": jnp.mean(is_real.astype(jnp.float32))
            }
            return policy_loss, p_metrics

        else:
            # 软加权分支保持你原来的不变
            prng_eps, prng_t = jax.random.split(prng, 2)
            eps = jax.random.normal(prng_eps, (N, K_plus_1, act_dim))
            t_idx = jax.random.randint(prng_t, (N, K_plus_1, 1), 0, self.config.flow_steps)
            t = self.get_schedule().t_current[t_idx]

            x_t = t * eps + (1.0 - t) * actions_pool
            obs_b = jnp.broadcast_to(flat_obs[:, None, :], (N, K_plus_1, flat_obs.shape[-1]))
            t_embed = self.embed_timestep(t)

            vel_pred = networks.flow_mlp_fwd(policy_params, obs_b, x_t, t_embed) * self.config.policy_mlp_output_scale

            if self.config.output_mode == "u_but_supervise_as_eps":
                x1_pred = (x_t - t * vel_pred) + vel_pred
                error_sq = jnp.sum((eps - x1_pred) ** 2, axis=-1)
            else:
                error_sq = jnp.sum((vel_pred - (eps - actions_pool)) ** 2, axis=-1)

            policy_loss = jnp.mean(jnp.sum(weights_pool * error_sq, axis=-1))
            return policy_loss, {"policy_loss": policy_loss}

    def _compute_value_loss(self, value_params, obs_norm, actions, truncation, target_qs, pool_actions,
                            current_cql_weight):
        concat_inputs = jnp.concatenate([obs_norm, actions], axis=-1)
        q_pred, _ = networks.value_mlp_fwd_with_features(value_params, concat_inputs)

        # 👑 终极防御：直接变形为 target_qs 的物理形状 (30, 1024)
        q_pred = q_pred.reshape(target_qs.shape)

        v_error = (target_qs - q_pred) * (1 - truncation)
        mse_loss = jnp.mean(v_error ** 2)

        N, K_plus_1, act_dim = pool_actions.shape
        flat_obs = obs_norm.reshape((N, obs_norm.shape[-1]))
        obs_b = jnp.broadcast_to(flat_obs[:, None, :], (N, K_plus_1, obs_norm.shape[-1]))
        concat_pool = jnp.concatenate([obs_b, pool_actions], axis=-1)
        q_pool_fake, _ = networks.value_mlp_fwd_with_features(value_params, concat_pool)

        # 👑 终极防御：直接变形为标准的 (N, K+1)
        q_pool_fake = q_pool_fake.reshape((N, K_plus_1))

        q_real_sg = jax.lax.stop_gradient(q_pool_fake[:, 0:1])
        q_fake = q_pool_fake[:, 1:]

        if self.config.use_hinge_cql:
            cql_penalty = jnp.mean(jax.nn.relu(q_fake - q_real_sg))
        else:
            cql_penalty = jnp.mean(q_fake - q_real_sg)

        total_v_loss = (
                               mse_loss + current_cql_weight * cql_penalty) * self.config.value_loss_coeff * self.config.w_v_loss
        return total_v_loss, cql_penalty

    def get_schedule(self) -> FlowSchedule:
        full_t_path = jnp.linspace(1.0, 0.0, self.config.flow_steps + 1)
        return FlowSchedule(t_current=full_t_path[:-1], t_next=full_t_path[1:])

    def embed_timestep(self, t: Array) -> Array:
        assert t.shape[-1] == 1
        freqs = 2 ** jnp.arange(self.config.timestep_embed_dim // 2)
        scaled_t = t * freqs
        out = jnp.concatenate([jnp.cos(scaled_t), jnp.sin(scaled_t)], axis=-1)
        assert out.shape == (*t.shape[:-1], self.config.timestep_embed_dim)
        return out

    def sample_action(self, obs: Array, prng: Array, deterministic: bool) -> tuple[Array, DGPOFMActionInfo]:
        obs_norm = (obs - self.obs_stats.mean) / self.obs_stats.std if self.config.normalize_observations else obs
        (*batch_dims, obs_dim) = obs.shape
        assert obs_dim == self.env.observation_size

        def euler_step(carry: Array, inputs: tuple[FlowSchedule, Array]) -> tuple[Array, Array]:
            x_t = carry
            schedule_t, noise = inputs
            dt = schedule_t.t_next - schedule_t.t_current
            velocity = networks.flow_mlp_fwd(
                self.params.policy, obs_norm, x_t,
                jnp.broadcast_to(self.embed_timestep(schedule_t.t_current[None]),
                                 (*batch_dims, self.config.timestep_embed_dim))
            ) * self.config.policy_mlp_output_scale
            return x_t + dt * velocity + self.config.sde_sigma * noise, x_t

        prng_sample, prng_loss, prng_feather, prng_noise = jax.random.split(prng, num=4)
        noise_path = jax.random.normal(prng_noise, (self.config.flow_steps, *batch_dims, self.env.action_size))
        x0, _ = jax.lax.scan(euler_step, jax.random.normal(prng_sample, (*batch_dims, self.env.action_size)),
                             (self.get_schedule(), noise_path))

        if not deterministic:
            x0 = x0 + jax.random.normal(prng_feather, (*batch_dims, self.env.action_size)) * self.config.feather_std
        return x0, DGPOFMActionInfo()

    @jdc.jit
    def training_step(self, transitions: DGPOFMTransition) -> tuple[DGPOFMState, dict[str, Array]]:
        config = self.config
        state = self
        if config.normalize_observations:
            with jdc.copy_and_mutate(state) as state:
                state.obs_stats = state.obs_stats.update(transitions.obs)

        # 1. 预计算所有状态的 obs_norm
        obs_norm = (
                               transitions.obs - state.obs_stats.mean) / state.obs_stats.std if config.normalize_observations else transitions.obs

        # 2. 计算 Q 预测与 Bootstrap Target
        concat_inputs = jnp.concatenate([obs_norm, transitions.action], axis=-1)
        q_pred, _ = networks.value_mlp_fwd_with_features(state.params.value, concat_inputs)
        q_pred = jax.lax.stop_gradient(q_pred)

        bootstrap_obs = transitions.next_obs[-1:, :, :]
        if config.normalize_observations:
            bootstrap_obs = (bootstrap_obs - state.obs_stats.mean) / state.obs_stats.std

        prng_boot = jax.random.fold_in(state.prng, state.steps)

        def boot_step_fn(x, t_tuple):
            t_curr, t_next = t_tuple
            t_embed_raw = state.embed_timestep(jnp.array([t_curr])[..., None])
            t_embed = jnp.broadcast_to(t_embed_raw[:, None, :], (1, bootstrap_obs.shape[1], config.timestep_embed_dim))
            vel = networks.flow_mlp_fwd(state.params.policy, bootstrap_obs, x, t_embed) * config.policy_mlp_output_scale
            return x + (t_next - t_curr) * vel, None

        boot_noise = jax.random.normal(prng_boot, (1, bootstrap_obs.shape[1], state.env.action_size))
        bootstrap_act, _ = jax.lax.scan(boot_step_fn, boot_noise,
                                        (state.get_schedule().t_current, state.get_schedule().t_next))
        bootstrap_q, _ = networks.value_mlp_fwd_with_features(state.params.value,
                                                              jnp.concatenate([bootstrap_obs, bootstrap_act], axis=-1))

        # 3. 计算绝对静态的 GAE Target
        target_qs, _ = jax.lax.stop_gradient(
            rollouts.compute_gae(
                truncation=transitions.truncation,
                discount=transitions.discount * config.discounting,
                rewards=transitions.reward * config.reward_scaling,
                values=q_pred,
                bootstrap_value=jax.lax.stop_gradient(bootstrap_q),
                gae_lambda=config.gae_lambda,
            )
        )

        # 4. 👑 注入 Target
        new_action_info = jdc.replace(transitions.action_info, target_qs=target_qs)
        new_transitions = jdc.replace(transitions, action_info=new_action_info)

        # 5. Minibatch 训练
        def step_batch(state: DGPOFMState, _):
            step_prng = jax.random.fold_in(state.prng, state.steps)
            state, metrics = jax.lax.scan(
                partial(DGPOFMState._step_minibatch, prng=jax.random.fold_in(step_prng, 0)),
                init=state,
                xs=new_transitions.prepare_minibatches(step_prng, config.num_minibatches, config.batch_size),
            )
            return state, metrics

        state, metrics = jax.lax.scan(step_batch, init=state, length=config.num_updates_per_batch)
        return state, metrics