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
    independent_noise_sampling: jdc.Static[bool] = True
    use_global_variance: jdc.Static[bool] = False
    temp_func_type: jdc.Static[Literal["log", "cbrt", "std"]] = "std"
    resampling_alpha_k: float = 0.3
    resampling_alpha_min: float = 0.0001
    f_x_forward: jdc.Static[bool] = True
    num_generated_actions: jdc.Static[int] = 2
    num_epsilon_samples: jdc.Static[int] = 8

    use_hard_resampling: jdc.Static[bool] = True

    w_v_loss: float = 1.0
    learning_rate_p: float = 3e-4
    learning_rate_v: float = 3e-4
    loop_v: jdc.Static[int] = 1

    flow_steps: jdc.Static[int] = 10
    output_mode: jdc.Static[Literal["u", "u_but_supervise_as_eps"]] = "u_but_supervise_as_eps"
    timestep_embed_dim: jdc.Static[int] = 8
    feather_std: float = 0.0
    policy_mlp_output_scale: float = 0.25
    sde_sigma: float = 0.0

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

    # 👑 保留：主脚本算 Epoch 需要用到
    @property
    def iterations_per_env(self) -> int:
        return (self.num_minibatches * self.batch_size * self.unroll_length) // self.num_envs


@jdc.pytree_dataclass
class DGPOFMParams:
    policy: MlpWeights
    value: MlpWeights


@jdc.pytree_dataclass
class DGPOFMActionInfo:
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
        )

    # 👑 修复 1：移除外挂 prng 参数，直接从 self 中取
    def _step_minibatch(self, transitions: DGPOFMTransition) -> tuple[DGPOFMState, dict[str, Array]]:
        # 👑 从自身状态裂变出新的 key
        prng_gen, prng_pol, next_prng = jax.random.split(self.prng, 3)
        cfg, sch = self.config, self.get_schedule()

        N = transitions.obs.shape[0] * transitions.obs.shape[1]
        obs_dim, act_dim, t_dim = self.env.observation_size, self.env.action_size, cfg.timestep_embed_dim

        obs_flat = ((
                            transitions.obs - self.obs_stats.mean) / self.obs_stats.std if cfg.normalize_observations else transitions.obs).reshape(
            (N, obs_dim))
        target_qs = transitions.action_info.target_qs.reshape((N, 1))

        K = cfg.num_generated_actions
        obs_b_gen = jnp.broadcast_to(obs_flat[:, None, :], (N, K, obs_dim))

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

        gen_acts, _ = jax.lax.scan(gen_step, jax.random.normal(prng_gen, (N, K, act_dim)),
                                   (fast_t_current, fast_t_next))
        pool_actions = jnp.concatenate([transitions.action.reshape((N, 1, act_dim)), gen_acts], axis=1)

        def value_inner_step(carry, _):
            v_p, v_opt = carry

            def v_loss_fn(p):
                total = self._compute_value_loss(p, obs_flat, transitions.action.reshape((N, act_dim)),
                                                 transitions.truncation.reshape((N, 1)), target_qs)
                return total, total

            (v_loss_val, _), v_grads = jax.value_and_grad(v_loss_fn, has_aux=True)(v_p)
            v_updates, next_v_opt = self.opt_value.update(v_grads, v_opt, v_p)

            return (optax.apply_updates(v_p, v_updates), next_v_opt), {"v_loss/total": v_loss_val}

        (new_v_params, new_v_opt_state), extra_v_metrics = jax.lax.scan(
            value_inner_step,
            (self.params.value, self.opt_state_value),
            None, length=cfg.loop_v
        )

        # 👑 修改前：
        # probs, fresh_metrics = self._compute_fresh_weights(new_v_params, obs_flat, pool_actions)

        # 👑 修改后：把 target_qs 传进去，这是算 TD Error 底噪的必需品！
        probs, fresh_metrics = self._compute_fresh_weights(new_v_params, obs_flat, pool_actions, target_qs)

        def policy_loss_fn(p_params):
            M = cfg.num_epsilon_samples
            p_idx, p_eps, p_t, p_acc = jax.random.split(prng_pol, 4)
            logits = jnp.log(probs + 1e-8)

            # 👑 1/K 无偏密度修正核心 (抛弃硬截断吸收池)
            # ==========================================
            K = cfg.num_generated_actions
            accept_threshold = 1.0 / K  # 核心数学配平系数

            if cfg.independent_noise_sampling:
                # 模式 A: 8个噪声各自抽签
                sampled_indices = jax.random.categorical(p_idx, jnp.broadcast_to(logits[:, None, :], (N, M, K + 1)),
                                                         axis=-1)
                a_target = jnp.take_along_axis(pool_actions[:, None, :, :], sampled_indices[..., None, None],
                                               axis=2).squeeze(2)
                rand_vals = jax.random.uniform(p_acc, (N, M))

                is_real = (sampled_indices == 0)
                is_fake = (sampled_indices > 0)

                # 👑 假动作面临 1/K 的命运审判
                is_fake_accepted = is_fake & (rand_vals < accept_threshold)
                valid_mask = (is_real | is_fake_accepted).astype(jnp.float32)

            else:
                # 模式 B: 一发入魂模式
                sampled_indices = jax.random.categorical(p_idx, logits, axis=-1)[:, None]  # (N, 1)
                a_target = jnp.broadcast_to(pool_actions[jnp.arange(N), sampled_indices[:, 0]][:, None, :],
                                            (N, M, act_dim))

                # 只生成 (N, 1) 的随机数
                rand_vals = jax.random.uniform(p_acc, (N, 1))

                is_real = (sampled_indices == 0)  # (N, 1)
                is_fake = (sampled_indices > 0)  # (N, 1)

                # 👑 假动作面临 1/K 的命运审判
                is_fake_accepted = is_fake & (rand_vals < accept_threshold)  # (N, 1)

                # 此时 valid_mask_single 的形状是纯正的 (N, 1)
                valid_mask_single = (is_real | is_fake_accepted).astype(jnp.float32)

                # 广播为 (N, M)
                valid_mask = jnp.broadcast_to(valid_mask_single, (N, M))

            # ==========================================
            # 📈 统一计算全新漏斗监控指标
            # ==========================================
            total_fake_winners = jnp.maximum(1.0, jnp.sum(is_fake.astype(jnp.float32)))
            actual_fake_accept_rate = jnp.sum(is_fake_accepted.astype(jnp.float32)) / total_fake_winners

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
                # Softmax 阶段的胜率博弈
                "q_guided/real_win_ratio": jnp.mean(is_real.astype(jnp.float32)),
                "q_guided/fake_win_ratio": jnp.mean(is_fake.astype(jnp.float32)),
                # 无偏估计的数学铁证 (应该死死钉在 1/K 附近)
                "q_guided/fake_accept_ratio": actual_fake_accept_rate,
                # 系统有效吞吐量
                "q_guided/overall_valid_ratio": jnp.mean(valid_mask),
                "q_guided/accept_threshold": jnp.array(accept_threshold, dtype=jnp.float32)
            }

        (p_loss, p_metrics), p_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(self.params.policy)
        p_updates, new_p_opt = self.opt_policy.update(p_grads, self.opt_state_policy, self.params.policy)

        # 👑 去掉了坑人的省略号，正确放入各更新后的参数，并且放入 next_prng
        new_state = jdc.replace(self,
                                params=DGPOFMParams(optax.apply_updates(self.params.policy, p_updates), new_v_params),
                                opt_state_policy=new_p_opt,
                                opt_state_value=new_v_opt_state,
                                steps=self.steps + 1,
                                prng=next_prng
                                )

        final_metrics = {**{k: v[-1] for k, v in extra_v_metrics.items()}, **p_metrics, **fresh_metrics}
        return new_state, final_metrics

    def _compute_fresh_weights(self, value_params, obs_norm, pool_actions, target_qs) -> tuple[Array, dict[str, Array]]:
        N, K_plus_1, act_dim = pool_actions.shape
        flat_obs = obs_norm.reshape((N, self.env.observation_size))
        obs_pool_b = jnp.broadcast_to(flat_obs[:, None, :], (N, K_plus_1, flat_obs.shape[-1]))
        q_pool, _ = networks.value_mlp_fwd_with_features(value_params,
                                                         jnp.concatenate([obs_pool_b, pool_actions], axis=-1))
        q_pool = jax.lax.stop_gradient(q_pool)

        # ==========================================
        # 👑 治本核心：计算宏观底噪屏蔽墙 (Magnitude/Error Gating)
        # ==========================================
        q_real = q_pool[:, 0:1] # 真动作的预测 Q 值
        td_error_abs = jnp.abs(target_qs - q_real) # 绝对 TD 误差

        # 设定底噪比例：假设绝对 Q 值的 2%，或者绝对 TD 误差的 10% 都是不可信的噪声
        noise_floor_ratio = 0.02
        td_penalty_ratio = 0.10

        # 算出宏观噪音温度墙 (取 Q值规模 与 TD误差 带来的噪音最大值)
        macro_noise_alpha = jnp.maximum(
            jnp.abs(q_real) * noise_floor_ratio,
            td_error_abs * td_penalty_ratio
        )

        # ==========================================
        # 原有逻辑：计算微观状态级方差
        # ==========================================
        if self.config.use_global_variance:
            x_var = jax.lax.stop_gradient(jnp.var(q_pool))
        else:
            x_var = jax.lax.stop_gradient(jnp.var(q_pool, axis=-1, keepdims=True))

        if self.config.temp_func_type == "log":
            f_x = jnp.log1p(x_var)
        elif self.config.temp_func_type == "cbrt":
            f_x = jnp.power(x_var + 1e-8, 1.0 / 3.0)
        else:
            f_x = jnp.sqrt(x_var + 1e-8)

        # 算出基础 Alpha
        if self.config.f_x_forward:
            base_alpha = jnp.maximum(self.config.resampling_alpha_min, self.config.resampling_alpha_k * f_x)
        else:
            base_alpha = self.config.resampling_alpha_min / (1 + self.config.resampling_alpha_k * f_x)

        # ==========================================
        # 👑 终极融合：微观方差与宏观底噪取最大值！
        # ==========================================
        # 如果底层噪音已经很大了，决不允许微观 alpha 小于底噪水平，防止 Softmax 放大幻觉！
        alpha = jnp.maximum(base_alpha, macro_noise_alpha)

        logits = (q_pool - jnp.max(q_pool, axis=-1, keepdims=True)) / alpha
        pool_probs = jax.nn.softmax(logits, axis=-1)

        return jax.lax.stop_gradient(pool_probs), {
            "q_guided/q_real_mean": jnp.mean(q_pool[:, 0]),
            "q_guided/prob_real_mean": jnp.mean(pool_probs[:, 0]),
            "q_guided/alpha_mean": jnp.mean(alpha),
            "q_guided/alpha_max": jnp.max(alpha),
            "q_guided/alpha_min": jnp.min(alpha),
            "q_guided/q_var_mean": jnp.mean(x_var),
            "q_guided/f_x_mean": jnp.mean(f_x),
            # 👑 加一个监控指标，看看这堵墙什么时候发挥作用
            "q_guided/macro_noise_floor_mean": jnp.mean(macro_noise_alpha)
        }

    def _compute_value_loss(self, value_params, obs_norm, actions, truncation, target_qs):
        concat_inputs = jnp.concatenate([obs_norm, actions], axis=-1)
        q_pred, _ = networks.value_mlp_fwd_with_features(value_params, concat_inputs)
        q_pred = q_pred.reshape(target_qs.shape)

        v_error = (target_qs - q_pred) * (1 - truncation)
        mse_loss = jnp.mean(v_error ** 2)

        total_v_loss = mse_loss * self.config.value_loss_coeff * self.config.w_v_loss
        return total_v_loss

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

        obs_norm = (
                           transitions.obs - state.obs_stats.mean) / state.obs_stats.std if config.normalize_observations else transitions.obs

        concat_inputs = jnp.concatenate([obs_norm, transitions.action], axis=-1)
        q_pred, _ = networks.value_mlp_fwd_with_features(state.params.value, concat_inputs)
        q_pred = jax.lax.stop_gradient(q_pred)

        bootstrap_obs = transitions.next_obs[-1:, :, :]
        if config.normalize_observations:
            bootstrap_obs = (bootstrap_obs - state.obs_stats.mean) / state.obs_stats.std

        # training_step 初始化时不用折叠 prng，直接交给后续的方法
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

        new_action_info = jdc.replace(transitions.action_info, target_qs=target_qs)
        new_transitions = jdc.replace(transitions, action_info=new_action_info)

        # 👑 修复 3：不再使用 partial 绑定固定的 prng
        def step_batch(state: DGPOFMState, _):
            step_prng = jax.random.fold_in(state.prng, state.steps)
            state, metrics = jax.lax.scan(
                DGPOFMState._step_minibatch,  # JAX scan 会自动把 state 当作 carry 传入
                init=state,
                xs=new_transitions.prepare_minibatches(step_prng, config.num_minibatches, config.batch_size),
            )
            return state, metrics

        state, metrics = jax.lax.scan(step_batch, init=state, length=config.num_updates_per_batch)
        return state, metrics