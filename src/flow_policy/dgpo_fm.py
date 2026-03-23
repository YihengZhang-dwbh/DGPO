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
    temp_func_type: jdc.Static[Literal["log", "cbrt", "std", "fixed", "max"]] = "fixed"
    fast_flow_step: jdc.Static[int] = 5
    # ... 其他 Config 参数 ...

    # 👑 概率分配模式控制：经典的 KL 闭式解 (Softmax) vs 全新的线性概率质量重分配
    # prob_allocation_mode: jdc.Static[Literal["kl_softmax", "linear_redistribution"]] = "linear_redistribution"
    prob_allocation_mode: jdc.Static[Literal["kl_softmax", "linear_redistribution"]] = "kl_softmax"
    # 全新非对称线性重分配参数 (l: 惩罚剥夺率, r: 奖励增幅率)
    resampling_l: float = 0.3
    resampling_r: float = 0.4
    # 👑 OOD 盲盒探索率 (基因突变注入)
    # 建议设在 0.05 到 0.15 之间，Cheetahrun 推荐从 0.1 起步
    resampling_ood_p: float = 0.1

    # ...

    # 👑 动作边界处理：加入了无损折叠映射
    action_clip: jdc.Static[Literal["hard", "margin", "tanh", "fold", "scale_clip"]] = "margin"
    clip_margin: float = 1.1
    penalty_coef: float = 0

    # 👑 采样算力分配模式
    sampling_mode: jdc.Static[Literal["absolute_budget", "relative_h_pool"]] = "relative_h_pool"
    h_fakes_in_pool: jdc.Static[int] = 3  # 当模式为 relative_h_pool 时，每次大循环固定的假动作个数

    base_tolerance: float = 1.0
    resampling_alpha_k: float = 0.3
    resampling_alpha_min: float = 0.0001
    f_x_forward: jdc.Static[bool] = True
    num_generated_actions: jdc.Static[int] = 48  # 外层全局生成动作数 (K)
    # 👑 物理显存上限 (决定了极品动作能爽到什么程度)
    # 如果显卡顶得住，拉到 128 甚至 256！
    num_epsilon_samples: jdc.Static[int] = 128

    beta_r: float = 0.9
    beta_v: float = 0.9
    tolerance_r: float = -10
    tolerance_v: float = 10

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
    raw_action: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    # 👑 新增：用于缓存大循环外层计算好的池子和概率，彻底解耦内层
    pool_actions: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    pool_probs: Array = dataclasses.field(default_factory=lambda: jnp.zeros(()))


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

    ema_reward: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    ema_reward_sq: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    ema_v_loss: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.zeros(()))
    ema_v_loss_sq: jnp.ndarray = dataclasses.field(default_factory=lambda: jnp.zeros(()))

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

    def _apply_clip(self, x: Array) -> Array:
        cfg = self.config
        if cfg.action_clip == "hard":
            return jnp.clip(x, -1.0, 1.0)
        elif cfg.action_clip == "margin":
            return jnp.clip(x, -cfg.clip_margin, cfg.clip_margin)
        elif cfg.action_clip == "tanh":
            return cfg.clip_margin * jnp.tanh(x / cfg.clip_margin)
        elif cfg.action_clip == "fold":
            # return jnp.abs(x - 4.0 * jnp.floor((x + 3.0) / (4.0)) + 1.0) - 1.0
            T = 4.0*cfg.clip_margin
            t = cfg.clip_margin
            # return jnp.abs(x+ 3*t - T * jnp.floor((x + 3*t) / T) - 2*t ) - t
            return jnp.abs(x - T * jnp.floor((x + 3*t) / T) + t ) - t
        return x

    def _compute_fresh_weights(self, value_params, obs_norm, pool_actions_raw) -> tuple[Array, dict[str, Array]]:
        N, K_plus_1, act_dim = pool_actions_raw.shape
        flat_obs = obs_norm.reshape((N, self.env.observation_size))
        obs_pool_b = jnp.broadcast_to(flat_obs[:, None, :], (N, K_plus_1, flat_obs.shape[-1]))

        eval_actions = self._apply_clip(pool_actions_raw)

        q_pool, _ = networks.value_mlp_fwd_with_features(
            value_params,
            jnp.concatenate([obs_pool_b, eval_actions], axis=-1)
        )
        q_pool = jax.lax.stop_gradient(q_pool).reshape((N, K_plus_1))

        out_of_bounds = jnp.maximum(jnp.abs(pool_actions_raw) - 1.0, 0.0)
        penalty = self.config.penalty_coef * jnp.sum(jnp.square(out_of_bounds), axis=-1)

        q_pool_penalized = q_pool - penalty
        pool_mean = jnp.mean(q_pool_penalized, axis=-1, keepdims=True)
        adv = q_pool_penalized - pool_mean

        # ==========================================
        # 👑 模式 A：经典的 KL 闭式解 (Softmax 分布)
        # ==========================================
        if self.config.prob_allocation_mode == "kl_softmax":
            if self.config.temp_func_type == "max":
                abs_adv = jnp.abs(adv)
                f_x = jnp.max(abs_adv) if self.config.use_global_variance else jnp.max(abs_adv, axis=-1, keepdims=True)
            else:
                x_var = jnp.var(q_pool_penalized) if self.config.use_global_variance else jnp.var(q_pool_penalized,
                                                                                                  axis=-1,
                                                                                                  keepdims=True)
                if self.config.temp_func_type == "log":
                    f_x = jnp.log1p(x_var)
                elif self.config.temp_func_type == "cbrt":
                    f_x = jnp.power(x_var + 1e-8, 1.0 / 3.0)
                elif self.config.temp_func_type == "std":
                    f_x = jnp.sqrt(x_var + 1e-8)
                else:
                    f_x = 1.0

            if self.config.f_x_forward:
                alpha = jnp.maximum(self.config.resampling_alpha_min, self.config.resampling_alpha_k * f_x)
            else:
                alpha = self.config.resampling_alpha_min / (1 + self.config.resampling_alpha_k * f_x)

            logits = (q_pool_penalized - jnp.max(q_pool_penalized, axis=-1, keepdims=True)) / alpha
            pool_probs = jax.nn.softmax(logits, axis=-1)

            metrics = {
                "q_guided/q_real_mean": jnp.mean(q_pool[:, 0]),
                "q_guided/q_real_penalized_mean": jnp.mean(q_pool_penalized[:, 0]),
                "q_guided/penalty_mean": jnp.mean(penalty),
                "q_guided/prob_real_mean": jnp.mean(pool_probs[:, 0]),
                "q_guided/alpha_mean": jnp.mean(alpha),
                "q_guided/f_x_mean": jnp.mean(f_x),
            }
            return jax.lax.stop_gradient(pool_probs), metrics

        # ==========================================
        # 👑 模式 B：全新的非对称线性概率质量重分配机制 (l, r 独立控制)
        # ==========================================
        elif self.config.prob_allocation_mode == "linear_redistribution":
            # 提取你设计的非对称双参
            l_val = self.config.resampling_l
            r_val = self.config.resampling_r

            p_base = jnp.ones((N, K_plus_1)) / K_plus_1

            is_pos = adv > 0.0
            is_neg = adv <= 0.0

            p_pos = jnp.sum(p_base * is_pos, axis=-1, keepdims=True)
            sum_A_pos = jnp.sum(adv * is_pos, axis=-1, keepdims=True) + 1e-8
            sum_A_neg = jnp.sum(adv * is_neg, axis=-1, keepdims=True) - 1e-8  # 注意：这里是负数

            # --- Case 1: (l+r)*p_+ <= l ---
            new_p_c1_pos = p_base * (1 + r_val) + (adv * is_pos / sum_A_pos) * (l_val - (l_val + r_val) * p_pos)
            new_p_c1_neg = p_base * (1 - l_val)
            p_case1 = jnp.where(is_pos, new_p_c1_pos, new_p_c1_neg)

            # --- Case 2: (l+r)*p_+ > l ---
            new_p_c2_pos = p_base * (1 + r_val)
            # 注意：sum_A_neg 和 adv*is_neg 都是负数，相除为正。减去后面的项意味着进一步削减差动作的概率
            new_p_c2_neg = p_base * (1 - l_val) - (adv * is_neg / sum_A_neg) * ((l_val + r_val) * p_pos - l_val)
            p_case2 = jnp.where(is_pos, new_p_c2_pos, new_p_c2_neg)

            # --- Case 3: p_+ * (1+r) > 1.0 (概率池枯竭，按排位“劫富济贫”) ---
            target_p = p_base * (1 + r_val) * is_pos
            sort_idx = jnp.argsort(-adv, axis=-1)
            sorted_target_p = jnp.take_along_axis(target_p, sort_idx, axis=-1)

            cum_p = jnp.cumsum(sorted_target_p, axis=-1)
            full_alloc_mask = cum_p <= 1.0

            prev_full_alloc_mask = jnp.concatenate([jnp.ones((N, 1), dtype=bool), full_alloc_mask[:, :-1]], axis=-1)
            boundary_mask = prev_full_alloc_mask & ~full_alloc_mask
            prev_cum_p = jnp.concatenate([jnp.zeros((N, 1)), cum_p[:, :-1]], axis=-1)

            sorted_new_p = jnp.where(full_alloc_mask, sorted_target_p, 0.0)
            sorted_new_p = jnp.where(boundary_mask, jnp.maximum(0.0, 1.0 - prev_cum_p), sorted_new_p)

            undo_sort_idx = jnp.argsort(sort_idx, axis=-1)
            p_case3 = jnp.take_along_axis(sorted_new_p, undo_sort_idx, axis=-1)

            # ==========================================
            # 👑 路由整合：判断每个样本属于哪个 Case
            # ==========================================
            cond_case3 = (p_pos * (1 + r_val) > 1.0)
            cond_case1 = ~cond_case3 & ((l_val + r_val) * p_pos <= l_val)

            pool_probs = jnp.where(cond_case3, p_case3, jnp.where(cond_case1, p_case1, p_case2))

            # 防弹装甲：防止浮点数精度误差累计或极个别劣质动作概率被减穿 0，做最后一次裁剪与归一化
            pool_probs = jnp.clip(pool_probs, 0.0, 1.0)
            pool_probs = pool_probs / jnp.sum(pool_probs, axis=-1, keepdims=True)

            metrics = {
                "q_guided/q_real_mean": jnp.mean(q_pool[:, 0]),
                "q_guided/q_real_penalized_mean": jnp.mean(q_pool_penalized[:, 0]),
                "q_guided/prob_real_mean": jnp.mean(pool_probs[:, 0]),
                "q_guided/l_val": l_val,
                "q_guided/r_val": r_val,
                "q_guided/case3_ratio": jnp.mean(cond_case3.astype(jnp.float32)),
                "q_guided/case1_ratio": jnp.mean(cond_case1.astype(jnp.float32)),
            }
            return jax.lax.stop_gradient(pool_probs), metrics

        else:
            raise ValueError(f"Unknown prob_allocation_mode: {self.config.prob_allocation_mode}")

    def _compute_value_loss(self, value_params, obs_norm, actions, truncation, target_qs):
        concat_inputs = jnp.concatenate([obs_norm, actions], axis=-1)
        q_pred, _ = networks.value_mlp_fwd_with_features(value_params, concat_inputs)
        q_pred = q_pred.reshape(target_qs.shape)
        v_error = (target_qs - q_pred) * (1 - truncation)
        mse_loss = jnp.mean(v_error ** 2)
        return mse_loss * self.config.value_loss_coeff * self.config.w_v_loss

    def get_schedule(self) -> FlowSchedule:
        full_t_path = jnp.linspace(1.0, 0.0, self.config.flow_steps + 1)
        return FlowSchedule(t_current=full_t_path[:-1], t_next=full_t_path[1:])

    def embed_timestep(self, t: Array) -> Array:
        assert t.shape[-1] == 1
        freqs = 2 ** jnp.arange(self.config.timestep_embed_dim // 2)
        scaled_t = t * freqs
        out = jnp.concatenate([jnp.cos(scaled_t), jnp.sin(scaled_t)], axis=-1)
        return out

    def sample_action(self, obs: Array, prng: Array, deterministic: bool) -> tuple[Array, DGPOFMActionInfo]:
        obs_norm = (obs - self.obs_stats.mean) / self.obs_stats.std if self.config.normalize_observations else obs
        (*batch_dims, obs_dim) = obs.shape

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

        x_raw = x0
        if not deterministic:
            prng_feather = jax.random.fold_in(prng, 0)
            noise = jax.random.normal(prng_feather, x_raw.shape)
            x_raw = x_raw + noise * self.config.feather_std

        x_final = self._apply_clip(x_raw)
        # 初始化时不用管 pool_actions，后续外层大循环会填满
        return x_final, DGPOFMActionInfo(raw_action=x_raw)

    def _update_critic_only(self, transitions: DGPOFMTransition, prng: Array) -> tuple[DGPOFMState, dict[str, Array]]:
        cfg = self.config
        obs_dim = self.env.observation_size
        act_dim = self.env.action_size
        N = transitions.obs.size // obs_dim
        obs_flat = ((transitions.obs - self.obs_stats.mean) / self.obs_stats.std
                    if cfg.normalize_observations else transitions.obs).reshape((N, obs_dim))
        target_qs = transitions.action_info.target_qs.reshape((N, 1))

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
            value_inner_step, (self.params.value, self.opt_state_value), None, length=cfg.loop_v
        )
        new_state = jdc.replace(self, params=jdc.replace(self.params, value=new_v_params),
                                opt_state_value=new_v_opt_state)
        return new_state, extra_v_metrics

    # ==========================================
    # 👑 全新架构：外层一次性目标准备函数 (防OOM分块处理)
    # ==========================================
    def _prepare_actor_targets_chunk(self, transitions_chunk: DGPOFMTransition, prng: Array) -> tuple[
        DGPOFMActionInfo, dict[str, Array]]:
        cfg = self.config
        prng_gen, prng_sample = jax.random.split(prng, 2)

        obs_dim = self.env.observation_size
        act_dim = self.env.action_size
        t_dim = cfg.timestep_embed_dim
        N = transitions_chunk.obs.shape[0]
        K_fakes = cfg.num_generated_actions

        obs_flat = ((transitions_chunk.obs - self.obs_stats.mean) / self.obs_stats.std
                    if cfg.normalize_observations else transitions_chunk.obs)

        real_action_flat = transitions_chunk.action_info.raw_action.reshape((N, 1, act_dim))
        obs_b_gen = jnp.broadcast_to(obs_flat[:, None, :], (N, K_fakes, obs_dim))

        fast_flow_steps = self.config.fast_flow_step
        fast_full_t = jnp.linspace(1.0, 0.0, fast_flow_steps + 1)
        fast_t_curr, fast_t_next = fast_full_t[:-1], fast_full_t[1:]

        def gen_step(x, t_tup):
            t_c, t_n = t_tup
            t_embed = jnp.broadcast_to(self.embed_timestep(jnp.array([t_c])[..., None])[:, None, :],
                                       (N, K_fakes, t_dim))
            vel = networks.flow_mlp_fwd(jax.lax.stop_gradient(self.params.policy), obs_b_gen, x,
                                        t_embed) * cfg.policy_mlp_output_scale
            return x + (t_n - t_c) * vel, None

        gen_acts, _ = jax.lax.scan(gen_step, jax.random.normal(prng_gen, (N, K_fakes, act_dim)),
                                   (fast_t_curr, fast_t_next))

        # 👑 这是包含 48 个假动作的终极全集
        pool_actions_full = jnp.concatenate([real_action_flat, gen_acts], axis=1)
        probs_full, fresh_metrics = self._compute_fresh_weights(self.params.value, obs_flat, pool_actions_full)

        if cfg.sampling_mode == "relative_h_pool":
            h = cfg.h_fakes_in_pool
            # 👑 做法A实现：在外层一次性抽好 h 个假动作，冻结小池！
            fake_indices = jax.random.randint(prng_sample, (N, h), 1, K_fakes + 1)
            zero_idx = jnp.zeros((N, 1), dtype=jnp.int32)
            pool_indices = jnp.concatenate([zero_idx, fake_indices], axis=1)  # 形状: (N, h+1)

            # 提取概率并相对归一化
            local_probs_raw = jnp.take_along_axis(probs_full, pool_indices, axis=1)
            local_probs = local_probs_raw / (jnp.sum(local_probs_raw, axis=-1, keepdims=True) + 1e-8)

            # 把这个压缩后的 h+1 专属池子和概率存起来
            final_actions = jnp.take_along_axis(pool_actions_full, pool_indices[..., None], axis=1)
            final_probs = local_probs
        else:
            # absolute_budget 模式：原样保留 49 个全集，交由内层去做绝对匹配
            final_actions = pool_actions_full
            final_probs = probs_full

        new_action_info = jdc.replace(transitions_chunk.action_info, pool_actions=final_actions, pool_probs=final_probs)
        return new_action_info, fresh_metrics

    # ==========================================
    # 👑 极速版内层循环：再也没有 ODE 生成！
    # ==========================================
    def _update_actor_only(self, transitions: DGPOFMTransition, prng: Array, global_v_loss: Array) -> tuple[
        DGPOFMState, dict[str, Array]]:
        cfg, sch = self.config, self.get_schedule()
        prng_pol, next_prng = jax.random.split(prng, 2)

        obs_dim = self.env.observation_size
        act_dim = self.env.action_size

        # 👑 绝对防弹的 Flatten：无论 minibatch 切成几维，用 size 强行算出绝对的 N！
        N = transitions.obs.size // obs_dim
        obs_flat = ((transitions.obs - self.obs_stats.mean) / self.obs_stats.std
                    if cfg.normalize_observations else transitions.obs).reshape((N, obs_dim))

        mb_mean_reward = jnp.mean(transitions.reward)
        final_v_loss = global_v_loss

        # 👑 提取大循环锁定的池子，并配套 Flatten 展平维数！
        raw_pool_acts = transitions.action_info.pool_actions
        raw_pool_probs = transitions.action_info.pool_probs
        pool_actions = raw_pool_acts.reshape((N, raw_pool_acts.shape[-2], act_dim))
        pool_probs = raw_pool_probs.reshape((N, raw_pool_probs.shape[-1]))

        def policy_loss_fn(p_params):
            M = cfg.num_epsilon_samples  # 物理显存上限，例如 48
            K_fakes = cfg.num_generated_actions  # 例如 7

            p_mask, p_eps, p_t, p_trust = jax.random.split(prng_pol, 4)

            # ==========================================
            # 👑 1. 强制锁死真动作，拒绝任何假动作
            # ==========================================
            real_acts = pool_actions[:, 0:1, :]
            # 把真动作广播填满整个物理显存槽 M
            a_target = jnp.broadcast_to(real_acts, (N, M, act_dim))
            is_real_slot = jnp.ones((N, M), dtype=bool)

            # ==========================================
            # 👑 2. 虚拟大池计算与显存截断 (Truncation)
            # ==========================================
            p_real = pool_probs[:, 0]

            # 物理槽位上限 (M，比如你现在改成了 128)
            M_float = float(M)

            # 👑 解耦：虚拟算力预算！
            # 之前我们是用 M * (K+1)，但这受限于物理上限。
            # 现在我们直接给它一个庞大的虚拟总池！
            # 假设你希望一个“完美动作(p_real=1.0)”能拿满 128 个槽，
            # 甚至你希望“普通好动作(p_real=0.2)”就能拿满 128 个槽（即预算 = 128 / 0.2 = 640）
            # 我们设一个显式的理论大池预算：
            VIRTUAL_BUDGET = 640.0

            # 算出真动作在这个 640 的大池子里该拿多少个
            n_exp_real = p_real * VIRTUAL_BUDGET

            # 物理截断：除以真实的物理槽位 M
            # 假设 M=128：
            # 若 n_exp_real = 200 -> 200/128 > 1.0 -> 截断到 1.0 (128个全亮)
            # 若 n_exp_real = 32  -> 32/128 = 0.25 -> (25%的槽亮，期望提供32个算力)
            keep_prob = jnp.clip(n_exp_real / M_float, 0.0, 1.0)

            # ==========================================
            # 👑 3. 纯正的硬随机采样 (Hard Bernoulli)
            # ==========================================
            rand_vals = jax.random.uniform(p_mask, (N, M))
            alloc_valid_mask = (rand_vals < keep_prob[:, None]).astype(jnp.float32)

            # --- 速度场目标像坍缩 ---
            if cfg.action_clip in ["hard", "margin", "fold"]:
                a_target = self._apply_clip(a_target)

            # --- Z-Score EMA 信任防线 ---
            t_outer = (self.steps // (cfg.num_updates_per_batch * cfg.num_minibatches)) + 1.0
            bc_v = 1.0 - jnp.power(cfg.beta_v, t_outer)
            bc_r = 1.0 - jnp.power(cfg.beta_r, t_outer)

            hat_r, hat_r_sq = self.ema_reward / bc_r, self.ema_reward_sq / bc_r
            hat_v, hat_v_sq = self.ema_v_loss / bc_v, self.ema_v_loss_sq / bc_v

            r_std = jnp.sqrt(jnp.maximum(hat_r_sq - jnp.square(hat_r), 0.0)) + 1e-5
            v_std = jnp.sqrt(jnp.maximum(hat_v_sq - jnp.square(hat_v), 0.0)) + 1e-5

            r_z = (mb_mean_reward - hat_r) / r_std
            v_z = (final_v_loss - hat_v) / v_std

            r_trust = jnp.clip(jnp.exp(-jnp.maximum(-r_z + cfg.tolerance_r, 0.0) / 0.5), 0.01, 1.0)
            v_trust = jnp.clip(jnp.exp(-jnp.maximum(v_z - cfg.tolerance_v, 0.0) / 0.5), 0.01, 1.0)

            combined_trust_prob = jnp.where(is_real_slot, r_trust, v_trust)
            trust_mask = (jax.random.uniform(p_trust, (N, M)) < combined_trust_prob).astype(jnp.float32)

            # 最终有效槽位 = 虚拟池摇中的槽位 AND 通过信任域的槽位
            final_valid_mask = alloc_valid_mask * trust_mask

            # --- 极速 Flow 拟合 ---
            eps = jax.random.normal(p_eps, (N, M, act_dim))
            t = sch.t_current[jax.random.randint(p_t, (N, M, 1), 0, cfg.flow_steps)]

            x_t = t * eps + (1.0 - t) * a_target

            t_embed = self.embed_timestep(t)
            obs_p = jnp.broadcast_to(obs_flat[:, None, :], (N, M, obs_dim))
            vel = networks.flow_mlp_fwd(p_params, obs_p, x_t, t_embed) * cfg.policy_mlp_output_scale

            if cfg.output_mode == "u_but_supervise_as_eps":
                err = jnp.sum((eps - ((x_t - t * vel) + vel)) ** 2, axis=-1)
            else:
                err = jnp.sum((vel - (eps - a_target)) ** 2, axis=-1)

            # jnp.mean 自动除以 M，因为无效的槽位被 mask 成了 0，
            # 总体梯度的 Magnitude 完美等比例缩放，没有产生软权重的衰减副作用！
            loss = jnp.mean(err * final_valid_mask)

            return loss, {
                "policy_loss": loss,
                "q_guided/real_trust_prob": r_trust,
                "q_guided/keep_prob": jnp.mean(keep_prob),
                "q_guided/theoretical_exp_noises": jnp.mean(n_exp_real),
                "q_guided/actual_hard_sampled_noises": jnp.mean(jnp.sum(alloc_valid_mask, axis=-1)),
                "q_guided/reward_z_score": r_z,
            }

        (p_loss, p_metrics), p_grads = jax.value_and_grad(policy_loss_fn, has_aux=True)(self.params.policy)
        p_updates, new_p_opt = self.opt_policy.update(p_grads, self.opt_state_policy, self.params.policy)

        new_state = jdc.replace(
            self,
            params=jdc.replace(self.params, policy=optax.apply_updates(self.params.policy, p_updates)),
            opt_state_policy=new_p_opt,
            steps=self.steps + 1,
            prng=next_prng
        )

        return new_state, p_metrics

    @jdc.jit
    def training_step(self, transitions: DGPOFMTransition) -> tuple[DGPOFMState, dict[str, Array]]:
        config, state = self.config, self
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

        clipped_boot_act = self._apply_clip(bootstrap_act)

        bootstrap_q, _ = networks.value_mlp_fwd_with_features(
            state.params.value,
            jnp.concatenate([bootstrap_obs, clipped_boot_act], axis=-1)
        )

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

        # ==========================================
        # 👑 核心修复：注入多维安全占位符，防止 JAX Scan 维度坍缩
        # ==========================================
        N_u, N_e = target_qs.shape[0], target_qs.shape[1]
        act_dim = state.env.action_size

        if config.sampling_mode == "relative_h_pool":
            dummy_pool_acts = jnp.zeros((N_u, N_e, config.h_fakes_in_pool + 1, act_dim))
            dummy_pool_probs = jnp.zeros((N_u, N_e, config.h_fakes_in_pool + 1))
        else:
            dummy_pool_acts = jnp.zeros((N_u, N_e, config.num_generated_actions + 1, act_dim))
            dummy_pool_probs = jnp.zeros((N_u, N_e, config.num_generated_actions + 1))

        new_action_info = jdc.replace(
            transitions.action_info,
            target_qs=target_qs,
            pool_actions=dummy_pool_acts,  # 占好四维的坑
            pool_probs=dummy_pool_probs  # 占好三维的坑
        )
        new_transitions = jdc.replace(transitions, action_info=new_action_info)

        # 后面的 def critic_epoch_step ... 保持完全不变

        def critic_epoch_step(carry_state, _):
            minibatches = new_transitions.prepare_minibatches(
                jax.random.fold_in(carry_state.prng, carry_state.steps), config.num_minibatches, config.batch_size
            )

            def minibatch_scan_fn(ms, mb):
                return ms._update_critic_only(mb, jax.random.fold_in(ms.prng, ms.steps + 1))

            return jax.lax.scan(minibatch_scan_fn, init=carry_state, xs=minibatches)

        state_after_v, all_v_metrics = jax.lax.scan(critic_epoch_step, init=state, length=config.num_updates_per_batch)
        current_global_v_loss = jnp.mean(all_v_metrics["v_loss/total"])

        batch_reward = jnp.mean(transitions.reward)
        new_state = jdc.replace(
            state_after_v,
            ema_reward=config.beta_r * state_after_v.ema_reward + (1.0 - config.beta_r) * batch_reward,
            ema_reward_sq=config.beta_r * state_after_v.ema_reward_sq + (1.0 - config.beta_r) * jnp.square(
                batch_reward),
            ema_v_loss=config.beta_v * state_after_v.ema_v_loss + (1.0 - config.beta_v) * current_global_v_loss,
            ema_v_loss_sq=config.beta_v * state_after_v.ema_v_loss_sq + (1.0 - config.beta_v) * jnp.square(
                current_global_v_loss)
        )

        # ==========================================
        # 👑 架构重塑核心：在外层大循环中，一次性安全分块计算 48 个假动作并抽出 h 池！
        # ==========================================
        N_total = new_transitions.obs.shape[0] * new_transitions.obs.shape[1]
        flat_transitions = jax.tree_util.tree_map(lambda x: x.reshape((N_total, *x.shape[2:])), new_transitions)
        chunk_size = N_total // config.num_minibatches
        chunked_transitions = jax.tree_util.tree_map(
            lambda x: x.reshape((config.num_minibatches, chunk_size, *x.shape[1:])), flat_transitions)

        prng_prep = jax.random.fold_in(new_state.prng, new_state.steps)
        prngs = jax.random.split(prng_prep, config.num_minibatches)

        def scan_prep_fn(carry, xs):
            trans_chunk, key = xs
            act_info, fresh_met = new_state._prepare_actor_targets_chunk(trans_chunk, key)
            return carry, (act_info, fresh_met)

        _, (prepped_action_info_chunked, prep_metrics_chunked) = jax.lax.scan(scan_prep_fn, None,
                                                                              (chunked_transitions, prngs))

        # 👑 终极防弹修复：直接拿着外层已经建好模的 new_action_info 作为形状对照表！
        # 无论 JAX 在 scan 里把数据拍扁成什么鬼样子（比如把 1920x16 拍成了 30720），
        # 只要总元素个数是对的，我们就强行把它塞回原本完美的形状中。
        prepped_action_info = jax.tree_util.tree_map(
            lambda orig_target, chunked_data: chunked_data.reshape(orig_target.shape),
            new_action_info,  # <--- 真理模具
            prepped_action_info_chunked  # <--- 被 JAX 拍扁的数据
        )
        prep_metrics = jax.tree_util.tree_map(lambda x: jnp.mean(x), prep_metrics_chunked)

        prepped_transitions = jdc.replace(new_transitions, action_info=prepped_action_info)

        # ==========================================
        # 👑 极速 Actor 小循环：只有纯粹的流拟合
        # ==========================================
        def actor_epoch_step(carry_state, _):
            minibatches = prepped_transitions.prepare_minibatches(
                jax.random.fold_in(carry_state.prng, carry_state.steps), config.num_minibatches, config.batch_size
            )

            def minibatch_scan_fn(ms, mb):
                return ms._update_actor_only(mb, jax.random.fold_in(ms.prng, ms.steps + 2), current_global_v_loss)

            return jax.lax.scan(minibatch_scan_fn, init=carry_state, xs=minibatches)

        final_state, all_p_metrics = jax.lax.scan(actor_epoch_step, init=new_state, length=config.num_updates_per_batch)

        return final_state, {**all_v_metrics, **prep_metrics, **all_p_metrics}