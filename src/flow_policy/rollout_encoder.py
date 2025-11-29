"""Rollout helpers for Encoder with decoder - handles z to action mapping efficiently."""

from __future__ import annotations

from typing import Protocol
from pathlib import Path

import jax
import jax_dataclasses as jdc
import mujoco_playground as mjp
from jax import Array
from jax import numpy as jnp
from mujoco import mjx

from . import rollouts
from .agent import EncoderFMAgent, EncoderDiffusionAgent


class EncoderAgentProtocol(Protocol):
    """Protocol for Encoder + Decoder agent state."""
    env: mjp.MjxEnv

    def sample_z(
        self, obs: Array, prng: Array, deterministic: bool
    ) -> tuple[Array, object]:
        """Sample z from encoder."""
        ...

    def map_z_to_action(
        self, obs: Array, z: Array
    ) -> Array:
        """Map z to action using decoder (deterministic)."""
        ...


@jdc.pytree_dataclass
class BatchedRolloutStateEncoderFM:
    """Rollout state for Encoder with FM decoder."""

    env: jdc.Static[mjp.MjxEnv]
    env_state: mjp.State
    first_obs: Array
    first_data: mjx.Data
    steps: Array
    num_envs: jdc.Static[int]
    prng: Array

    @staticmethod
    @jdc.jit
    def init(
        env: jdc.Static[mjp.MjxEnv],
        prng: Array,
        num_envs: jdc.Static[int],
    ) -> "BatchedRolloutStateEncoderFM":
        """Reset the environment."""
        prng, reset_prng = jax.random.split(prng, num=2)
        state = jax.vmap(env.reset)(jax.random.split(reset_prng, num=num_envs))
        return BatchedRolloutStateEncoderFM(
            env=env,
            env_state=state,
            first_obs=state.obs,
            first_data=state.data,
            steps=jnp.zeros_like(state.done),
            num_envs=num_envs,
            prng=prng,
        )

    @jdc.jit
    def rollout(
        self,
        agent_state: EncoderAgentProtocol,
        episode_length: jdc.Static[int],
        iterations_per_env: jdc.Static[int],
        auto_reset: jdc.Static[bool] = True,
        deterministic: jdc.Static[bool] = False,
        apply_tanh_in_rollout: jdc.Static[bool] = True,
    ) -> tuple["BatchedRolloutStateEncoderFM", rollouts.TransitionStruct]:
        """Perform rollout with z→action mapping via decoder."""

        def env_step(carry: "BatchedRolloutStateEncoderFM", _):
            state = carry

            prng_z, prng_next = jax.random.split(state.prng)
            assert isinstance(state.env_state.obs, Array)
            z, z_info = agent_state.sample_z(
                state.env_state.obs, prng_z, deterministic=deterministic
            )

            action = agent_state.map_z_to_action(state.env_state.obs, z)

            if apply_tanh_in_rollout:
                env_action = jnp.tanh(action)
            else:
                env_action = action

            next_env_state = jax.vmap(state.env.step)(
                state.env_state, env_action
            )
            assert isinstance(next_env_state.obs, Array)

            next_steps = state.steps + 1
            truncation = next_steps >= episode_length
            done_env = next_env_state.done.astype(bool)
            done_or_tr = jnp.logical_or(done_env, truncation)
            discount = 1.0 - done_env.astype(jnp.float32)

            transition = rollouts.TransitionStruct(
                obs=state.env_state.obs,
                next_obs=next_env_state.obs,
                action=z,  # Store z, NOT action!
                action_info=z_info,
                reward=next_env_state.reward,
                truncation=truncation.astype(jnp.float32),
                discount=discount,
            )

            if auto_reset:
                where_done = lambda x, y: jnp.where(
                    done_or_tr.reshape(
                        done_or_tr.shape + (1,) * (x.ndim - done_or_tr.ndim)
                    ),
                    x,
                    y,
                )

                next_env_state = next_env_state.replace(
                    obs=jax.tree.map(
                        where_done,
                        state.first_obs,
                        next_env_state.obs,
                    ),
                    data=jax.tree.map(
                        where_done,
                        state.first_data,
                        next_env_state.data,
                    ),
                    done=jnp.zeros_like(next_env_state.done),
                )

                next_steps = jnp.where(done_or_tr, 0, next_steps)

            with jdc.copy_and_mutate(state) as new_state:
                new_state.env_state = next_env_state
                new_state.steps = next_steps
                new_state.prng = prng_next

            return new_state, transition

        carry, transitions = jax.lax.scan(
            env_step,
            init=self,
            xs=None,
            length=iterations_per_env,
        )

        return carry, transitions

    def rollout_with_actions(
        self, agent, episode_length: int, iterations_per_env: int,
        apply_tanh_in_rollout: bool = True
    ) -> tuple[BatchedRolloutStateEncoderFM, Array, Array, Array]:
        """Rollout that returns actual actions (not z) for data collection."""
        def step_fn(state, _):
            prng, prng_sample = jax.random.split(state.prng)

            z, z_info = jax.vmap(agent.sample_z, in_axes=(0, 0, None))(
                state.env_state.obs,
                jax.random.split(prng_sample, self.num_envs),
                False
            )

            action = jax.vmap(agent.map_z_to_action, in_axes=(0, 0))(
                state.env_state.obs,
                z,
            )

            if apply_tanh_in_rollout:
                env_action = jnp.tanh(action)
            else:
                env_action = action
            next_env_state = jax.vmap(state.env.step)(
                state.env_state, env_action
            )

            done = next_env_state.done.astype(bool)
            where_done = lambda x, y: jnp.where(
                done.reshape(done.shape + (1,) * (x.ndim - done.ndim)),
                x,
                y,
            )

            next_env_state = next_env_state.replace(
                obs=jax.tree.map(where_done, state.first_obs, next_env_state.obs),
                data=jax.tree.map(where_done, state.first_data, next_env_state.data),
                done=jnp.zeros_like(next_env_state.done),
            )

            with jdc.copy_and_mutate(state) as new_state:
                new_state.env_state = next_env_state
                new_state.prng = prng

            return new_state, (state.env_state.obs, action, next_env_state.reward)

        final_state, (states, actions, rewards) = jax.lax.scan(
            step_fn,
            self,
            length=iterations_per_env
        )

        return final_state, states, actions, rewards


# Alias for Diffusion - same implementation, different name for clarity
BatchedRolloutStateEncoderDiffusion = BatchedRolloutStateEncoderFM


@jdc.jit
def eval_policy_encoder_fm(
    agent_state: EncoderAgentProtocol,
    prng: Array,
    num_envs: jdc.Static[int],
    max_episode_length: jdc.Static[int],
    apply_tanh_in_rollout: jdc.Static[bool] = True,
) -> rollouts.EvalOutputs:
    """Run policy evaluation for Encoder with FM decoder."""
    rollout_state = BatchedRolloutStateEncoderFM.init(
        agent_state.env, prng, num_envs
    )

    _, transitions = rollout_state.rollout(
        agent_state,
        episode_length=max_episode_length,
        iterations_per_env=max_episode_length,
        auto_reset=False,
        deterministic=True,
        apply_tanh_in_rollout=apply_tanh_in_rollout,
    )
    valid_mask = transitions.discount > 0.0

    rewards = jnp.sum(transitions.reward, axis=0)
    steps = jnp.sum(valid_mask, axis=0)

    scalar_metrics = {
        "reward_mean": jnp.mean(rewards),
        "reward_min": jnp.min(rewards),
        "reward_max": jnp.max(rewards),
        "reward_std": jnp.std(rewards),
        "steps_mean": jnp.mean(steps),
        "steps_min": jnp.min(steps),
        "steps_max": jnp.max(steps),
        "steps_std": jnp.std(steps),
    }

    histogram_metrics = {
        "reward": rewards.flatten(),
        "steps": steps.flatten(),
    }

    return rollouts.EvalOutputs(
        scalar_metrics=scalar_metrics,
        histogram_metrics=histogram_metrics,
        actions=transitions.action,
        action_timestep_mask=valid_mask,
    )


# Alias for Diffusion
eval_policy_encoder_diffusion = eval_policy_encoder_fm
