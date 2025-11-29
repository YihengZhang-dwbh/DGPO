"""Combined Encoder + Decoder agent for efficient training."""

from typing import Union

import jax
import jax_dataclasses as jdc
from jax import Array

from . import encoder_ppo
from .decoder_fm import DecoderFMState
from .decoder_diffusion import DecoderDiffusionState


@jdc.pytree_dataclass
class EncoderFMAgent:
    """Agent that combines Encoder and FM decoder for action generation."""

    ppo_z_state: encoder_ppo.EncoderState
    fm_state: DecoderFMState

    @property
    def env(self):
        return self.ppo_z_state.env

    def sample_z(
        self, obs: Array, prng: Array, deterministic: bool
    ) -> tuple[Array, encoder_ppo.EncoderActionInfo]:
        """Sample z from encoder policy."""
        return self.ppo_z_state.sample_z(obs, prng, deterministic)

    def map_z_to_action(self, obs: Array, z: Array) -> Array:
        """Map z to action using FM decoder (deterministic)."""
        dummy_prng = jax.random.PRNGKey(0)
        action = self.fm_state.sample_action_from_z(
            obs, z, dummy_prng, deterministic=True
        )
        return action

    def training_step(
        self, transitions: encoder_ppo.PpoZTransition
    ) -> tuple["EncoderFMAgent", dict[str, Array]]:
        """Update only encoder parameters (decoder is frozen)."""
        new_ppo_z_state, metrics = self.ppo_z_state.training_step(transitions)

        with jdc.copy_and_mutate(self) as agent:
            agent.ppo_z_state = new_ppo_z_state

        return agent, metrics


@jdc.pytree_dataclass
class EncoderDiffusionAgent:
    """Agent that combines Encoder and Diffusion decoder for action generation."""

    ppo_z_state: encoder_ppo.EncoderState
    diffusion_state: DecoderDiffusionState

    @property
    def env(self):
        return self.ppo_z_state.env

    def sample_z(
        self, obs: Array, prng: Array, deterministic: bool
    ) -> tuple[Array, encoder_ppo.EncoderActionInfo]:
        """Sample z from encoder policy."""
        return self.ppo_z_state.sample_z(obs, prng, deterministic)

    def map_z_to_action(self, obs: Array, z: Array) -> Array:
        """Map z to action using Diffusion decoder (deterministic)."""
        dummy_prng = jax.random.PRNGKey(0)
        action = self.diffusion_state.sample_action_from_z(
            obs, z, dummy_prng, deterministic=True
        )
        return action

    def training_step(
        self, transitions: encoder_ppo.PpoZTransition
    ) -> tuple["EncoderDiffusionAgent", dict[str, Array]]:
        """Update only encoder parameters (decoder is frozen)."""
        new_ppo_z_state, metrics = self.ppo_z_state.training_step(transitions)

        with jdc.copy_and_mutate(self) as agent:
            agent.ppo_z_state = new_ppo_z_state

        return agent, metrics
