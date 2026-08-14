"""Verifiers: the external constraints that steer sampling.

``base.Verifier`` is the interface every verifier implements, including the
policy, tool and episodic verifiers built in Phase 2.
"""

from verifiers.base import Verifier

__all__ = ["Verifier"]
