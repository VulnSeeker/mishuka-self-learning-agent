"""
Onyx — Self-Learning AI Agent
==============================

A self-learning agent that bootstraps skills from web research, builds
per-skill knowledge bases, and incrementally improves itself.

Basic usage:
    >>> from onyx import Onyx
    >>> agent = Onyx()
    >>> agent.learn("OSINT")
    >>> result = agent.run("osint", "Find emails for example.com")
    >>> print(result.answer)

Author: vulnseeker
License: MIT
"""

from onyx.version import __version__
from onyx.agent import Onyx, AgentError, SkillNotFoundError

__all__ = [
    "__version__",
    "Onyx",
    "AgentError",
    "SkillNotFoundError",
]

__author__ = "vulnseeker"
__license__ = "MIT"
__repo__ = "https://github.com/vulnseeker/onyx-self-learning-agent"
