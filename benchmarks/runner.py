"""
CLI shim: ``python -m benchmarks.runner`` → MemoryAgentBench driver.

Prefer ``python -m benchmarks.memory_agent_bench`` for clarity.
"""

from benchmarks.memory_agent_bench.__main__ import main

if __name__ == "__main__":
    main()
