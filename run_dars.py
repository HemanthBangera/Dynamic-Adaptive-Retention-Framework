import asyncio
import signal
import sys
import logging

from config.settings import DARSConfig
from core.layer_c.triage import TriageOrchestrator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    config = DARSConfig()
    orchestrator = TriageOrchestrator()
    
    stop_event = asyncio.Event()

    def handle_signal(sig):
        logger.info(f"Received signal {sig}, initiating graceful shutdown...")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig_name in ('SIGINT', 'SIGTERM'):
        try:
            sig = getattr(signal, sig_name)
            loop.add_signal_handler(sig, handle_signal, sig.name)
        except NotImplementedError:
            # For Windows compatibility
            pass
            
    # On Windows, add_signal_handler for SIGINT runs into issues, 
    # but asyncio handles KeyboardInterrupt anyway if run in top-level.
    
    # Start background components
    logger.info("DARS Framework running. Press Ctrl+C to stop.")
    
    try:
        await stop_event.wait()
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt caught, initiating graceful shutdown...")
        
    await orchestrator.shutdown(config.SHUTDOWN_TIMEOUT_SECONDS)
    logger.info("Shutdown complete.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
