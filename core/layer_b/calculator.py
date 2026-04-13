import time

class ScoreCalculator:
    """Calculates updated DARS variables (U, F, R)."""
    
    @staticmethod
    def calculate_updates(success: bool, current_success_count: int, current_failure_count: int, current_access_count: int) -> dict:
        """
        Calculates new U, F, and R.
        U uses Laplacian Smoothing: (Successes + 1) / (Attempts + 2)
        """
        new_success = current_success_count + (1 if success else 0)
        new_failure = current_failure_count + (0 if success else 1)
        new_frequency = current_access_count + 1
        
        attempts = new_success + new_failure
        utility = (new_success + 1) / (attempts + 2) if attempts > 0 else 0.5
        
        return {
            "success_count": new_success,
            "failure_count": new_failure,
            "frequency": new_frequency,
            "utility": utility,
            "recency": time.time()
        }
