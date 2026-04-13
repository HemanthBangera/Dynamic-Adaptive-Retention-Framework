import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from core.layer_b.evaluator import SuccessEvaluator
from core.layer_b.calculator import ScoreCalculator
from core.layer_b.engine import LearningEngine

@pytest.mark.asyncio
async def test_success_evaluator_yes():
    with patch("config.settings.DARSConfig.GEMINI_API_KEY", "real_key"):
        evaluator = SuccessEvaluator()
        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_resp = AsyncMock()
            mock_resp.status = 200
            mock_resp.json.return_value = {
                "candidates": [{"content": {"parts": [{"text": "YES"}]}}]
            }
            mock_post.return_value.__aenter__.return_value = mock_resp
            
            result = await evaluator.evaluate_success("query", "response", "memories")
            assert result == "YES"

def test_laplacian_smoothing_calculator():
    # Test Laplacian Smoothing formula: (Successes + 1) / (Attempts + 2)
    # Starts with 0 success, 0 failure. Adding 1 failure gives 0 success, 1 failure.
    updates = ScoreCalculator.calculate_updates(success=False, current_success_count=0, current_failure_count=0, current_access_count=0)
    
    assert updates["success_count"] == 0
    assert updates["failure_count"] == 1
    assert updates["frequency"] == 1
    # 1 / (1 + 2) = 1/3 = 0.333...
    assert pytest.approx(updates["utility"], 0.01) == 0.333

@pytest.mark.asyncio
async def test_atomic_patch_execution():
    evaluator_mock = Mock()
    evaluator_mock.evaluate_success = AsyncMock(return_value="YES")
    vault_mock = Mock()
    
    engine = LearningEngine(evaluator=evaluator_mock, vault=vault_mock)
    
    memories = [
        {"id": "A", "payload": {"text_content": "A", "success_count": 0, "failure_count": 0, "access_count": 0}}
    ]
    
    await engine.process_feedback_loop("q", "r", memories)
    
    vault_mock.patch_payload.assert_called_once()
    args, kwargs = vault_mock.patch_payload.call_args
    assert args[0] == "A"
    assert "utility" in args[1]
    assert args[1]["success_count"] == 1
