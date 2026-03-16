import os
import logging
from .llm_api_client import APIClient

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_client():
    """Test the API client functionality"""
    client = APIClient(
        api_key="sk-729ef159c2b74926874860f6e7e12ca6",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name="qwen-vl-max",
        timeout=60
    )
    
    test_image_path = "test_image.jpg"
    
    test_cases = [
        {
            "name": "Simple Text Chat",
            "func": lambda: client.simple_chat("Hello, how are you?"),
        },
        {
            "name": "Image Chat",
            "func": lambda: client.image_chat(
                "What do you see in this image?",
                test_image_path
            ),
        },
        {
            "name": "Multi-turn Chat",
            "func": lambda: client.multi_turn_chat(
                "Let's continue our conversation",
                conversation_context=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"}
                ]
            ),
        }
    ]
    
    _run_tests(test_cases)

def _run_tests(test_cases):
    """Run test cases and log results"""
    for test in test_cases:
        logger.info(f"\nTesting: {test['name']}")
        try:
            response = test["func"]()
            logger.info(f"Response: {response}")
            if response:
                logger.info("Test passed ✓")
            else:
                logger.error("Test failed ✗")
        except Exception as e:
            logger.error(f"Test failed ✗: {str(e)}")

if __name__ == "__main__":
    test_api_client() 