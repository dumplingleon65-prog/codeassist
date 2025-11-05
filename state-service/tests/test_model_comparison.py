import requests
import json
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    name: str
    model_id: str
    description: str


class FIMTester:
    """Test FIM (Fill-in-Middle) capabilities of different Ollama models."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()

        # Define the models to test
        self.models = [
            ModelConfig(
                "qwen2.5-coder-0.5b", "qwen2.5-coder:0.5b", "Qwen2.5 Coder 0.5B"
            ),
            ModelConfig(
                "qwen2.5-coder:0.5b-base",
                "qwen2.5-coder:0.5b-base",
                "qwen2.5-coder:0.5b-base",
            ),
            ModelConfig(
                "qwen2.5-coder:1.5b-base",
                "qwen2.5-coder:1.5b-base",
                "qwen2.5-coder:1.5b-base",
            ),
            ModelConfig(
                "qwen2.5-coder-instruct-0.5b",
                "qwen2.5-coder:0.5b-instruct",
                "Qwen2.5 Coder Instruct 0.5B",
            ),
        ]

    def health_check(self) -> bool:
        """Check if Ollama is running."""
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def ensure_model_available(self, model_id: str) -> bool:
        """Ensure the specified model is available, pull if necessary."""
        try:
            # Check if model exists
            response = self.session.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [model.get("name") for model in models]

                if model_id in model_names:
                    logger.info(f"Model {model_id} is already available")
                    return True

            # Pull the model if not available
            logger.info(f"Pulling model {model_id}...")
            pull_response = self.session.post(
                f"{self.base_url}/api/pull", json={"name": model_id}, timeout=300
            )

            if pull_response.status_code == 200:
                logger.info(f"Successfully pulled model {model_id}")
                return True
            else:
                logger.error(f"Failed to pull model {model_id}: {pull_response.text}")
                return False

        except Exception as e:
            logger.error(f"Error ensuring model availability: {e}")
            return False

    def generate_fim(
        self, model_id: str, prefix: str, suffix: str, **kwargs
    ) -> Optional[str]:
        """Generate text using FIM (Fill-in-Middle) approach."""
        try:
            prompt = "Complete the current line of code"
            # Create FIM prompt with special tokens
            fim_prompt = f"<|fim_prefix|>{prefix}<|fim_suffix|>{suffix}<|fim_middle|>"

            payload = {
                "model": model_id,
                "prompt": fim_prompt,
                "stream": False,
                "options": {
                    "num_predict": kwargs.get("max_tokens", 512),
                    "temperature": kwargs.get("temperature", 0.1),
                    "top_p": kwargs.get("top_p", 0.9),
                },
            }

            response = self.session.post(
                f"{self.base_url}/api/generate", json=payload, timeout=60
            )

            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"Generation failed: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error during FIM generation: {e}")
            return None

    def test_coding_scenarios(self) -> List[Dict[str, Any]]:
        """Test various coding scenarios with FIM."""

        scenarios = [
            {
                "name": "Simple Variable Assignment",
                "prefix": "# Python code only, no explanation\nimport math\n\ndef calculate_area(radius):",
                "suffix": "return area",
                "expected": "def calculate_area(radius):",
                "description": "Complete a simple variable assignment in a function",
            },
            {
                "name": "List Comprehension",
                "prefix": "numbers = [1, 2, 3, 4, 5]\neven_numbers = ",
                "suffix": "\nprint(even_numbers)",
                "expected": "[x for x in numbers if x % 2 == 0]",
                "description": "Complete a list comprehension to filter even numbers",
            },
            {
                "name": "Function Call",
                "prefix": 'def greet(name):\n    return f"Hello, {name}!"\n\nmessage = ',
                "suffix": "\nprint(message)",
                "expected": 'greet("World")',
                "description": "Complete a function call",
            },
            {
                "name": "Conditional Statement",
                "prefix": "age = 18\nif ",
                "suffix": ':\n    print("You are an adult")\nelse:\n    print("You are a minor")',
                "expected": "age >= 18",
                "description": "Complete a conditional statement",
            },
            {
                "name": "Loop with Range",
                "prefix": "for i in ",
                "suffix": ":\n    print(i)",
                "expected": "range(5)",
                "description": "Complete a for loop with range",
            },
        ]

        results = []

        for model_config in self.models:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Testing {model_config.description}")
            logger.info(f"{'=' * 60}")

            # Ensure model is available
            if not self.ensure_model_available(model_config.model_id):
                logger.error(
                    f"Failed to ensure model {model_config.model_id} is available"
                )
                continue

            model_results = {
                "model": model_config.name,
                "model_id": model_config.model_id,
                "description": model_config.description,
                "scenarios": [],
            }

            for scenario in scenarios:
                logger.info(f"\nTesting: {scenario['name']}")
                logger.info(f"Description: {scenario['description']}")
                logger.info(f"Prefix: {scenario['prefix']}")
                logger.info(f"Suffix: {scenario['suffix']}")
                logger.info(f"Expected: {scenario['expected']}")

                # Generate completion
                start_time = time.time()
                completion = self.generate_fim(
                    model_config.model_id, scenario["prefix"], scenario["suffix"]
                )
                end_time = time.time()

                if completion:
                    logger.info(f"Generated: {completion}")
                    logger.info(f"Time taken: {end_time - start_time:.2f}s")

                    # Simple accuracy check (exact match)
                    is_exact_match = completion.strip() == scenario["expected"].strip()

                    scenario_result = {
                        "name": scenario["name"],
                        "description": scenario["description"],
                        "prefix": scenario["prefix"],
                        "suffix": scenario["suffix"],
                        "expected": scenario["expected"],
                        "generated": completion,
                        "is_exact_match": is_exact_match,
                        "time_taken": end_time - start_time,
                    }
                else:
                    logger.error("Failed to generate completion")
                    scenario_result = {
                        "name": scenario["name"],
                        "description": scenario["description"],
                        "prefix": scenario["prefix"],
                        "suffix": scenario["suffix"],
                        "expected": scenario["expected"],
                        "generated": None,
                        "is_exact_match": False,
                        "time_taken": 0,
                    }

                model_results["scenarios"].append(scenario_result)

            results.append(model_results)

        return results

    def print_summary(self, results: List[Dict[str, Any]]):
        """Print a summary of the test results."""
        print("\n" + "=" * 80)
        print("FIM TESTING SUMMARY")
        print("=" * 80)

        for model_result in results:
            print(f"\n{model_result['description']}")
            print("-" * 50)

            total_scenarios = len(model_result["scenarios"])
            successful_scenarios = sum(
                1 for s in model_result["scenarios"] if s["generated"] is not None
            )
            exact_matches = sum(
                1 for s in model_result["scenarios"] if s["is_exact_match"]
            )
            avg_time = sum(
                s["time_taken"]
                for s in model_result["scenarios"]
                if s["time_taken"] > 0
            ) / max(successful_scenarios, 1)

            print(f"Total scenarios: {total_scenarios}")
            print(f"Successful generations: {successful_scenarios}/{total_scenarios}")
            print(f"Exact matches: {exact_matches}/{total_scenarios}")
            print(f"Average generation time: {avg_time:.2f}s")

            # Print individual scenario results
            for scenario in model_result["scenarios"]:
                status = (
                    "✓"
                    if scenario["is_exact_match"]
                    else "✗"
                    if scenario["generated"]
                    else "ERROR"
                )
                print(
                    f"  {status} {scenario['name']}: {scenario['generated'] or 'FAILED'}"
                )

    def save_results(
        self, results: List[Dict[str, Any]], filename: str = "fim_test_results.json"
    ):
        """Save test results to a JSON file."""
        try:
            with open(filename, "w") as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {filename}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


def main():
    """Main function to run FIM testing."""
    tester = FIMTester()

    # Check if Ollama is running
    if not tester.health_check():
        logger.error("Ollama is not running. Please start Ollama first.")
        return

    logger.info("Starting FIM testing with Qwen2.5 models...")

    # Run tests
    results = tester.test_coding_scenarios()

    # Print summary
    tester.print_summary(results)

    # Save results
    tester.save_results(results)

    logger.info("FIM testing completed!")


if __name__ == "__main__":
    main()
