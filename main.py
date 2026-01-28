
import asyncio
import json
import numpy as np
import os
import time
from contextlib import redirect_stdout
from io import StringIO
from typing import Any, Callable, TypedDict

from anthropic import AsyncAnthropic
from anthropic.types import MessageParam, ToolUnionParam
import anthropic


class PythonExpressionToolResult(TypedDict):
    result: Any
    error: str | None


class SubmitAnswerToolResult(TypedDict):
    answer: Any
    submitted: bool


def python_expression_tool(expression: str) -> PythonExpressionToolResult:
    """Execute Python code and capture output."""
    try:
        namespace = {}
        stdout = StringIO()
        with redirect_stdout(stdout):
            exec(expression, namespace, namespace)
        return {"result": stdout.getvalue(), "error": None}
    except KeyboardInterrupt:
        raise
    except Exception as e:
        return {"result": None, "error": str(e)}


def submit_answer_tool(answer: Any) -> SubmitAnswerToolResult:
    """Submit the final answer."""
    return {"answer": answer, "submitted": True}


def validate_answer(submitted_answer) -> bool:
    """
    Validates ML pipeline with automatic class imbalance detection.
    Targets 15-35% pass rate by requiring comprehensive ML engineering skills.
    """

    if isinstance(submitted_answer, str):
        try:
            import ast
            submitted_answer = ast.literal_eval(submitted_answer)
        except Exception as e:
            print(f"Failed to parse answer as dictionary: {e}")
            return False

    if not isinstance(submitted_answer, dict):
        print(f"Answer must be a dictionary, got: {type(submitted_answer)}")
        return False

    # Required keys for ML pipeline implementation
    required_keys = [
        'imbalance_severity', 'imbalance_ratio', 'primary_metric', 'hardware_detected',
        'approaches_implemented', 'best_approach', 'best_f1_macro',
        'rare_class_improvement', 'inference_pipeline_ready', 'used_test_split'
    ]
    missing_keys = [key for key in required_keys if key not in submitted_answer]
    if missing_keys:
        print(f"Missing required keys: {missing_keys}")
        return False

    # Validate imbalance severity classification
    severity = submitted_answer.get('imbalance_severity', '')
    valid_severities = ['Balanced', 'Mild', 'Moderate', 'Severe', 'Extreme']
    if severity not in valid_severities:
        print(f"Invalid imbalance severity: {severity}. Must be one of {valid_severities}")
        return False

    # Validate imbalance ratio
    ratio = submitted_answer.get('imbalance_ratio', -1)
    if not isinstance(ratio, (int, float)) or ratio < 1.0 or ratio > 50.0:
        print(f"nvalid imbalance ratio: {ratio}. Must be float between 1.0-50.0")
        return False

    # Validate logical consistency between severity and ratio
    severity_ranges = {
        'Balanced': (1.0, 2.0),
        'Mild': (2.0, 3.0),
        'Moderate': (3.0, 5.0),
        'Severe': (5.0, 10.0),
        'Extreme': (10.0, 50.0)
    }
    expected_min, expected_max = severity_ranges[severity]
    if not (expected_min <= ratio <= expected_max):
        print(f" Severity '{severity}' doesn't match ratio {ratio}. Expected {expected_min}-{expected_max}")
        return False
    print("Imbalance severity and ratio are consistent")

    # Validate primary metric selection
    metric = submitted_answer.get('primary_metric', '')
    if ratio >= 2.0 and metric != 'f1_macro':
        print(f"Imbalanced data (ratio={ratio}) must use f1_macro, got: {metric}")
        return False
    if ratio < 2.0 and metric != 'accuracy':
        print(f"Balanced data (ratio={ratio}) should use accuracy, got: {metric}")
        return False
    print("Appropriate primary metric selected")

    # Validate hardware detection
    hardware = submitted_answer.get('hardware_detected', '')
    if hardware not in ['CPU', 'GPU']:
        print(f"nvalid hardware detected: {hardware}. Must be 'CPU' or 'GPU'")
        return False

    # Validate approaches implementation
    approaches = submitted_answer.get('approaches_implemented', [])
    valid_approaches = ['classical_ml', 'logistic_regression', 'ensemble_methods', 'llm_prompting', 'transformer']
    if not isinstance(approaches, list) or len(approaches) < 3:
        print(f"Must implement at least 3 approaches from: {valid_approaches}")
        return False

    invalid_approaches = [a for a in approaches if a not in valid_approaches]
    if invalid_approaches:
        print(f"Invalid approaches: {invalid_approaches}. Must be from {valid_approaches}")
        return False
    print("Multiple modeling approaches implemented")

    # Validate best approach selection (with flexible matching)
    best_approach = submitted_answer.get('best_approach', '')
    # Allow flexible naming: 'ensemble' -> 'ensemble_methods', etc.
    approach_aliases = {
        'ensemble': 'ensemble_methods',
        'svm': 'classical_ml',
        'random_forest': 'classical_ml',
        'logistic': 'logistic_regression'
    }
    normalized_best = approach_aliases.get(best_approach, best_approach)
    if normalized_best not in approaches:
        print(f"Best approach '{best_approach}' not in implemented approaches: {approaches}")
        return False

    # Validate F1-macro score (realistic range for proper test set evaluation)
    f1_score = submitted_answer.get('best_f1_macro', -1)
    if not isinstance(f1_score, (int, float)) or f1_score < 0.0 or f1_score > 1.0:
        print(f"Invalid F1-macro score: {f1_score}. Must be float between 0.0-1.0")
        return False

    # Validate rare class improvement flag (accept boolean or numeric)
    rare_improvement = submitted_answer.get('rare_class_improvement', None)
    if isinstance(rare_improvement, (int, float)):
        # Convert numeric improvement to boolean
        rare_improvement = rare_improvement > 0
        print(f"Converted numeric rare class improvement to boolean: {rare_improvement}")
    elif not isinstance(rare_improvement, bool):
        print(f"Rare class improvement must be boolean or numeric, got: {type(rare_improvement)}")
        return False

    # Validate inference pipeline readiness
    inference_ready = submitted_answer.get('inference_pipeline_ready', None)
    if not isinstance(inference_ready, bool):
        print(f"Inference pipeline readiness must be boolean, got: {type(inference_ready)}")
        return False

    # Pipeline completeness check
    if not inference_ready:
        print(f"Inference pipeline must be implemented and ready")
        return False

    # Train/test split validation
    used_test_split = submitted_answer.get('used_test_split', None)
    if not isinstance(used_test_split, bool) or not used_test_split:
        print(f"Must use proper train/test split and evaluate only on test set")
        return False

    print("All ML pipeline validation checks passed!")
    return True


async def run_agent_loop(
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    max_steps: int = 35,
    model: str = "claude-3-haiku-20240307",  # Reliable Haiku model
    verbose: bool = True,
) -> Any | None:
    """Run agent loop with tools, return submitted answer or None."""
    api_key = os.environ.get('ANTHROPIC_API_KEY')
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set")
    if verbose:
        print(f"Using API key: {api_key[:15]}...")
    client = AsyncAnthropic(api_key=api_key)
    messages: list[MessageParam] = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        if verbose:
            print(f"\n=== Step {step + 1}/{max_steps} ===")

        # Retry logic for API overload errors
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                response = await client.messages.create(
                    model=model, max_tokens=2000, tools=tools, messages=messages
                )
                break
            except anthropic.InternalServerError as e:
                # Retry on 500/529 errors (overloaded, api_error, etc.)
                if attempt < max_retries - 1 and ("overloaded" in str(e).lower() or "500" in str(e) or "internal server error" in str(e).lower()):
                    if verbose:
                        print(f"API error ({e}), retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    raise

        has_tool_use = False
        tool_results = []
        submitted_answer = None
        for content in response.content:
            if content.type == "text":
                if verbose:
                    print(f"Assistant: {content.text}")
            elif content.type == "tool_use":
                has_tool_use = True
                tool_name = content.name

                if tool_name in tool_handlers:
                    if verbose:
                        print(f"Using tool: {tool_name}")
                    handler = tool_handlers[tool_name]
                    tool_input = content.input

                    if tool_name == "python_expression":
                        if not isinstance(tool_input, dict) or "expression" not in tool_input:
                            result = {"result": None, "error": "Invalid python_expression call - missing 'expression' parameter"}
                        else:
                            if verbose:
                                print(f"\nCode: {tool_input['expression']}")
                            result = handler(tool_input["expression"])
                            if verbose and result.get("result"):
                                print(f"Output: {result['result']}")
                    elif tool_name == "submit_answer":
                        assert isinstance(tool_input, dict) and "answer" in tool_input
                        result = handler(tool_input["answer"])
                        submitted_answer = result["answer"]
                    else:
                        result = handler(**tool_input) if isinstance(tool_input, dict) else handler(tool_input)

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content.id,
                            "content": json.dumps(result),
                        }
                    )

        if has_tool_use:
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})
            if submitted_answer is not None:
                if verbose:
                    print(f"\nAgent submitted answer: {submitted_answer}")
                return submitted_answer
        else:
            if verbose:
                print("\nNo tool use in response, ending loop.")
            break

    if verbose:
        print(f"\nReached maximum steps ({max_steps}) without submitting answer.")
    return None


async def run_single_test(
    run_id: int,
    num_runs: int,
    prompt: str,
    tools: list[ToolUnionParam],
    tool_handlers: dict[str, Callable[..., Any]],
    answer_validator: Callable[[Any], bool],
    verbose: bool = False,
) -> tuple[int, bool, Any]:
    if verbose:
        print(f"\n\n{'=' * 20} RUN {run_id}/{num_runs} {'=' * 20}")

    result = await run_agent_loop(
        prompt=prompt,
        tools=tools,
        tool_handlers=tool_handlers,
        max_steps=10,  # Streamlined for quick analysis and submission
        verbose=verbose,
    )

    success = answer_validator(result) if result is not None else False

    if success:
        print(f"Run {run_id}: SUCCESS - Got {result}")
    else:
        print(f"Run {run_id}: FAILURE - Got {result}")

    return run_id, success, result


async def main(concurrent: bool = True):
    tools: list[ToolUnionParam] = [
        {
            "name": "python_expression",
            "description": "Execute Python code. Pre-installed: datasets, transformers, torch, sklearn, numpy, pandas. Use for data loading, model training, evaluation, and pipeline implementation. DO NOT call submit_answer() from Python - use the separate submit_answer tool.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Python code to execute. Use print() for output. Complete ALL analysis in ONE call. DO NOT call submit_answer() from Python.",
                    }
                },
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit the complete ML pipeline results with all required keys: imbalance_severity, imbalance_ratio, primary_metric, hardware_detected, approaches_implemented (3+ from: classical_ml, logistic_regression, ensemble_methods, llm_prompting, transformer), best_approach, best_f1_macro (from TEST SET only, expect 0.15-0.85), rare_class_improvement, inference_pipeline_ready, used_test_split",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {"description": "Complete dictionary with all 9 required pipeline analysis keys"}},
                "required": ["answer"],
            },
        },
    ]

    tool_handlers = {
        "python_expression": python_expression_tool,
        "submit_answer": submit_answer_tool,
    }

    # ML Pipeline with Automatic Class Imbalance Detection Task
    num_runs = 10  # Test validation fixes
    prompt = """You're an ML engineer at a fast-growing startup providing emotion detection services. Build a robust text classification pipeline with automatic class imbalance detection using the `mteb/emotion` dataset.

PROBLEM: Current ML pipelines fail in production because they don't analyze class distribution or adapt to imbalance, resulting in models that miss critical rare classes despite good overall accuracy.

YOUR TASK: Build a comprehensive pipeline that automatically detects and handles class imbalance.

DETAILED REQUIREMENTS:

1. **Data Analysis & Imbalance Detection:**
   - Load the mteb/emotion dataset and use the first 500 training samples
   - Analyze the class distribution and calculate imbalance ratios using max_count divided by min_count
   - Classify severity based on ratio: Balanced (<2:1), Mild (2-3:1), Moderate (3-5:1), Severe (5-10:1), Extreme (>10:1)
   - Select appropriate evaluation metric: F1-macro for imbalanced data (≥2:1), accuracy for balanced data (<2:1)
   - Create proper train/test split (80/20) with stratification to maintain class balance across both sets**

2. **Hardware Detection:**
   - Check if GPU acceleration is available on the system
   - Report hardware as either 'GPU' or 'CPU'
   - Use this information to inform your model selection recommendations

3. **Multiple Modeling Approaches (implement exactly 3):**
   - **classical_ml**: TF-IDF vectorization + SVM with class weighting
   - **logistic_regression**: With class weighting for imbalanced data
   - **ensemble_methods**: Random Forest with class weighting

   **Important:** Use `zero_division=0` parameter in sklearn metrics to handle edge cases gracefully
   **Note:** Use exact approach names above for the 'approaches_implemented' and 'best_approach' fields
   **Critical:** Keep implementations simple and fast - focus on getting results, not perfect tuning

4. **Conditional Imbalance Handling based on severity:**
   - Mild: Apply class weighting only
   - Moderate: Combine sampling techniques with class weighting
   - Severe: Add ensemble methods for robust performance
   - Extreme: Implement comprehensive multi-technique approach

5. **Comprehensive Evaluation:**
   - **CRITICAL**: Evaluate ONLY on test set (never on training data)
   - Calculate per-class precision, recall, F1-scores
   - Generate confusion matrix analysis

   - Use primary metric determined by imbalance detection
   - Compare performance on rare classes vs baseline approaches

6. **Inference Pipeline:**
   - Select the best-performing model based on evaluation results
   - Create simple interface for making emotion predictions on new text samples
   - Return both predicted emotion labels and confidence scores

**TWO-STEP PROCESS: 1) PYTHON ANALYSIS 2) TOOL SUBMISSION**

**STEP 1: Complete analysis in one Python call**
- Load data: `dataset = load_dataset("mteb/emotion")`, take 500 samples, proper train/test split
- Analyze: Class distribution, imbalance ratio, severity classification
- Implement 3 approaches: TF-IDF+SVM, LogisticRegression, RandomForest (all with class_weight='balanced')
- Evaluate: F1-macro on test set only, select best approach
- Compute all required values and print them

**STEP 2: Use submit_answer TOOL (separate from Python)**
After Python execution, immediately use the submit_answer tool with your computed values.

**DO NOT call submit_answer() inside Python code - it won't work!**

**CRITICAL: submit_answer is a TOOL, not Python code!**

After your Python analysis, you MUST use the submit_answer TOOL separately:

**STEP 1:** Run Python code to compute all your values
**STEP 2:** Use submit_answer tool with your computed results

**IMPORTANT:** Do NOT call `submit_answer()` inside Python code - use it as a separate tool after your analysis is complete.**"""

    execution_mode = "concurrently" if concurrent else "sequentially"
    print(f"Running {num_runs} test iterations {execution_mode}...")
    print("=" * 60)

    # Run concurrently or sequentially based on the flag
    results = []

    if concurrent:
        # Create tasks and run concurrently
        tasks = []
        for i in range(num_runs):
            task = asyncio.create_task(
                run_single_test(
                    run_id=i + 1,
                    num_runs=num_runs,
                    prompt=prompt,
                    tools=tools,
                    tool_handlers=tool_handlers,
                    answer_validator=validate_answer,
                    verbose=True,
                )
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)
    else:
        # Run sequentially with delay to avoid API overload
        for i in range(num_runs):
            if i > 0:  # Add delay between runs (except first)
                await asyncio.sleep(1)

            result = await run_single_test(
                run_id=i + 1,
                num_runs=num_runs,
                prompt=prompt,
                tools=tools,
                tool_handlers=tool_handlers,
                answer_validator=validate_answer,
                verbose=True,  # Debug validation fixes
            )
            results.append(result)

    # Count successes
    successes = sum(1 for _, success, _ in results if success is True)

    # Calculate and display pass rate
    pass_rate = (successes / num_runs) * 100
    print(f"\n{'=' * 60}")
    print("Test Results:")
    print(f"  Passed: {successes}/{num_runs}")
    print(f"  Failed: {num_runs - successes}/{num_runs}")
    print(f"  Pass Rate: {pass_rate:.1f}%")

    # Pass rate assessment for RL training requirements
    if 10 <= pass_rate <= 40:
        status = "✅ MEETS REQUIREMENTS"
    elif pass_rate < 10:
        status = "⚠️  TOO DIFFICULT"
    else:
        status = "⚠️  TOO EASY"

    print(f"  Assessment: {status}\n{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main(concurrent=False))
