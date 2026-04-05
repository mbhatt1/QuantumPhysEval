#!/usr/bin/env python3

"""QuantumPhysEval benchmark runner."""

import argparse
import ast
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import product
from tempfile import NamedTemporaryFile
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from openai import OpenAI
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from tqdm import tqdm


EPS = 1e-8
SOLVED_EPS = 1e-9
__version__ = "0.1.0"
DEFAULT_REASONING_DEPTHS = (1, 2, 4)
DEFAULT_MODEL_SIZES: Dict[str, float] = {
    "gpt-4o-mini": 2e9,
    "gpt-4.1-mini": 5e9,
    "gpt-4.1": 1e10,
}
CATEGORY_ORDER = [
    "circuit_evolution",
    "operator_composition",
    "measurement_prediction",
    "entanglement_classification",
]
CATEGORY_LABELS = {
    "circuit_evolution": "Circuit evolution",
    "operator_composition": "Operator composition",
    "measurement_prediction": "Measurement prediction",
    "entanglement_classification": "Entanglement classification",
}
MODEL_COLORS = {
    "gpt-4o-mini": "#d95f02",
    "gpt-4.1-mini": "#c17c00",
    "gpt-4.1": "#1b9e77",
    "gpt-5.1": "#5f6caf",
    "gpt-5.2": "#2a9d8f",
    "gpt-5.4-mini": "#e07a5f",
    "gpt-5.4": "#264653",
    "claude-3-5-haiku-20241022": "#c77d52",
    "claude-3-7-sonnet-20250219": "#7b6dbe",
    "claude-sonnet-4-20250514": "#4f8fba",
    "claude-opus-4-1-20250805": "#1f5c99",
}
MODEL_LABELS = {
    "gpt-4o-mini": "GPT-4o mini",
    "gpt-4.1-mini": "GPT-4.1 mini",
    "gpt-4.1": "GPT-4.1",
    "gpt-5.1": "GPT-5.1",
    "gpt-5.2": "GPT-5.2",
    "gpt-5.4-mini": "GPT-5.4 mini",
    "gpt-5.4": "GPT-5.4",
    "claude-3-5-haiku-20241022": "Claude 3.5 Haiku",
    "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
    "claude-sonnet-4-20250514": "Claude Sonnet 4",
    "claude-opus-4-1-20250805": "Claude Opus 4.1",
}
SHORT_CATEGORY_LABELS = {
    "circuit_evolution": "Circuit",
    "operator_composition": "Operator",
    "measurement_prediction": "Measurement",
    "entanglement_classification": "Entanglement",
}
DEPTH_COLORS = {
    1: "#355070",
    2: "#6d597a",
    4: "#b56576",
}
FAMILY_COLORS = {
    "GPT-4": "#9a6b00",
    "GPT-5": "#27566b",
}
THREAD_LOCAL = threading.local()

I2 = np.eye(2, dtype=complex)
H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
S = np.array([[1, 0], [0, 1j]], dtype=complex)
T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4.0)]], dtype=complex)
CNOT_01 = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
)
CNOT_10 = np.array(
    [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=complex
)
CZ = np.diag([1, 1, 1, -1]).astype(complex)

ONE_QUBIT_GATES = {
    "I": I2,
    "H": H,
    "X": X,
    "Y": Y,
    "Z": Z,
    "S": S,
    "T": T,
}
TWO_QUBIT_GATES = {
    "CNOT(0,1)": CNOT_01,
    "CNOT(1,0)": CNOT_10,
    "CZ": CZ,
}


@dataclass
class BenchmarkPrompt:
    category: str
    prompt: str
    target_kind: str
    target: object
    metadata: Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="QuantumPhysEval benchmark with exact small-system quantum ground truth."
    )
    parser.add_argument(
        "--provider",
        choices=("openai", "anthropic"),
        default="openai",
        help="Model provider used for API calls.",
    )
    parser.add_argument(
        "--experiment",
        choices=("scaling", "time_budget"),
        default="scaling",
        help="Run the standard scaling sweep or a time-budgeted repair benchmark.",
    )
    parser.add_argument("--n-per-category", type=int, default=16)
    parser.add_argument(
        "--num-qubits",
        type=int,
        default=2,
        help="Number of qubits used for circuit, measurement, and entanglement tasks.",
    )
    parser.add_argument(
        "--reasoning-depths",
        type=int,
        nargs="+",
        default=list(DEFAULT_REASONING_DEPTHS),
    )
    parser.add_argument(
        "--model",
        action="append",
        metavar="NAME:SIZE",
        help="Override models as repeated NAME:SIZE pairs.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--time-budget-seconds",
        type=float,
        default=300.0,
        help="Maximum wall-clock budget per prompt for time_budget runs.",
    )
    parser.add_argument(
        "--checkpoint-seconds",
        type=float,
        nargs="+",
        default=[30.0, 60.0, 120.0, 300.0],
        help="Deadlines used for pass@time reporting in time_budget runs.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=24,
        help="Maximum repair attempts per prompt in time_budget runs.",
    )
    parser.add_argument(
        "--repair-mode",
        choices=("no_retry", "self_retry", "verifier_feedback"),
        default="verifier_feedback",
        help="Retry policy for time_budget runs.",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=24,
        help="Concurrent API worker threads used during benchmark evaluation.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=45.0,
        help="Per-request timeout in seconds for provider API calls.",
    )
    parser.add_argument(
        "--validate-models",
        action="store_true",
        help="Probe model availability before launching the full benchmark.",
    )
    parser.add_argument(
        "--skip-unavailable-models",
        action="store_true",
        help="When validation is enabled, drop inaccessible model IDs instead of failing.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Rewrite raw results JSON every N completed evaluations.",
    )
    parser.add_argument(
        "--plot-every",
        type=int,
        default=100,
        help="Refresh summary, fit, and plots every N completed evaluations.",
    )
    parser.add_argument("--results-json", default="artifacts/results/gpt4_family_results.json")
    parser.add_argument("--summary-json", default="artifacts/results/gpt4_family_summary.json")
    parser.add_argument("--fit-json", default="artifacts/results/gpt4_family_fit.json")
    parser.add_argument("--main-plot", default="artifacts/figures/gpt4_family_scaling.png")
    parser.add_argument("--breakdown-plot", default="artifacts/figures/gpt4_family_breakdown.png")
    parser.add_argument(
        "--reuse-results-json",
        help="Reuse an existing results JSON and only regenerate summaries/plots.",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the generated figures interactively.",
    )
    return parser.parse_args()


def parse_models(model_args: Optional[Sequence[str]]) -> Dict[str, float]:
    if not model_args:
        return dict(DEFAULT_MODEL_SIZES)

    parsed: Dict[str, float] = {}
    for item in model_args:
        if ":" not in item:
            raise ValueError(f"Invalid --model entry: {item!r}")
        name, size = item.split(":", 1)
        parsed[name] = float(size)
    return parsed


def require_api_key(provider: str) -> None:
    env_var = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    if not os.getenv(env_var):
        raise RuntimeError(f"{env_var} is not set.")


def apply_one_qubit_gate(state: np.ndarray, gate: np.ndarray, qubit: int, num_qubits: int) -> np.ndarray:
    tensor = np.asarray(state, dtype=complex).reshape((2,) * num_qubits)
    moved = np.moveaxis(tensor, qubit, 0)
    updated = np.tensordot(gate, moved, axes=([1], [0]))
    restored = np.moveaxis(updated, 0, qubit)
    return restored.reshape(-1)


def apply_two_qubit_gate(
    state: np.ndarray,
    gate: np.ndarray,
    qubit_a: int,
    qubit_b: int,
    num_qubits: int,
) -> np.ndarray:
    if qubit_a == qubit_b:
        raise ValueError("Two-qubit gate requires distinct qubits.")
    tensor = np.asarray(state, dtype=complex).reshape((2,) * num_qubits)
    moved = np.moveaxis(tensor, (qubit_a, qubit_b), (0, 1))
    leading_shape = moved.shape[2:]
    updated = gate @ moved.reshape(4, -1)
    restored = updated.reshape((2, 2) + leading_shape)
    final = np.moveaxis(restored, (0, 1), (qubit_a, qubit_b))
    return final.reshape(-1)


def initial_basis_state(bits: Sequence[int]) -> np.ndarray:
    num_qubits = len(bits)
    state = np.zeros(2**num_qubits, dtype=complex)
    index = 0
    for bit in bits:
        index = (index << 1) | int(bit)
    state[index] = 1.0 + 0.0j
    return state


def complex_to_list(arr: np.ndarray) -> List[List[float]]:
    flat = np.asarray(arr, dtype=complex).reshape(-1)
    return [[float(np.real(x)), float(np.imag(x))] for x in flat]


def matrix_to_nested_list(arr: np.ndarray) -> List[List[List[float]]]:
    mat = np.asarray(arr, dtype=complex)
    return [complex_to_list(row) for row in mat]


def format_complex(value: complex, digits: int = 8) -> str:
    real = round(float(np.real(value)), digits)
    imag = round(float(np.imag(value)), digits)
    if abs(imag) < 10 ** (-digits):
        return f"{real:.{digits}f}"
    sign = "+" if imag >= 0 else "-"
    return f"{real:.{digits}f}{sign}{abs(imag):.{digits}f}j"


def format_complex_vector(arr: np.ndarray, digits: int = 8) -> str:
    return "[" + ", ".join(format_complex(x, digits=digits) for x in arr.reshape(-1)) + "]"


def format_complex_matrix(arr: np.ndarray, digits: int = 8) -> str:
    rows = []
    for row in arr:
        rows.append("[" + ", ".join(format_complex(x, digits=digits) for x in row) + "]")
    return "[" + ", ".join(rows) + "]"


def format_prob_vector(arr: np.ndarray, digits: int = 8) -> str:
    return "[" + ", ".join(f"{float(x):.{digits}f}" for x in arr.reshape(-1)) + "]"


def random_single_qubit_sequence(rng: np.random.Generator, min_len: int = 2, max_len: int = 4) -> List[str]:
    names = ["H", "X", "Y", "Z", "S", "T"]
    length = int(rng.integers(min_len, max_len + 1))
    return [str(rng.choice(names)) for _ in range(length)]


def random_two_qubit_sequence(rng: np.random.Generator, min_len: int = 2, max_len: int = 4) -> List[str]:
    names = [
        "H(0)",
        "H(1)",
        "X(0)",
        "X(1)",
        "Y(0)",
        "Y(1)",
        "Z(0)",
        "Z(1)",
        "S(0)",
        "S(1)",
        "CNOT(0,1)",
        "CNOT(1,0)",
        "CZ",
    ]
    length = int(rng.integers(min_len, max_len + 1))
    return [str(rng.choice(names)) for _ in range(length)]


def random_n_qubit_sequence(
    rng: np.random.Generator,
    num_qubits: int,
    min_len: int = 3,
    max_len: int = 6,
) -> List[str]:
    length = int(rng.integers(min_len, max_len + 1))
    names: List[str] = []
    one_qubit_names = ["H", "X", "Y", "Z", "S", "T"]
    entangling_pairs = [(idx, idx + 1) for idx in range(num_qubits - 1)]
    for _ in range(length):
        if entangling_pairs and float(rng.random()) < 0.38:
            control, target = entangling_pairs[int(rng.integers(0, len(entangling_pairs)))]
            if float(rng.random()) < 0.5:
                names.append(f"CNOT({control},{target})")
            else:
                names.append(f"CZ({control},{target})")
        else:
            gate = str(rng.choice(one_qubit_names))
            qubit = int(rng.integers(0, num_qubits))
            names.append(f"{gate}({qubit})")
    return names


def apply_operation_to_state(state: np.ndarray, op_name: str, num_qubits: int) -> np.ndarray:
    if op_name in TWO_QUBIT_GATES and num_qubits == 2:
        return TWO_QUBIT_GATES[op_name] @ state
    match = re.fullmatch(r"([A-Z]+)\((\d+)(?:,(\d+))?\)", op_name)
    if not match:
        raise ValueError(f"Unsupported operation name: {op_name}")
    gate_name = match.group(1)
    qubit_a = int(match.group(2))
    qubit_b_text = match.group(3)
    if qubit_b_text is None:
        return apply_one_qubit_gate(state, ONE_QUBIT_GATES[gate_name], qubit_a, num_qubits)
    qubit_b = int(qubit_b_text)
    if gate_name == "CNOT":
        gate = CNOT_01
    elif gate_name == "CZ":
        gate = CZ
    else:
        raise ValueError(f"Unsupported two-qubit gate: {gate_name}")
    return apply_two_qubit_gate(state, gate, qubit_a, qubit_b, num_qubits)


def build_circuit_state(bits: Sequence[int], ops: Sequence[str], num_qubits: int) -> np.ndarray:
    state = initial_basis_state(bits)
    for op_name in ops:
        state = apply_operation_to_state(state, op_name, num_qubits)
    return state


def build_single_qubit_operator(ops: Sequence[str]) -> np.ndarray:
    unitary = np.eye(2, dtype=complex)
    for op_name in ops:
        unitary = ONE_QUBIT_GATES[op_name] @ unitary
    return unitary


def kron_all(states: Sequence[np.ndarray]) -> np.ndarray:
    result = np.asarray([1.0 + 0.0j], dtype=complex)
    for state in states:
        result = np.kron(result, np.asarray(state, dtype=complex))
    return result


def product_state_for_bits(bits: Sequence[int]) -> np.ndarray:
    return initial_basis_state(bits)


def entangled_state_from_template(template: str, num_qubits: int) -> Tuple[np.ndarray, bool]:
    inv_sqrt2 = 1.0 / np.sqrt(2.0)
    plus = np.array([inv_sqrt2, inv_sqrt2], dtype=complex)
    zero = np.array([1.0, 0.0], dtype=complex)
    one = np.array([0.0, 1.0], dtype=complex)
    bell_phi_plus = np.array([inv_sqrt2, 0, 0, inv_sqrt2], dtype=complex)
    bell_psi_plus = np.array([0, inv_sqrt2, inv_sqrt2, 0], dtype=complex)
    if num_qubits == 2:
        states = {
            "bell_phi_plus": (bell_phi_plus, True),
            "bell_phi_minus": (np.array([inv_sqrt2, 0, 0, -inv_sqrt2], dtype=complex), True),
            "bell_psi_plus": (bell_psi_plus, True),
            "product_00": (np.array([1, 0, 0, 0], dtype=complex), False),
            "product_plus0": (np.array([inv_sqrt2, 0, inv_sqrt2, 0], dtype=complex), False),
            "product_plus1": (np.array([0, inv_sqrt2, 0, inv_sqrt2], dtype=complex), False),
        }
        return states[template]
    if num_qubits == 4:
        ghz4 = np.zeros(16, dtype=complex)
        ghz4[0] = inv_sqrt2
        ghz4[-1] = inv_sqrt2
        cluster_4 = np.array(
            [0.5, 0, 0, 0.5, 0, 0, 0.5, 0, 0, 0.5, 0, 0, 0.5, 0, 0, -0.5],
            dtype=complex,
        )
        states = {
            "ghz4": (ghz4, True),
            "bell_pair_00": (np.kron(bell_phi_plus, product_state_for_bits((0, 0))), True),
            "00_bell_pair": (np.kron(product_state_for_bits((0, 0)), bell_phi_plus), True),
            "double_bell_pair": (np.kron(bell_phi_plus, bell_psi_plus), True),
            "cluster4": (cluster_4, True),
            "product_0000": (product_state_for_bits((0, 0, 0, 0)), False),
            "product_plus000": (kron_all([plus, zero, zero, zero]), False),
            "product_plusplus00": (kron_all([plus, plus, zero, zero]), False),
            "product_plus0plus1": (kron_all([plus, zero, plus, one]), False),
        }
        return states[template]
    if num_qubits == 8:
        ghz8 = np.zeros(256, dtype=complex)
        ghz8[0] = inv_sqrt2
        ghz8[-1] = inv_sqrt2
        bell_pair = bell_phi_plus
        states = {
            "ghz8": (ghz8, True),
            "bellpair_000000": (np.kron(bell_pair, product_state_for_bits((0, 0, 0, 0, 0, 0))), True),
            "0000_bellpair_00": (np.kron(product_state_for_bits((0, 0, 0, 0)), np.kron(bell_pair, product_state_for_bits((0, 0)))), True),
            "double_bell_0000": (np.kron(np.kron(bell_pair, bell_psi_plus), product_state_for_bits((0, 0, 0, 0))), True),
            "triple_bell_00": (np.kron(np.kron(np.kron(bell_pair, bell_phi_plus), bell_psi_plus), product_state_for_bits((0, 0))), True),
            "product_00000000": (product_state_for_bits((0, 0, 0, 0, 0, 0, 0, 0)), False),
            "product_plus0000000": (kron_all([plus, zero, zero, zero, zero, zero, zero, zero]), False),
            "product_plusplus000000": (kron_all([plus, plus, zero, zero, zero, zero, zero, zero]), False),
            "product_plus0plus10000": (kron_all([plus, zero, plus, one, zero, zero, zero, zero]), False),
            "product_alt_basis": (kron_all([plus, one, plus, zero, plus, zero, one, zero]), False),
        }
        return states[template]
    raise ValueError(f"Unsupported entanglement template size: {num_qubits}")


def generate_prompts(n_per_category: int, seed: int, num_qubits: int = 2) -> List[BenchmarkPrompt]:
    rng = np.random.default_rng(seed)
    prompts: List[BenchmarkPrompt] = []
    if num_qubits < 2:
        raise ValueError("--num-qubits must be at least 2.")

    circuit_sequence_fn = random_two_qubit_sequence if num_qubits == 2 else lambda generator: random_n_qubit_sequence(generator, num_qubits)
    basis_labels = [f"{idx:0{num_qubits}b}" for idx in range(2**num_qubits)]

    for _ in range(n_per_category):
        bits = tuple(int(x) for x in rng.integers(0, 2, size=num_qubits))
        ops = circuit_sequence_fn(rng)
        target_state = build_circuit_state(bits, ops, num_qubits)
        bit_string = "".join(str(bit) for bit in bits)
        prompt = (
            f"Start from |{bit_string}>. Apply the {num_qubits}-qubit gate sequence "
            f"{', '.join(ops)} in that order. Return the final state vector as a Python "
            f"list of {2**num_qubits} complex numbers using decimal literals only and no explanation."
        )
        prompts.append(
            BenchmarkPrompt(
                category="circuit_evolution",
                prompt=prompt,
                target_kind="state_vector",
                target=complex_to_list(target_state),
                metadata={"bits": list(bits), "ops": list(ops), "num_qubits": num_qubits},
            )
        )

    for _ in range(n_per_category):
        ops = random_single_qubit_sequence(rng)
        target_unitary = build_single_qubit_operator(ops)
        prompt = (
            f"Compute the 2x2 unitary matrix for the single-qubit gate sequence "
            f"{', '.join(ops)} applied in that order. Return only a Python 2x2 list of "
            "complex numbers using decimal literals."
        )
        prompts.append(
            BenchmarkPrompt(
                category="operator_composition",
                prompt=prompt,
                target_kind="operator",
                target=matrix_to_nested_list(target_unitary),
                metadata={"ops": list(ops), "num_qubits": 1},
            )
        )

    for _ in range(n_per_category):
        bits = tuple(int(x) for x in rng.integers(0, 2, size=num_qubits))
        ops = circuit_sequence_fn(rng)
        final_state = build_circuit_state(bits, ops, num_qubits)
        probabilities = np.abs(final_state) ** 2
        bit_string = "".join(str(bit) for bit in bits)
        label_string = ", ".join(f"p{label}" for label in basis_labels)
        prompt = (
            f"Start from |{bit_string}>. Apply the {num_qubits}-qubit gate sequence "
            f"{', '.join(ops)} in that order. Return the measurement probabilities "
            f"[{label_string}] as a Python list of {2**num_qubits} decimal numbers only."
        )
        prompts.append(
            BenchmarkPrompt(
                category="measurement_prediction",
                prompt=prompt,
                target_kind="probabilities",
                target=[float(x) for x in probabilities],
                metadata={"bits": list(bits), "ops": list(ops), "num_qubits": num_qubits},
            )
        )

    if num_qubits == 2:
        entanglement_templates = [
            "bell_phi_plus",
            "bell_phi_minus",
            "bell_psi_plus",
            "product_00",
            "product_plus0",
            "product_plus1",
        ]
    elif num_qubits == 4:
        entanglement_templates = [
            "ghz4",
            "bell_pair_00",
            "00_bell_pair",
            "double_bell_pair",
            "cluster4",
            "product_0000",
            "product_plus000",
            "product_plusplus00",
            "product_plus0plus1",
        ]
    elif num_qubits == 8:
        entanglement_templates = [
            "ghz8",
            "bellpair_000000",
            "0000_bellpair_00",
            "double_bell_0000",
            "triple_bell_00",
            "product_00000000",
            "product_plus0000000",
            "product_plusplus000000",
            "product_plus0plus10000",
            "product_alt_basis",
        ]
    else:
        raise ValueError("Entanglement templates are currently implemented for 2-qubit, 4-qubit, and 8-qubit runs only.")
    for _ in range(n_per_category):
        template = str(rng.choice(entanglement_templates))
        state, is_entangled = entangled_state_from_template(template, num_qubits)
        prompt = (
            f"Consider the {num_qubits}-qubit state {format_complex_vector(state)}. "
            "Is it entangled? Return only True or False."
        )
        prompts.append(
            BenchmarkPrompt(
                category="entanglement_classification",
                prompt=prompt,
                target_kind="boolean",
                target=bool(is_entangled),
                metadata={"template": template, "num_qubits": num_qubits},
            )
        )

    rng.shuffle(prompts)
    return prompts


def normalize_complex_literal(text: str) -> str:
    return (
        text.replace("i", "j")
        .replace("I", "j")
        .replace("−", "-")
        .replace("\n", " ")
    )


def safe_eval_numeric(expr: str):
    node = ast.parse(expr, mode="eval")

    def visit(current):
        if isinstance(current, ast.Expression):
            return visit(current.body)
        if isinstance(current, ast.List):
            return [visit(item) for item in current.elts]
        if isinstance(current, ast.Tuple):
            return tuple(visit(item) for item in current.elts)
        if isinstance(current, ast.Constant):
            if isinstance(current.value, (int, float, complex, bool)):
                return current.value
            raise ValueError(f"Unsupported constant: {current.value!r}")
        if isinstance(current, ast.UnaryOp) and isinstance(current.op, (ast.UAdd, ast.USub)):
            operand = visit(current.operand)
            return +operand if isinstance(current.op, ast.UAdd) else -operand
        if isinstance(current, ast.BinOp) and isinstance(
            current.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
        ):
            left = visit(current.left)
            right = visit(current.right)
            if isinstance(current.op, ast.Add):
                return left + right
            if isinstance(current.op, ast.Sub):
                return left - right
            if isinstance(current.op, ast.Mult):
                return left * right
            if isinstance(current.op, ast.Div):
                return left / right
            return left ** right
        raise ValueError(f"Unsupported expression: {ast.dump(current)}")

    return visit(node)


def parse_python_literal(text: str):
    last_error = None
    for candidate in literal_candidates(text):
        try:
            return safe_eval_numeric(normalize_complex_literal(candidate))
        except Exception as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise ValueError("No parseable literal candidates found.")


def extract_code_blocks(text: str) -> List[str]:
    blocks = []
    pattern = re.compile(r"```(?:[A-Za-z0-9_+-]+)?\n(.*?)```", flags=re.DOTALL)
    for match in pattern.finditer(text):
        blocks.append(match.group(1).strip())
    return blocks


def extract_last_bracketed_literal(text: str) -> Optional[str]:
    end = text.rfind("]")
    if end == -1:
        return None
    depth = 0
    for index in range(end, -1, -1):
        char = text[index]
        if char == "]":
            depth += 1
        elif char == "[":
            depth -= 1
            if depth == 0:
                return text[index : end + 1].strip()
    return None


def literal_candidates(text: str) -> List[str]:
    candidates: List[str] = []

    def add(candidate: Optional[str]) -> None:
        if candidate is None:
            return
        cleaned = candidate.strip()
        if cleaned and cleaned not in candidates:
            candidates.append(cleaned)

    stripped = text.strip()
    add(stripped)
    for block in extract_code_blocks(text):
        add(block)
        add(extract_last_bracketed_literal(block))
    add(extract_last_bracketed_literal(text))
    return candidates


def extract_boolean_value(text: str) -> Optional[bool]:
    stripped = text.strip().strip("`").strip()
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    matches = re.findall(r"\b(True|False)\b", stripped, flags=re.IGNORECASE)
    if not matches:
        return None
    return matches[-1].lower() == "true"


def parse_complex_vector(text: str) -> Optional[np.ndarray]:
    try:
        value = parse_python_literal(text)
        arr = np.asarray(value, dtype=complex).reshape(-1)
        return arr
    except Exception:
        return None


def parse_complex_matrix(text: str) -> Optional[np.ndarray]:
    try:
        value = parse_python_literal(text)
        arr = np.asarray(value, dtype=complex)
        if arr.ndim != 2:
            return None
        return arr
    except Exception:
        return None


def parse_real_vector(text: str) -> Optional[np.ndarray]:
    try:
        value = parse_python_literal(text)
        arr = np.asarray(value, dtype=float).reshape(-1)
        return arr
    except Exception:
        return None


def phase_align(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    overlap = np.vdot(reference.reshape(-1), candidate.reshape(-1))
    if abs(overlap) < EPS:
        return candidate
    return candidate * np.exp(-1j * np.angle(overlap))


def bounded_state_error(target: np.ndarray, predicted: Optional[np.ndarray]) -> float:
    if predicted is None or predicted.shape != target.shape:
        return 1.0
    pred_norm = np.linalg.norm(predicted)
    if pred_norm < EPS:
        return 1.0
    norm_penalty = min(1.0, abs(pred_norm - 1.0))
    predicted = predicted / pred_norm
    aligned = phase_align(target, predicted)
    fidelity = abs(np.vdot(target, aligned)) ** 2
    return min(1.0, norm_penalty + (1.0 - fidelity))


def bounded_probability_error(target: np.ndarray, predicted: Optional[np.ndarray]) -> float:
    if predicted is None or predicted.shape != target.shape:
        return 1.0
    nonneg = np.clip(predicted, 0.0, None)
    total = float(np.sum(nonneg))
    if total < EPS:
        return 1.0
    normalized = nonneg / total
    distance = 0.5 * float(np.sum(np.abs(normalized - target)))
    norm_penalty = min(1.0, abs(float(np.sum(predicted)) - 1.0))
    return min(1.0, distance + norm_penalty)


def bounded_operator_error(target: np.ndarray, predicted: Optional[np.ndarray]) -> float:
    if predicted is None or predicted.shape != target.shape:
        return 1.0
    dim = target.shape[0]
    unitary_penalty = min(
        1.0,
        float(np.linalg.norm(predicted.conj().T @ predicted - np.eye(dim))) / (2.0 * np.sqrt(dim)),
    )
    aligned = phase_align(target, predicted)
    target_penalty = min(
        1.0,
        float(np.linalg.norm(aligned - target)) / (2.0 * np.sqrt(dim)),
    )
    return min(1.0, 0.5 * unitary_penalty + 0.5 * target_penalty)


def serialize_target(target_kind: str, target: object) -> object:
    return {"kind": target_kind, "value": target}


def deserialize_target(target_obj: object) -> Tuple[str, object]:
    if isinstance(target_obj, dict) and "kind" in target_obj and "value" in target_obj:
        return str(target_obj["kind"]), target_obj["value"]
    raise ValueError("Invalid serialized target.")


def evaluate_output(category: str, target_kind: str, target_obj: object, output: str) -> float:
    if target_kind == "state_vector":
        target = np.asarray([complex(r, i) for r, i in target_obj], dtype=complex)
        return bounded_state_error(target, parse_complex_vector(output))
    if target_kind == "operator":
        target = np.asarray(
            [[complex(r, i) for r, i in row] for row in target_obj],
            dtype=complex,
        )
        return bounded_operator_error(target, parse_complex_matrix(output))
    if target_kind == "probabilities":
        target = np.asarray(target_obj, dtype=float)
        return bounded_probability_error(target, parse_real_vector(output))
    if target_kind == "boolean":
        predicted_value = extract_boolean_value(output)
        if predicted_value is None:
            return 1.0
        predicted = predicted_value
        return 0.0 if predicted == bool(target_obj) else 1.0
    raise ValueError(f"Unsupported target kind: {target_kind}")


def query_openai(
    client: OpenAI,
    model: str,
    prompt: str,
    temperature: float,
    max_completion_tokens: int,
) -> str:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_completion_tokens=int(max_completion_tokens),
        messages=[
            {
                "role": "system",
                "content": (
                    "Return machine-readable answers only. "
                    "Do not explain, justify, or add markdown."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def query_anthropic(
    model: str,
    prompt: str,
    temperature: float,
    request_timeout: float,
    max_tokens: int,
) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set.")
    payload = {
        "model": model,
        "max_tokens": int(max_tokens),
        "temperature": temperature,
        "system": "Return machine-readable answers only. Do not explain, justify, or add markdown.",
        "messages": [{"role": "user", "content": prompt}],
    }
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=request_timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Anthropic API error ({exc.code}): {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Anthropic connection error: {exc}") from exc
    texts = [
        block.get("text", "")
        for block in body.get("content", [])
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "".join(texts).strip()


def query_model(
    provider: str,
    model: str,
    prompt: str,
    temperature: float,
    request_timeout: float,
    max_output_tokens: int = 512,
):
    if provider == "openai":
        return query_openai(
            client=get_thread_openai_client(request_timeout),
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_completion_tokens=max_output_tokens,
        )
    if provider == "anthropic":
        return query_anthropic(
            model=model,
            prompt=prompt,
            temperature=temperature,
            request_timeout=request_timeout,
            max_tokens=max_output_tokens,
        )
    raise ValueError(f"Unsupported provider: {provider}")


def response_token_budget(prompt_spec: BenchmarkPrompt) -> int:
    category = prompt_spec.category
    num_qubits = int(prompt_spec.metadata.get("num_qubits", 2))
    if category == "boolean":
        return 32
    if category == "operator_composition":
        return 512
    if num_qubits >= 8 and category in {"circuit_evolution", "measurement_prediction"}:
        return 4096
    if num_qubits >= 8 and category == "entanglement_classification":
        return 1024
    if num_qubits >= 4 and category in {"circuit_evolution", "measurement_prediction"}:
        return 1536
    if num_qubits >= 4 and category == "entanglement_classification":
        return 768
    return 512


def request_timeout_budget(prompt_spec: BenchmarkPrompt, base_timeout: float) -> float:
    category = prompt_spec.category
    num_qubits = int(prompt_spec.metadata.get("num_qubits", 2))
    if num_qubits >= 8 and category in {"circuit_evolution", "measurement_prediction"}:
        return max(float(base_timeout), 180.0)
    if num_qubits >= 8 and category == "entanglement_classification":
        return max(float(base_timeout), 90.0)
    if num_qubits >= 4 and category in {"circuit_evolution", "measurement_prediction"}:
        return max(float(base_timeout), 75.0)
    return float(base_timeout)


def validate_models(
    provider: str,
    models: Dict[str, float],
    temperature: float,
    request_timeout: float,
    skip_unavailable_models: bool,
) -> Dict[str, float]:
    require_api_key(provider)
    available: Dict[str, float] = {}
    failures: Dict[str, str] = {}

    for model_name, size in models.items():
        try:
            query_model(
                provider=provider,
                model=model_name,
                prompt="Return only OK.",
                temperature=temperature,
                request_timeout=request_timeout,
                max_output_tokens=32,
            )
            available[model_name] = size
        except Exception as exc:
            failures[model_name] = str(exc)

    if failures and not skip_unavailable_models:
        failure_lines = [f"{name}: {message}" for name, message in failures.items()]
        raise RuntimeError("Model validation failed:\n" + "\n".join(failure_lines))

    for name, message in failures.items():
        print(f"Skipping unavailable model {name}: {message}")
    return available


def get_thread_openai_client(request_timeout: float) -> OpenAI:
    client = getattr(THREAD_LOCAL, "openai_client", None)
    timeout = getattr(THREAD_LOCAL, "openai_timeout", None)
    if client is None or timeout != request_timeout:
        THREAD_LOCAL.openai_client = OpenAI(timeout=request_timeout, max_retries=2)
        THREAD_LOCAL.openai_timeout = request_timeout
    return THREAD_LOCAL.openai_client


def evaluate_task(
    provider: str,
    model_name: str,
    model_size: float,
    depth: int,
    prompt_spec: BenchmarkPrompt,
    temperature: float,
    request_timeout: float,
) -> dict:
    full_prompt = (
        f"{prompt_spec.prompt} Use exactly {depth} reasoning steps internally, "
        "but return only the final answer."
    )
    try:
        max_output_tokens = response_token_budget(prompt_spec)
        effective_timeout = request_timeout_budget(prompt_spec, request_timeout)
        output = query_model(
            provider=provider,
            model=model_name,
            prompt=full_prompt,
            temperature=temperature,
            request_timeout=effective_timeout,
            max_output_tokens=max_output_tokens,
        )
        error_text = None
        d_phys = evaluate_output(
            prompt_spec.category,
            prompt_spec.target_kind,
            prompt_spec.target,
            output,
        )
    except Exception as exc:
        output = ""
        error_text = str(exc)
        d_phys = None

    return {
        "provider": provider,
        "model": model_name,
        "size": model_size,
        "depth": int(depth),
        "num_qubits": int(prompt_spec.metadata.get("num_qubits", 2)),
        "category": prompt_spec.category,
        "prompt": prompt_spec.prompt,
        "metadata": prompt_spec.metadata,
        "target": serialize_target(prompt_spec.target_kind, prompt_spec.target),
        "output": output,
        "d_phys": d_phys,
        "api_error": error_text,
    }


def make_repair_prompt(
    prompt_spec: BenchmarkPrompt,
    depth: int,
    previous_output: Optional[str],
    attempt_index: int,
    repair_mode: str,
) -> str:
    base = (
        f"{prompt_spec.prompt} Use exactly {depth} reasoning steps internally, "
        "but return only the final answer."
    )
    if attempt_index == 0 or not previous_output:
        return base
    if repair_mode == "no_retry":
        return base
    if repair_mode == "self_retry":
        return (
            f"{base}\n"
            "Take another independent pass. Re-check normalization, unitarity, amplitudes, probabilities, "
            "and entanglement conditions as appropriate, and return only the final answer.\n"
            f"Previous answer: {previous_output}"
        )
    return (
        f"{base}\n"
        "Your previous final answer was judged incorrect against exact quantum ground truth. "
        "Re-solve from scratch, check the relevant physical constraints carefully, and return only the corrected final answer.\n"
        f"Previous answer: {previous_output}"
    )


def format_checkpoint_label(seconds: float) -> str:
    if abs(seconds - round(seconds)) < 1e-9:
        return f"{int(round(seconds))}s"
    return f"{seconds:.1f}s"


def evaluate_time_budget_task(
    provider: str,
    model_name: str,
    model_size: float,
    depth: int,
    prompt_spec: BenchmarkPrompt,
    temperature: float,
    request_timeout: float,
    time_budget_seconds: float,
    checkpoint_seconds: Sequence[float],
    max_attempts: int,
    repair_mode: str,
) -> dict:
    start = time.monotonic()
    attempts: List[dict] = []
    api_errors: List[str] = []
    previous_output: Optional[str] = None
    first_d_phys: Optional[float] = None
    best_d_phys: Optional[float] = None
    final_d_phys: Optional[float] = None
    solved_at_s: Optional[float] = None

    checkpoints = sorted({float(x) for x in checkpoint_seconds if float(x) > 0.0})
    if time_budget_seconds not in checkpoints:
        checkpoints.append(float(time_budget_seconds))
    pass_by_time = {format_checkpoint_label(x): False for x in checkpoints}

    allowed_attempts = 1 if repair_mode == "no_retry" else max_attempts
    for attempt_index in range(allowed_attempts):
        elapsed_before = time.monotonic() - start
        remaining_budget = time_budget_seconds - elapsed_before
        if remaining_budget <= 0:
            break

        prompt = make_repair_prompt(prompt_spec, depth, previous_output, attempt_index, repair_mode)
        latency_s = 0.0
        output = ""
        try:
            max_output_tokens = response_token_budget(prompt_spec)
            effective_timeout = request_timeout_budget(prompt_spec, request_timeout)
            call_started = time.monotonic()
            output = query_model(
                provider=provider,
                model=model_name,
                prompt=prompt,
                temperature=temperature,
                request_timeout=min(effective_timeout, max(remaining_budget, 1.0)),
                max_output_tokens=max_output_tokens,
            )
            latency_s = time.monotonic() - call_started
            d_phys = evaluate_output(
                prompt_spec.category,
                prompt_spec.target_kind,
                prompt_spec.target,
                output,
            )
            final_d_phys = d_phys
            if first_d_phys is None:
                first_d_phys = d_phys
            if best_d_phys is None or d_phys < best_d_phys:
                best_d_phys = d_phys
            elapsed_after = time.monotonic() - start
            solved = d_phys <= SOLVED_EPS
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "elapsed_s": float(elapsed_after),
                    "latency_s": float(latency_s),
                    "d_phys": float(d_phys),
                    "solved": bool(solved),
                    "output": output,
                }
            )
            if solved and solved_at_s is None:
                solved_at_s = float(elapsed_after)
                for checkpoint in checkpoints:
                    if solved_at_s <= checkpoint + 1e-9:
                        pass_by_time[format_checkpoint_label(checkpoint)] = True
                break
            previous_output = output
        except Exception as exc:
            elapsed_after = time.monotonic() - start
            error_text = str(exc)
            api_errors.append(error_text)
            attempts.append(
                {
                    "attempt": attempt_index + 1,
                    "elapsed_s": float(elapsed_after),
                    "latency_s": float(latency_s),
                    "d_phys": None,
                    "solved": False,
                    "output": output,
                    "api_error": error_text,
                }
            )
            previous_output = output or previous_output

    total_elapsed = time.monotonic() - start
    solved = solved_at_s is not None
    return {
        "provider": provider,
        "model": model_name,
        "size": model_size,
        "depth": int(depth),
        "num_qubits": int(prompt_spec.metadata.get("num_qubits", 2)),
        "category": prompt_spec.category,
        "prompt": prompt_spec.prompt,
        "metadata": prompt_spec.metadata,
        "target": serialize_target(prompt_spec.target_kind, prompt_spec.target),
        "experiment": "time_budget",
        "repair_mode": repair_mode,
        "time_budget_s": float(time_budget_seconds),
        "attempt_count": len(attempts),
        "attempts": attempts,
        "initial_d_phys": float(first_d_phys) if first_d_phys is not None else None,
        "best_d_phys": float(best_d_phys) if best_d_phys is not None else None,
        "final_d_phys": float(final_d_phys) if final_d_phys is not None else None,
        "solved": bool(solved),
        "solve_time_s": float(solved_at_s) if solved_at_s is not None else None,
        "total_elapsed_s": float(total_elapsed),
        "pass_by_time": pass_by_time,
        "api_error": "; ".join(api_errors) if api_errors else None,
    }


def run_benchmark(
    provider: str,
    prompts: Sequence[BenchmarkPrompt],
    models: Dict[str, float],
    depths: Sequence[int],
    temperature: float,
    results_json: str,
    summary_json: str,
    fit_json: str,
    main_plot: str,
    breakdown_plot: str,
    checkpoint_every: int,
    plot_every: int,
    max_workers: int,
    request_timeout: float,
) -> List[dict]:
    require_api_key(provider)
    existing_results: List[dict] = []
    if os.path.exists(results_json):
        existing_results = load_results(results_json)
    results: List[dict] = list(existing_results)
    existing_counts = Counter(result_task_key(row) for row in existing_results)
    work_items = []
    for model_name, depth in product(models.keys(), depths):
        for prompt_spec in prompts:
            key = (model_name, int(depth), prompt_spec.prompt)
            if existing_counts[key] > 0:
                existing_counts[key] -= 1
                continue
            work_items.append((model_name, models[model_name], int(depth), prompt_spec))

    if existing_results:
        save_json(results_json, results)
        maybe_write_intermediate_artifacts(
            results=results,
            summary_json=summary_json,
            fit_json=fit_json,
            main_plot=main_plot,
            breakdown_plot=breakdown_plot,
        )

    if not work_items:
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                evaluate_task,
                provider,
                model_name,
                model_size,
                depth,
                prompt_spec,
                temperature,
                request_timeout,
            )
            for model_name, model_size, depth, prompt_spec in work_items
        ]
        completed = len(existing_results)
        total = completed + len(futures)
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Evaluations (resume {completed}/{total})",
        ):
            results.append(future.result())
            completed += 1

            if checkpoint_every > 0 and completed % checkpoint_every == 0:
                save_json(results_json, results)

            if plot_every > 0 and completed % plot_every == 0:
                save_json(results_json, results)
                maybe_write_intermediate_artifacts(
                    results=results,
                    summary_json=summary_json,
                    fit_json=fit_json,
                    main_plot=main_plot,
                    breakdown_plot=breakdown_plot,
                )

    with open(results_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


def run_time_budget_benchmark(
    provider: str,
    prompts: Sequence[BenchmarkPrompt],
    models: Dict[str, float],
    depths: Sequence[int],
    temperature: float,
    results_json: str,
    summary_json: str,
    fit_json: str,
    main_plot: str,
    breakdown_plot: str,
    checkpoint_every: int,
    plot_every: int,
    max_workers: int,
    request_timeout: float,
    time_budget_seconds: float,
    checkpoint_seconds: Sequence[float],
    max_attempts: int,
    repair_mode: str,
) -> List[dict]:
    require_api_key(provider)
    existing_results: List[dict] = []
    if os.path.exists(results_json):
        existing_results = load_results(results_json)
    results: List[dict] = list(existing_results)
    existing_counts = Counter(result_task_key(row) for row in existing_results)
    work_items = []
    for model_name, depth in product(models.keys(), depths):
        for prompt_spec in prompts:
            key = (model_name, int(depth), prompt_spec.prompt)
            if existing_counts[key] > 0:
                existing_counts[key] -= 1
                continue
            work_items.append((model_name, models[model_name], int(depth), prompt_spec))

    if existing_results:
        save_json(results_json, results)
        maybe_write_time_budget_artifacts(
            results=results,
            summary_json=summary_json,
            fit_json=fit_json,
            main_plot=main_plot,
            breakdown_plot=breakdown_plot,
            checkpoint_seconds=checkpoint_seconds,
        )

    if not work_items:
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                evaluate_time_budget_task,
                provider,
                model_name,
                model_size,
                depth,
                prompt_spec,
                temperature,
                request_timeout,
                time_budget_seconds,
                checkpoint_seconds,
                max_attempts,
                repair_mode,
            )
            for model_name, model_size, depth, prompt_spec in work_items
        ]
        completed = len(existing_results)
        total = completed + len(futures)
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Time-budget evals (resume {completed}/{total})",
        ):
            results.append(future.result())
            completed += 1

            if checkpoint_every > 0 and completed % checkpoint_every == 0:
                save_json(results_json, results)

            if plot_every > 0 and completed % plot_every == 0:
                save_json(results_json, results)
                maybe_write_time_budget_artifacts(
                    results=results,
                    summary_json=summary_json,
                    fit_json=fit_json,
                    main_plot=main_plot,
                    breakdown_plot=breakdown_plot,
                    checkpoint_seconds=checkpoint_seconds,
                )

    save_json(results_json, results)
    return results


def fit_scaling_law(results: Sequence[dict]) -> Tuple[float, float, float]:
    usable = [row for row in results if row["d_phys"] is not None]
    if len(usable) < 3:
        raise RuntimeError("Need at least 3 usable rows for scaling fit.")
    sizes = np.array([row["size"] for row in usable], dtype=float)
    depths = np.array([row["depth"] for row in usable], dtype=float)
    errors = np.array([max(float(row["d_phys"]), EPS) for row in usable], dtype=float)
    design = np.column_stack([np.ones_like(sizes), -np.log(sizes), np.log(depths)])
    coeffs, _, _, _ = np.linalg.lstsq(design, np.log(errors), rcond=None)
    log_k, alpha, beta = coeffs
    return float(np.exp(log_k)), float(alpha), float(beta)


def bootstrap_fit(results: Sequence[dict], samples: int, seed: int) -> Dict[str, object]:
    usable = [row for row in results if row["d_phys"] is not None]
    rng = np.random.default_rng(seed)
    fits = []
    for _ in range(samples):
        draw = [usable[int(i)] for i in rng.integers(0, len(usable), size=len(usable))]
        try:
            fits.append(fit_scaling_law(draw))
        except Exception:
            continue
    arr = np.asarray(fits, dtype=float)
    if arr.size == 0:
        raise RuntimeError("Bootstrap fit failed for all resamples.")
    return {
        "samples": len(fits),
        "k_mean": float(np.mean(arr[:, 0])),
        "alpha_mean": float(np.mean(arr[:, 1])),
        "beta_mean": float(np.mean(arr[:, 2])),
        "k_ci95": [float(x) for x in np.quantile(arr[:, 0], [0.025, 0.975])],
        "alpha_ci95": [float(x) for x in np.quantile(arr[:, 1], [0.025, 0.975])],
        "beta_ci95": [float(x) for x in np.quantile(arr[:, 2], [0.025, 0.975])],
    }


def summarize_results(results: Sequence[dict]) -> Dict[str, object]:
    usable = [row for row in results if row["d_phys"] is not None]
    summary_rows = []
    grouped: Dict[Tuple[str, int, str], List[float]] = {}
    for row in usable:
        key = (row["model"], int(row["depth"]), row["category"])
        grouped.setdefault(key, []).append(float(row["d_phys"]))
    for (model, depth, category), values in sorted(grouped.items()):
        summary_rows.append(
            {
                "model": model,
                "depth": depth,
                "category": category,
                "count": len(values),
                "mean_d_phys": float(np.mean(values)),
                "median_d_phys": float(np.median(values)),
                "stderr_d_phys": float(np.std(values, ddof=1) / np.sqrt(len(values)))
                if len(values) > 1
                else 0.0,
            }
        )
    return {
        "num_rows": len(results),
        "num_usable_rows": len(usable),
        "num_api_errors": sum(1 for row in results if row["api_error"]),
        "rows": summary_rows,
    }


def summarize_time_budget_results(results: Sequence[dict], checkpoint_seconds: Sequence[float]) -> Dict[str, object]:
    usable = [row for row in results]
    checkpoint_labels = [format_checkpoint_label(x) for x in sorted({float(x) for x in checkpoint_seconds if float(x) > 0.0})]
    repair_modes = sorted({str(row.get("repair_mode", "verifier_feedback")) for row in usable})
    overall_pass = {
        label: float(np.mean([1.0 if row.get("pass_by_time", {}).get(label, False) else 0.0 for row in usable]))
        if usable
        else 0.0
        for label in checkpoint_labels
    }
    solved_rows = [row for row in usable if row.get("solved")]
    summary_rows = []
    grouped: Dict[Tuple[str, str], List[dict]] = {}
    for row in usable:
        grouped.setdefault((str(row["model"]), str(row["category"])), []).append(row)
    for (model, category), rows_for_key in sorted(grouped.items()):
        solve_times = [float(row["solve_time_s"]) for row in rows_for_key if row.get("solve_time_s") is not None]
        best_errors = [float(row["best_d_phys"]) for row in rows_for_key if row.get("best_d_phys") is not None]
        summary_row = {
            "model": model,
            "category": category,
            "count": len(rows_for_key),
            "solve_rate": float(np.mean([1.0 if row.get("solved") else 0.0 for row in rows_for_key])),
            "mean_attempts": float(np.mean([float(row.get("attempt_count", 0)) for row in rows_for_key])),
            "mean_best_d_phys": float(np.mean(best_errors)) if best_errors else None,
            "mean_solve_time_s": float(np.mean(solve_times)) if solve_times else None,
            "median_solve_time_s": float(np.median(solve_times)) if solve_times else None,
        }
        for label in checkpoint_labels:
            summary_row[f"pass_{label}"] = float(
                np.mean([1.0 if row.get("pass_by_time", {}).get(label, False) else 0.0 for row in rows_for_key])
            )
        summary_rows.append(summary_row)
    return {
        "experiment": "time_budget",
        "num_rows": len(results),
        "num_api_errors": sum(1 for row in results if row.get("api_error")),
        "repair_modes": repair_modes,
        "solve_rate": float(np.mean([1.0 if row.get("solved") else 0.0 for row in usable])) if usable else 0.0,
        "mean_solve_time_s": float(np.mean([float(row["solve_time_s"]) for row in solved_rows])) if solved_rows else None,
        "median_solve_time_s": float(np.median([float(row["solve_time_s"]) for row in solved_rows])) if solved_rows else None,
        "overall_pass_by_time": overall_pass,
        "rows": summary_rows,
    }


def save_json(path: str, payload: object) -> None:
    destination = os.path.abspath(path)
    parent = os.path.dirname(destination) or "."
    os.makedirs(parent, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", dir=parent, delete=False) as handle:
        json.dump(payload, handle, indent=2)
        temp_path = handle.name
    os.replace(temp_path, destination)


def result_task_key(row: dict) -> Tuple[str, int, str]:
    return (str(row["model"]), int(row["depth"]), str(row["prompt"]))


def maybe_write_intermediate_artifacts(
    results: Sequence[dict],
    summary_json: str,
    fit_json: str,
    main_plot: str,
    breakdown_plot: str,
) -> None:
    if not results:
        return

    usable = [row for row in results if row["d_phys"] is not None]
    save_json(summary_json, summarize_results(results))
    if len(usable) < 3:
        return

    fit_params = fit_scaling_law(results)
    save_json(
        fit_json,
        {
            "k": fit_params[0],
            "alpha": fit_params[1],
            "beta": fit_params[2],
            "equation": "d_phys = k * S^(-alpha) * D^(beta)",
            "fit_method": "log_linear_least_squares",
            "bootstrap": None,
            "is_partial": True,
            "num_rows": len(results),
            "num_usable_rows": len(usable),
        },
    )
    placeholder_bootstrap = {
        "alpha_ci95": [fit_params[1], fit_params[1]],
        "beta_ci95": [fit_params[2], fit_params[2]],
    }
    save_main_plot(results, fit_params, placeholder_bootstrap, main_plot, False)
    save_breakdown_plot(results, breakdown_plot, False)


def maybe_write_time_budget_artifacts(
    results: Sequence[dict],
    summary_json: str,
    fit_json: str,
    main_plot: str,
    breakdown_plot: str,
    checkpoint_seconds: Sequence[float],
) -> None:
    if not results:
        return
    summary = summarize_time_budget_results(results, checkpoint_seconds)
    save_json(summary_json, summary)
    save_json(
        fit_json,
        {
            "experiment": "time_budget",
            "is_partial": True,
            "num_rows": len(results),
            "solve_rate": summary["solve_rate"],
            "overall_pass_by_time": summary["overall_pass_by_time"],
        },
    )
    save_time_budget_main_plot(results, checkpoint_seconds, main_plot, False)
    save_time_budget_breakdown_plot(results, checkpoint_seconds, breakdown_plot, False)


def apply_plot_style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, which="major", color="#d7dbe2", linewidth=0.7, alpha=0.65)
    ax.grid(True, which="minor", color="#eceff4", linewidth=0.45, alpha=0.5)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#6b7280")
    ax.spines["bottom"].set_color("#6b7280")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.tick_params(colors="#1f2937", labelsize=9.5)
    ax.xaxis.label.set_size(10.5)
    ax.yaxis.label.set_size(10.5)


def mean_and_se(values: Sequence[float]) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    mean_val = float(np.mean(arr))
    if arr.size == 1:
        return mean_val, 0.0
    se_val = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    return mean_val, se_val


def mean_and_ci(values: Sequence[float]) -> Tuple[float, float, float]:
    mean_val, se_val = mean_and_se(values)
    delta = 1.96 * se_val
    return mean_val, mean_val - delta, mean_val + delta


def model_family(model: str) -> str:
    if model.startswith("claude"):
        return "Anthropic"
    return "GPT-5" if model.startswith("gpt-5") else "GPT-4"


def add_box(ax: plt.Axes, xy: Tuple[float, float], wh: Tuple[float, float], title: str, body: Sequence[str], facecolor: str) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.0,
        edgecolor="#cad2db",
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + 0.03 * w, y + h - 0.18 * h, title, fontsize=10.1, fontweight="bold", va="top", color="#15212b")
    ax.text(
        x + 0.03 * w,
        y + h - 0.34 * h,
        "\n".join(body),
        fontsize=8.4,
        va="top",
        color="#334155",
        linespacing=1.35,
    )


def ordered_models(usable: Sequence[dict]) -> List[str]:
    return sorted(
        {str(row["model"]) for row in usable},
        key=lambda name: (
            min(float(row["size"]) for row in usable if str(row["model"]) == name),
            name,
        ),
    )


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def save_overview_plot(path: str) -> None:
    ensure_parent_dir(path)
    fig, ax = plt.subplots(figsize=(13.2, 5.2), facecolor="white")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Title block
    ax.text(0.03, 0.93, "QuantumPhysEval pipeline", fontsize=16.0, fontweight="bold", color="#122033")
    ax.text(
        0.03,
        0.875,
        "The benchmark is self-labeled: every prompt has exact quantum ground truth, so model error is measured directly rather than judged heuristically.",
        fontsize=9.6,
        color="#475569",
    )

    # Main stage boxes
    stages = [
        (
            (0.03, 0.33),
            (0.20, 0.40),
            "1. Construct tasks",
            [
                "Exact circuit, operator,",
                "measurement, and",
                "entanglement prompts",
            ],
            "#eef6ff",
            "#4f8fba",
        ),
        (
            (0.28, 0.33),
            (0.20, 0.40),
            "2. Compute targets",
            [
                "Analytic state vectors",
                "Exact unitaries and",
                "Born-rule probabilities",
            ],
            "#f5f7fb",
            "#7b8da3",
        ),
        (
            (0.53, 0.33),
            (0.20, 0.40),
            "3. Evaluate models",
            [
                "GPT and Claude families",
                "Depth settings 1, 2, 4",
                "Machine-readable outputs",
            ],
            "#fff5e9",
            "#d0874b",
        ),
        (
            (0.78, 0.33),
            (0.19, 0.40),
            "4. Score + analyze",
            [
                "$d_{phys}\\in[0,1]$",
                "Scaling fits, rankings,",
                "stress tests, repairs",
            ],
            "#eef9f2",
            "#6a9c78",
        ),
    ]

    for (x, y), (w, h), title, body, facecolor, accent in stages:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.0,
            edgecolor="#d4dce5",
            facecolor=facecolor,
        )
        ax.add_patch(patch)
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.016, y + h - 0.10),
                0.085,
                0.055,
                boxstyle="round,pad=0.01,rounding_size=0.025",
                linewidth=0,
                facecolor=accent,
                alpha=0.95,
            )
        )
        ax.text(x + 0.0585, y + h - 0.072, title.split(".")[0], fontsize=8.2, fontweight="bold", color="white", ha="center", va="center")
        ax.text(x + 0.018, y + h - 0.135, title, fontsize=10.8, fontweight="bold", va="top", color="#15212b")
        ax.text(
            x + 0.018,
            y + h - 0.235,
            "\n".join(body),
            fontsize=8.8,
            va="top",
            color="#334155",
            linespacing=1.42,
        )

    # Flow arrows
    arrow_y = 0.53
    for x0, x1 in [(0.235, 0.28), (0.485, 0.53), (0.735, 0.78)]:
        ax.add_patch(
            FancyArrowPatch(
                (x0, arrow_y),
                (x1, arrow_y),
                arrowstyle="-|>",
                mutation_scale=14,
                linewidth=1.6,
                color="#8a97a8",
            )
        )

    # Bottom synthesis strip
    strip = FancyBboxPatch(
        (0.03, 0.08),
        0.94,
        0.16,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        linewidth=1.0,
        edgecolor="#d4dce5",
        facecolor="#fbfcfe",
    )
    ax.add_patch(strip)
    ax.text(0.05, 0.205, "Outputs used in the paper", fontsize=10.6, fontweight="bold", color="#15212b", va="center")
    chips = [
        ("Pooled 2q benchmark", "#4f8fba"),
        ("Matched 4q stress test", "#d0874b"),
        ("Anthropic 8q complexity ladder", "#6e78b8"),
        ("Verifier-feedback ablations", "#6a9c78"),
    ]
    chip_x = 0.24
    for label, color in chips:
        width = 0.095 + 0.00285 * len(label)
        ax.add_patch(
            FancyBboxPatch(
                (chip_x, 0.155),
                width,
                0.055,
                boxstyle="round,pad=0.01,rounding_size=0.022",
                linewidth=0,
                facecolor=color,
                alpha=0.92,
            )
        )
        ax.text(
            chip_x + width / 2,
            0.1825,
            label,
            fontsize=7.7,
            color="white",
            ha="center",
            va="center",
            fontweight="bold",
        )
        chip_x += width + 0.018

    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_takeaway_plot(results: Sequence[dict], path: str, show_plot: bool) -> None:
    ensure_parent_dir(path)
    usable = [row for row in results if row["d_phys"] is not None]
    models = sorted(
        ordered_models(usable),
        key=lambda name: mean_and_ci([float(row["d_phys"]) for row in usable if str(row["model"]) == name])[0],
    )
    depths = sorted({int(row["depth"]) for row in usable})
    categories = sorted(
        CATEGORY_ORDER,
        key=lambda category: np.mean([float(row["d_phys"]) for row in usable if str(row["category"]) == category]),
        reverse=True,
    )

    fig = plt.figure(figsize=(12.4, 4.9), facecolor="white")
    grid = fig.add_gridspec(1, 3, width_ratios=[1.08, 0.85, 0.95], wspace=0.32)
    ax_models = fig.add_subplot(grid[0, 0])
    ax_depth = fig.add_subplot(grid[0, 1])
    ax_categories = fig.add_subplot(grid[0, 2])

    for ax in (ax_models, ax_depth, ax_categories):
        apply_plot_style(ax)

    model_means = []
    model_low = []
    model_high = []
    for model in models:
        values = [float(row["d_phys"]) for row in usable if str(row["model"]) == model]
        mean_val, ci_lo, ci_hi = mean_and_ci(values)
        model_means.append(mean_val)
        model_low.append(mean_val - ci_lo)
        model_high.append(ci_hi - mean_val)
    y = np.arange(len(models), dtype=float)
    ax_models.hlines(y, 0, model_means, color="#d5dce5", linewidth=2.4, zorder=1)
    ax_models.errorbar(
        model_means,
        y,
        xerr=[model_low, model_high],
        fmt="none",
        ecolor="#8a94a3",
        elinewidth=1.1,
        capsize=3,
        zorder=2,
    )
    ax_models.scatter(
        model_means,
        y,
        s=72,
        color=[MODEL_COLORS.get(model, "#4c4c4c") for model in models],
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    ax_models.set_yticks(y)
    ax_models.set_yticklabels([MODEL_LABELS.get(model, model) for model in models])
    ax_models.invert_yaxis()
    ax_models.set_xlim(0.25, 0.62)
    ax_models.set_xlabel("Mean normalized physics error")
    ax_models.set_title("A. Stronger models lower error", loc="left", fontsize=10.5, fontweight="bold", pad=7)

    depth_means = []
    depth_low = []
    depth_high = []
    for depth in depths:
        values = [float(row["d_phys"]) for row in usable if int(row["depth"]) == depth]
        mean_val, ci_lo, ci_hi = mean_and_ci(values)
        depth_means.append(mean_val)
        depth_low.append(mean_val - ci_lo)
        depth_high.append(ci_hi - mean_val)
    x_positions = np.arange(len(depths), dtype=float)
    ax_depth.plot(x_positions, depth_means, color="#6b7280", linewidth=1.7, zorder=2)
    ax_depth.scatter(
        x_positions,
        depth_means,
        s=70,
        color=[DEPTH_COLORS.get(depth, "#4c4c4c") for depth in depths],
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    ax_depth.errorbar(
        x_positions,
        depth_means,
        yerr=[depth_low, depth_high],
        fmt="none",
        ecolor="#8a8a8a",
        elinewidth=1.1,
        capsize=3,
        zorder=2,
    )
    ax_depth.set_xticks(x_positions)
    ax_depth.set_xticklabels([f"Depth {depth}" for depth in depths])
    ax_depth.set_ylim(0.39, 0.47)
    ax_depth.set_ylabel("Mean error")
    ax_depth.set_title("B. Prompted depth barely helps", loc="left", fontsize=10.5, fontweight="bold", pad=7)
    ax_depth.text(0.05, 0.04, "Depth effect is small\nand not robust.", transform=ax_depth.transAxes, fontsize=8.2, color="#475569")

    category_means = []
    category_low = []
    category_high = []
    for category in categories:
        values = [float(row["d_phys"]) for row in usable if str(row["category"]) == category]
        mean_val, ci_lo, ci_hi = mean_and_ci(values)
        category_means.append(mean_val)
        category_low.append(mean_val - ci_lo)
        category_high.append(ci_hi - mean_val)
    y_cat = np.arange(len(categories), dtype=float)
    category_colors = ["#b56576" if cat in {"circuit_evolution", "measurement_prediction"} else "#9db4c0" for cat in categories]
    ax_categories.barh(y_cat, category_means, color=category_colors, edgecolor="white", linewidth=0.9, height=0.58)
    ax_categories.errorbar(
        category_means,
        y_cat,
        xerr=[category_low, category_high],
        fmt="none",
        ecolor="#6f6f6f",
        elinewidth=1.0,
        capsize=3,
    )
    ax_categories.set_yticks(y_cat)
    ax_categories.set_yticklabels([SHORT_CATEGORY_LABELS[category] for category in categories])
    ax_categories.invert_yaxis()
    ax_categories.set_xlim(0.0, 0.8)
    ax_categories.set_xlabel("Mean normalized physics error")
    ax_categories.set_title("C. Dynamics tasks remain hardest", loc="left", fontsize=10.5, fontweight="bold", pad=7)

    fig.savefig(path, dpi=240, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def save_main_plot(
    results: Sequence[dict],
    fit_params: Tuple[float, float, float],
    bootstrap_summary: Dict[str, object],
    main_plot: str,
    show_plot: bool,
) -> None:
    ensure_parent_dir(main_plot)
    usable = [row for row in results if row["d_phys"] is not None]
    depths = sorted({int(row["depth"]) for row in usable})
    k, alpha, beta = fit_params
    fig = plt.figure(figsize=(10.8, 4.8), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.2, 0.92], wspace=0.28)
    ax_scale = fig.add_subplot(grid[0, 0])
    ax_depth = fig.add_subplot(grid[0, 1])

    apply_plot_style(ax_scale)
    apply_plot_style(ax_depth)

    models = ordered_models(usable)
    model_points = []
    for model in models:
        values = [float(row["d_phys"]) for row in usable if str(row["model"]) == model]
        mean_val, ci_lo, ci_hi = mean_and_ci(values)
        size = min(float(row["size"]) for row in usable if str(row["model"]) == model)
        model_points.append((model, size, mean_val, ci_lo, ci_hi))
    model_points.sort(key=lambda item: item[1])

    xs = np.asarray([item[1] for item in model_points], dtype=float)
    ys = np.asarray([item[2] for item in model_points], dtype=float)
    ci_los = np.asarray([item[3] for item in model_points], dtype=float)
    ci_his = np.asarray([item[4] for item in model_points], dtype=float)
    lower = ys - ci_los
    upper = ci_his - ys
    x_log = np.log10(xs)
    slope, intercept = np.polyfit(x_log, ys, 1)
    x_fit_log = np.linspace(np.min(x_log), np.max(x_log), 256)
    y_fit = slope * x_fit_log + intercept
    residual = ys - (slope * x_log + intercept)
    band = max(0.015, float(np.std(residual, ddof=1)) if residual.size > 2 else 0.02)

    ax_scale.errorbar(
        xs,
        ys,
        yerr=[lower, upper],
        fmt="none",
        ecolor="#8d8d8d",
        elinewidth=1.2,
        capsize=3,
        zorder=1,
    )
    span = 10 ** x_fit_log
    ax_scale.fill_between(
        span,
        np.clip(y_fit - band, 0.0, 1.0),
        np.clip(y_fit + band, 0.0, 1.0),
        color="#dbe7f4",
        alpha=0.9,
        zorder=1,
    )
    ax_scale.plot(span, y_fit, color="#1f4e79", linewidth=2.0, zorder=2)
    ax_scale.scatter(
        xs,
        ys,
        s=86,
        color=[MODEL_COLORS.get(model, "#4c4c4c") for model, *_ in model_points],
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    label_offsets = {
        "gpt-4o-mini": (6, -8),
        "gpt-4.1-mini": (6, -10),
        "gpt-4.1": (6, -9),
        "gpt-5.1": (6, -11),
        "gpt-5.2": (6, -8),
        "gpt-5.4-mini": (6, -10),
        "gpt-5.4": (6, -8),
    }
    for model, x, y, _, _ in model_points:
        dx, dy = label_offsets.get(model, (5, -8))
        ax_scale.annotate(
            MODEL_LABELS.get(model, model),
            (x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8.0,
            color="#374151",
        )

    ax_scale.set_xscale("log")
    ax_scale.set_xlabel("Model size proxy")
    ax_scale.set_ylabel("Mean normalized physics error")
    ax_scale.set_ylim(0.26, 0.62)
    ax_scale.set_title("A. Capability Trend", loc="left", fontsize=11.3, fontweight="bold", pad=7)
    alpha_ci = bootstrap_summary["alpha_ci95"]
    beta_ci = bootstrap_summary["beta_ci95"]
    ax_scale.text(
        0.03,
        0.05,
        f"Pooled fit: alpha={alpha:.2f} [{alpha_ci[0]:.2f}, {alpha_ci[1]:.2f}]\n"
        f"Depth effect: beta={beta:.2f} [{beta_ci[0]:.2f}, {beta_ci[1]:.2f}]",
        transform=ax_scale.transAxes,
        fontsize=8.2,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#d5dae0", "alpha": 0.97, "boxstyle": "round,pad=0.28"},
    )

    depth_means = []
    depth_lower = []
    depth_upper = []
    for depth in depths:
        values = [float(row["d_phys"]) for row in usable if int(row["depth"]) == depth]
        mean_val, ci_lo, ci_hi = mean_and_ci(values)
        depth_means.append(mean_val)
        depth_lower.append(mean_val - ci_lo)
        depth_upper.append(ci_hi - mean_val)

    x_positions = np.arange(len(depths), dtype=float)
    ax_depth.axhline(float(np.mean([float(row["d_phys"]) for row in usable])), color="#c7ccd4", linewidth=1.0, linestyle="--", zorder=1)
    ax_depth.plot(
        x_positions,
        depth_means,
        color="#6c7a89",
        linewidth=1.6,
        zorder=2,
    )
    ax_depth.scatter(
        x_positions,
        depth_means,
        s=72,
        color=[DEPTH_COLORS.get(depth, "#4c4c4c") for depth in depths],
        edgecolor="white",
        linewidth=0.9,
        zorder=3,
    )
    ax_depth.errorbar(
        x_positions,
        depth_means,
        yerr=[depth_lower, depth_upper],
        fmt="none",
        ecolor="#6f6f6f",
        elinewidth=1.1,
        capsize=3,
        zorder=2,
    )
    ax_depth.set_xticks(x_positions)
    ax_depth.set_xticklabels([f"Depth {depth}" for depth in depths])
    ax_depth.set_ylabel("Mean normalized physics error")
    ax_depth.set_xlim(-0.25, len(depths) - 0.75)
    ax_depth.set_ylim(0.39, 0.47)
    ax_depth.set_title("B. Prompted Depth", loc="left", fontsize=11.3, fontweight="bold", pad=7)
    for xpos, value in zip(x_positions, depth_means):
        ax_depth.text(xpos, value + 0.0045, f"{value:.3f}", ha="center", va="bottom", fontsize=8.2, color="#374151")

    fig.subplots_adjust(left=0.08, right=0.98, top=0.88, bottom=0.18, wspace=0.28)
    fig.savefig(main_plot, dpi=240, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def save_breakdown_plot(results: Sequence[dict], breakdown_plot: str, show_plot: bool) -> None:
    ensure_parent_dir(breakdown_plot)
    usable = [row for row in results if row["d_phys"] is not None]
    models = sorted(
        ordered_models(usable),
        key=lambda name: mean_and_ci([float(row["d_phys"]) for row in usable if str(row["model"]) == name])[0],
    )

    fig = plt.figure(figsize=(11.2, 5.2), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[0.9, 1.1], wspace=0.24)
    ax_mean = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[0, 1])

    category_palette = {
        "circuit_evolution": "#b56576",
        "measurement_prediction": "#e07a5f",
        "operator_composition": "#3d405b",
        "entanglement_classification": "#81b29a",
    }

    heat = np.zeros((len(models), len(CATEGORY_ORDER)), dtype=float)
    for row_idx, model in enumerate(models):
        for col_idx, category in enumerate(CATEGORY_ORDER):
            values = [
                float(row["d_phys"])
                for row in usable
                if str(row["model"]) == model and str(row["category"]) == category
            ]
            heat[row_idx, col_idx] = float(np.mean(values)) if values else 0.0

    apply_plot_style(ax_mean)
    category_means = []
    category_ci = []
    ordered_categories = sorted(
        CATEGORY_ORDER,
        key=lambda category: np.mean(
            [float(row["d_phys"]) for row in usable if str(row["category"]) == category]
        ),
        reverse=True,
    )
    for category in ordered_categories:
        values = [float(row["d_phys"]) for row in usable if str(row["category"]) == category]
        mean_val, ci_lo, ci_hi = mean_and_ci(values)
        category_means.append(mean_val)
        category_ci.append((mean_val - ci_lo, ci_hi - mean_val))
    y_positions = np.arange(len(ordered_categories), dtype=float)
    for ypos, category, value, ci in zip(y_positions, ordered_categories, category_means, category_ci):
        ax_mean.hlines(ypos, 0.0, value, color=category_palette[category], linewidth=5.0, alpha=0.24, zorder=1)
        ax_mean.hlines(ypos, value - ci[0], value + ci[1], color="#737373", linewidth=1.2, zorder=2)
        ax_mean.scatter(value, ypos, s=82, color=category_palette[category], edgecolor="white", linewidth=0.9, zorder=3)
    ax_mean.set_yticks(y_positions)
    ax_mean.set_yticklabels([CATEGORY_LABELS[category] for category in ordered_categories])
    ax_mean.invert_yaxis()
    ax_mean.set_xlabel("Mean normalized physics error")
    ax_mean.set_title("A. Category Difficulty", loc="left", fontsize=11.3, fontweight="bold", pad=7)
    ax_mean.set_xlim(0.0, 0.82)
    for ypos, value in zip(y_positions, category_means):
        ax_mean.text(value + 0.018, ypos, f"{value:.2f}", va="center", ha="left", fontsize=8.3, color="#374151")

    ax_heat.set_facecolor("white")
    im = ax_heat.imshow(heat, cmap="YlGnBu", vmin=0.0, vmax=max(float(np.max(heat)), 0.8), aspect="auto")
    ax_heat.set_xticks(np.arange(len(CATEGORY_ORDER)))
    ax_heat.set_xticklabels([SHORT_CATEGORY_LABELS[c] for c in CATEGORY_ORDER], rotation=0)
    ax_heat.set_yticks(np.arange(len(models)))
    ax_heat.set_yticklabels([MODEL_LABELS.get(model, model) for model in models])
    ax_heat.set_title("B. Model-by-Category Error", loc="left", fontsize=11.3, fontweight="bold", pad=7)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.tick_params(length=0)
    ax_heat.set_xticks(np.arange(-0.5, len(CATEGORY_ORDER), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.2)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    for row_idx in range(len(models)):
        for col_idx in range(len(CATEGORY_ORDER)):
            value = heat[row_idx, col_idx]
            text_color = "#143642" if value < 0.46 else "white"
            ax_heat.text(
                col_idx,
                row_idx,
                f"{value:.2f}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=8.1,
                fontweight="bold" if value == np.min(heat[row_idx]) else "normal",
            )
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Mean error", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)

    fig.subplots_adjust(left=0.16, right=0.98, top=0.88, bottom=0.16, wspace=0.28)
    fig.savefig(breakdown_plot, dpi=240, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def save_ablation_plot(results: Sequence[dict], ablation_plot: str, show_plot: bool) -> None:
    ensure_parent_dir(ablation_plot)
    usable = [row for row in results if row["d_phys"] is not None]
    models = sorted(
        ordered_models(usable),
        key=lambda name: mean_and_ci([float(row["d_phys"]) for row in usable if str(row["model"]) == name])[0],
    )
    depths = sorted({int(row["depth"]) for row in usable})

    fig = plt.figure(figsize=(11.2, 6.8), facecolor="white")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.96], wspace=0.25, hspace=0.35)
    ax_depth = fig.add_subplot(grid[0, 0])
    ax_family = fig.add_subplot(grid[0, 1])
    ax_rates = fig.add_subplot(grid[1, :])

    apply_plot_style(ax_depth)
    apply_plot_style(ax_family)
    apply_plot_style(ax_rates)

    # Panel A: model-by-depth means
    for model in models:
        means = []
        for depth in depths:
            values = [
                float(row["d_phys"])
                for row in usable
                if str(row["model"]) == model and int(row["depth"]) == depth
            ]
            means.append(float(np.mean(values)))
        ax_depth.plot(
            depths,
            means,
            color=MODEL_COLORS.get(model, "#4c4c4c"),
            linewidth=1.9,
            marker="o",
            markersize=5.2,
            markeredgecolor="white",
            markeredgewidth=0.9,
        )
        ax_depth.text(
            depths[-1] + 0.07,
            means[-1],
            MODEL_LABELS.get(model, model),
            color=MODEL_COLORS.get(model, "#4c4c4c"),
            fontsize=7.8,
            va="center",
        )
    ax_depth.set_xticks(depths)
    ax_depth.set_xticklabels([f"Depth {depth}" for depth in depths])
    ax_depth.set_ylabel("Mean normalized physics error")
    ax_depth.set_xlim(min(depths) - 0.1, max(depths) + 0.95)
    ax_depth.set_title("A. Model-by-Depth Ablation", loc="left", fontsize=11.3, fontweight="bold", pad=7)

    # Panel B: GPT-4 vs GPT-5 category comparison
    family_order = ["GPT-4", "GPT-5"]
    category_means_by_family: Dict[str, List[float]] = {family: [] for family in family_order}
    for family in family_order:
        for category in CATEGORY_ORDER:
            values = [
                float(row["d_phys"])
                for row in usable
                if (
                    ((model_family(str(row["model"])) == "GPT-5" and family == "GPT-5")
                    or (model_family(str(row["model"])) == "GPT-4" and family == "GPT-4"))
                    and str(row["category"]) == category
                )
            ]
            category_means_by_family[family].append(float(np.mean(values)))
    y_positions = np.arange(len(CATEGORY_ORDER), dtype=float)
    for idx, category in enumerate(CATEGORY_ORDER):
        x4 = category_means_by_family["GPT-4"][idx]
        x5 = category_means_by_family["GPT-5"][idx]
        ax_family.hlines(y_positions[idx], x5, x4, color="#cfd5dd", linewidth=2.6, zorder=1)
        ax_family.scatter(x4, y_positions[idx], s=64, color=FAMILY_COLORS["GPT-4"], edgecolor="white", linewidth=0.8, zorder=2)
        ax_family.scatter(x5, y_positions[idx], s=64, color=FAMILY_COLORS["GPT-5"], edgecolor="white", linewidth=0.8, zorder=2)
        ax_family.text(max(x4, x5) + 0.015, y_positions[idx], f"-{(x4 - x5):.2f}", va="center", fontsize=8.0, color="#374151")
    ax_family.set_yticks(y_positions)
    ax_family.set_yticklabels([SHORT_CATEGORY_LABELS[c] for c in CATEGORY_ORDER])
    ax_family.invert_yaxis()
    ax_family.set_xlabel("Mean normalized physics error")
    ax_family.set_title("B. GPT-4 vs. GPT-5 by Category", loc="left", fontsize=11.3, fontweight="bold", pad=7)
    ax_family.set_xlim(0.0, 0.8)
    ax_family.scatter([], [], s=64, color=FAMILY_COLORS["GPT-4"], label="GPT-4")
    ax_family.scatter([], [], s=64, color=FAMILY_COLORS["GPT-5"], label="GPT-5")
    ax_family.legend(frameon=False, fontsize=8.8, loc="lower right")

    # Panel C: exact-correct vs full-error by model
    y_positions = np.arange(len(models))
    zero_rates = []
    full_error_rates = []
    for model in models:
        values = np.asarray(
            [float(row["d_phys"]) for row in usable if str(row["model"]) == model],
            dtype=float,
        )
        zero_rates.append(float(np.mean(values == 0.0)))
        full_error_rates.append(float(np.mean(values >= 0.999999)))
    for idx in range(len(models)):
        ax_rates.plot(
            [zero_rates[idx], full_error_rates[idx]],
            [idx, idx],
            color="#c8c1b2",
            linewidth=2.8,
            solid_capstyle="round",
        )
    ax_rates.scatter(zero_rates, y_positions, s=72, color="#2a9d8f", edgecolor="white", linewidth=0.9, label="Exact-correct rate", zorder=3)
    ax_rates.scatter(full_error_rates, y_positions, s=72, color="#c75c5c", edgecolor="white", linewidth=0.9, label="Full-error rate", zorder=3)
    ax_rates.set_yticks(y_positions)
    ax_rates.set_yticklabels([MODEL_LABELS.get(model, model) for model in models])
    ax_rates.invert_yaxis()
    ax_rates.set_xlim(0.0, 0.52)
    ax_rates.set_xlabel("Rate")
    ax_rates.set_title("C. Exact-Correct vs. Full-Error Rate", loc="left", fontsize=11.3, fontweight="bold", pad=7)
    ax_rates.legend(frameon=False, fontsize=9, loc="lower right", ncol=2)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.90, bottom=0.11, wspace=0.27, hspace=0.36)
    fig.savefig(ablation_plot, dpi=240, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def save_time_budget_main_plot(
    results: Sequence[dict],
    checkpoint_seconds: Sequence[float],
    main_plot: str,
    show_plot: bool,
) -> None:
    ensure_parent_dir(main_plot)
    rows = [row for row in results]
    checkpoint_values = sorted({float(x) for x in checkpoint_seconds if float(x) > 0.0})
    checkpoint_labels = [format_checkpoint_label(x) for x in checkpoint_values]
    models = sorted(
        {str(row["model"]) for row in rows},
        key=lambda name: np.mean([1.0 if row.get("solved") else 0.0 for row in rows if str(row["model"]) == name]),
        reverse=True,
    )
    categories = sorted(
        CATEGORY_ORDER,
        key=lambda category: np.mean([1.0 if row.get("solved") else 0.0 for row in rows if str(row["category"]) == category]),
    )

    fig = plt.figure(figsize=(11.4, 5.2), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.95], wspace=0.28)
    ax_curves = fig.add_subplot(grid[0, 0])
    ax_solve = fig.add_subplot(grid[0, 1])
    apply_plot_style(ax_curves)
    apply_plot_style(ax_solve)

    for model in models:
        pass_rates = []
        model_rows = [row for row in rows if str(row["model"]) == model]
        for label in checkpoint_labels:
            pass_rates.append(float(np.mean([1.0 if row.get("pass_by_time", {}).get(label, False) else 0.0 for row in model_rows])))
        ax_curves.plot(
            checkpoint_values,
            pass_rates,
            marker="o",
            markersize=5.0,
            linewidth=2.0,
            color=MODEL_COLORS.get(model, "#4c4c4c"),
            label=MODEL_LABELS.get(model, model),
        )
    ax_curves.set_xlabel("Wall-clock budget (s)")
    ax_curves.set_ylabel("Solved fraction")
    ax_curves.set_ylim(0.0, 1.02)
    ax_curves.set_xticks(checkpoint_values)
    ax_curves.set_title("A. Pass@time by model", loc="left", fontsize=11.3, fontweight="bold", pad=7)
    ax_curves.legend(frameon=False, fontsize=8.4, loc="lower right")

    final_label = checkpoint_labels[-1]
    solve_rates = [
        float(np.mean([1.0 if row.get("pass_by_time", {}).get(final_label, False) else 0.0 for row in rows if str(row["category"]) == category]))
        for category in categories
    ]
    colors = ["#b56576" if category in {"circuit_evolution", "measurement_prediction"} else "#8fb6c9" for category in categories]
    y_pos = np.arange(len(categories), dtype=float)
    ax_solve.barh(y_pos, solve_rates, color=colors, edgecolor="white", linewidth=0.9, height=0.58)
    ax_solve.set_yticks(y_pos)
    ax_solve.set_yticklabels([SHORT_CATEGORY_LABELS[category] for category in categories])
    ax_solve.invert_yaxis()
    ax_solve.set_xlim(0.0, 1.0)
    ax_solve.set_xlabel(f"Solved by {final_label}")
    ax_solve.set_title("B. Hard tasks remain hard", loc="left", fontsize=11.3, fontweight="bold", pad=7)
    for ypos, value in zip(y_pos, solve_rates):
        ax_solve.text(min(value + 0.02, 0.98), ypos, f"{value:.2f}", va="center", ha="left", fontsize=8.2, color="#374151")

    fig.subplots_adjust(left=0.10, right=0.98, top=0.88, bottom=0.16, wspace=0.26)
    fig.savefig(main_plot, dpi=240, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def save_time_budget_breakdown_plot(
    results: Sequence[dict],
    checkpoint_seconds: Sequence[float],
    breakdown_plot: str,
    show_plot: bool,
) -> None:
    ensure_parent_dir(breakdown_plot)
    rows = [row for row in results]
    checkpoint_values = sorted({float(x) for x in checkpoint_seconds if float(x) > 0.0})
    checkpoint_label = format_checkpoint_label(checkpoint_values[-1])
    models = sorted(
        {str(row["model"]) for row in rows},
        key=lambda name: np.mean([1.0 if row.get("pass_by_time", {}).get(checkpoint_label, False) else 0.0 for row in rows if str(row["model"]) == name]),
        reverse=True,
    )

    heat = np.zeros((len(models), len(CATEGORY_ORDER)), dtype=float)
    for row_idx, model in enumerate(models):
        for col_idx, category in enumerate(CATEGORY_ORDER):
            subset = [
                row for row in rows
                if str(row["model"]) == model and str(row["category"]) == category
            ]
            if subset:
                heat[row_idx, col_idx] = float(
                    np.mean([1.0 if row.get("pass_by_time", {}).get(checkpoint_label, False) else 0.0 for row in subset])
                )

    fig = plt.figure(figsize=(11.4, 5.2), facecolor="white")
    grid = fig.add_gridspec(1, 2, width_ratios=[1.02, 0.98], wspace=0.26)
    ax_time = fig.add_subplot(grid[0, 0])
    ax_heat = fig.add_subplot(grid[0, 1])
    apply_plot_style(ax_time)

    solve_time_rows = []
    for model in models:
        solved_times = [float(row["solve_time_s"]) for row in rows if str(row["model"]) == model and row.get("solve_time_s") is not None]
        if solved_times:
            solve_time_rows.append((model, float(np.mean(solved_times)), float(np.median(solved_times))))
        else:
            solve_time_rows.append((model, math.nan, math.nan))
    y = np.arange(len(models), dtype=float)
    means = [item[1] for item in solve_time_rows]
    medians = [item[2] for item in solve_time_rows]
    for idx, (mean_val, median_val) in enumerate(zip(means, medians)):
        if not math.isnan(mean_val) and not math.isnan(median_val):
            ax_time.plot([mean_val, median_val], [idx, idx], color="#d5d9e0", linewidth=2.0, zorder=2)
    finite_mean_points = [(x, ypos) for x, ypos in zip(means, y) if not math.isnan(x)]
    finite_median_points = [(x, ypos) for x, ypos in zip(medians, y) if not math.isnan(x)]
    if finite_mean_points:
        ax_time.scatter(
            [item[0] for item in finite_mean_points],
            [item[1] for item in finite_mean_points],
            s=72,
            color="#355070",
            edgecolor="white",
            linewidth=0.9,
            label="Mean solve time",
            zorder=3,
        )
    if finite_median_points:
        ax_time.scatter(
            [item[0] for item in finite_median_points],
            [item[1] for item in finite_median_points],
            s=72,
            color="#b56576",
            edgecolor="white",
            linewidth=0.9,
            label="Median solve time",
            zorder=3,
        )
    ax_time.set_yticks(y)
    ax_time.set_yticklabels([MODEL_LABELS.get(model, model) for model in models])
    ax_time.invert_yaxis()
    ax_time.set_xlabel("Solve time among solved prompts (s)")
    ax_time.set_title("A. Mean/median time to correctness", loc="left", fontsize=11.3, fontweight="bold", pad=7)
    ax_time.legend(frameon=False, fontsize=8.4, loc="lower right")

    ax_heat.set_facecolor("white")
    im = ax_heat.imshow(heat, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax_heat.set_xticks(np.arange(len(CATEGORY_ORDER)))
    ax_heat.set_xticklabels([SHORT_CATEGORY_LABELS[c] for c in CATEGORY_ORDER], rotation=0)
    ax_heat.set_yticks(np.arange(len(models)))
    ax_heat.set_yticklabels([MODEL_LABELS.get(model, model) for model in models])
    ax_heat.set_title(f"B. Solved by {checkpoint_label}", loc="left", fontsize=11.3, fontweight="bold", pad=7)
    for spine in ax_heat.spines.values():
        spine.set_visible(False)
    ax_heat.tick_params(length=0)
    ax_heat.set_xticks(np.arange(-0.5, len(CATEGORY_ORDER), 1), minor=True)
    ax_heat.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
    ax_heat.grid(which="minor", color="white", linewidth=1.2)
    ax_heat.tick_params(which="minor", bottom=False, left=False)
    for row_idx in range(len(models)):
        for col_idx in range(len(CATEGORY_ORDER)):
            value = heat[row_idx, col_idx]
            text_color = "#143642" if value < 0.60 else "white"
            ax_heat.text(col_idx, row_idx, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8.1, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label("Solved fraction", fontsize=9.5)
    cbar.ax.tick_params(labelsize=8.5)

    fig.subplots_adjust(left=0.12, right=0.98, top=0.88, bottom=0.16, wspace=0.28)
    fig.savefig(breakdown_plot, dpi=240, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close(fig)


def load_results(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    args = parse_args()
    checkpoint_seconds = sorted({float(x) for x in args.checkpoint_seconds if float(x) > 0.0})
    if args.reuse_results_json:
        results = load_results(args.reuse_results_json)
    else:
        prompts = generate_prompts(args.n_per_category, args.seed, num_qubits=args.num_qubits)
        models = parse_models(args.model)
        if args.validate_models:
            models = validate_models(
                provider=args.provider,
                models=models,
                temperature=args.temperature,
                request_timeout=args.request_timeout,
                skip_unavailable_models=args.skip_unavailable_models,
            )
            if not models:
                raise RuntimeError("No validated models remain after filtering unavailable IDs.")
        if args.experiment == "time_budget":
            results = run_time_budget_benchmark(
                provider=args.provider,
                prompts=prompts,
                models=models,
                depths=args.reasoning_depths,
                temperature=args.temperature,
                results_json=args.results_json,
                summary_json=args.summary_json,
                fit_json=args.fit_json,
                main_plot=args.main_plot,
                breakdown_plot=args.breakdown_plot,
                checkpoint_every=args.checkpoint_every,
                plot_every=args.plot_every,
                max_workers=args.max_workers,
                request_timeout=args.request_timeout,
                time_budget_seconds=args.time_budget_seconds,
                checkpoint_seconds=checkpoint_seconds,
                max_attempts=args.max_attempts,
                repair_mode=args.repair_mode,
            )
        else:
            results = run_benchmark(
                provider=args.provider,
                prompts=prompts,
                models=models,
                depths=args.reasoning_depths,
                temperature=args.temperature,
                results_json=args.results_json,
                summary_json=args.summary_json,
                fit_json=args.fit_json,
                main_plot=args.main_plot,
                breakdown_plot=args.breakdown_plot,
                checkpoint_every=args.checkpoint_every,
                plot_every=args.plot_every,
                max_workers=args.max_workers,
                request_timeout=args.request_timeout,
            )

    if args.experiment == "time_budget":
        summary = summarize_time_budget_results(results, checkpoint_seconds)
        fit_payload = {
            "experiment": "time_budget",
            "time_budget_s": args.time_budget_seconds,
            "repair_mode": args.repair_mode,
            "checkpoint_seconds": checkpoint_seconds,
            "solve_rate": summary["solve_rate"],
            "mean_solve_time_s": summary["mean_solve_time_s"],
            "median_solve_time_s": summary["median_solve_time_s"],
            "overall_pass_by_time": summary["overall_pass_by_time"],
            "is_partial": False,
        }
        save_json(args.summary_json, summary)
        save_json(args.fit_json, fit_payload)
        save_time_budget_main_plot(results, checkpoint_seconds, args.main_plot, args.show_plot)
        save_time_budget_breakdown_plot(results, checkpoint_seconds, args.breakdown_plot, args.show_plot)
        print(f"Time-budget solve rate: {summary['solve_rate']:.6g}")
    else:
        fit_params = fit_scaling_law(results)
        bootstrap_summary = bootstrap_fit(results, args.bootstrap_samples, args.seed + 101)
        summary = summarize_results(results)
        fit_payload = {
            "k": fit_params[0],
            "alpha": fit_params[1],
            "beta": fit_params[2],
            "equation": "d_phys = k * S^(-alpha) * D^(beta)",
            "fit_method": "log_linear_least_squares",
            "bootstrap": bootstrap_summary,
            "is_partial": False,
        }
        save_json(args.summary_json, summary)
        save_json(args.fit_json, fit_payload)
        save_main_plot(results, fit_params, bootstrap_summary, args.main_plot, args.show_plot)
        save_breakdown_plot(results, args.breakdown_plot, args.show_plot)
        print(f"Scaling law fit: d_phys = {fit_params[0]:.6g} * S^-{fit_params[1]:.6g} * D^{fit_params[2]:.6g}")

    print(f"Saved results to {args.results_json if not args.reuse_results_json else args.reuse_results_json}")
    print(f"Saved summary to {args.summary_json}")
    print(f"Saved fit to {args.fit_json}")
    print(f"Saved main plot to {args.main_plot}")
    print(f"Saved breakdown plot to {args.breakdown_plot}")


if __name__ == "__main__":
    main()
