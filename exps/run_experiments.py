import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import platform
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from string import Template


ENVIRONMENTS = ("CARTPOLE", "PENDULUM", "PENDULUM_STOCHASTIC")
ALGORITHMS = (
    "NELDER_MEAD",
    "STOCHASTIC_NELDER_MEAD",
    "CHANG_NELDER_MEAD",
    "GPS",
    "GPS_SURROGATE_QUADRATIC",
    "GPS_SURROGATE_RBF",
    "GPS_RINOTT",
    "PSO",
    "CMA_ES",
    "OPENAI_ES",
    "SIMULATED_ANNEALING",
)
STOPPING_CRITERIA = (
    "MAXIMUM_ITERATIONS",
    "MAXIMUM_EVALUATIONS",
    "NO_IMPROVEMENT",
    "TARGET_VALUE",
)
CPP_TYPES = {
    "NelderMeadHyperparameters",
    "StochasticHeuristicNelderMeadHyperparameters",
    "ChangStochasticNelderMeadHyperparameters",
    "GeneralizedPatternSearchHyperparameters",
    "RinottRankingSelectionGPSHyperparameters",
    "SimulatedAnnealingHyperparameters",
    "PSOHyperparameters",
    "CMAESHyperparameters",
    "OpenAIESHyperparameters",
}
POLICY_TOKENS = {
    "ActivationType::Linear",
    "ActivationType::Tanh",
    "ActivationType::Sigmoid",
    "ActivationType::ReLu",
    "RegularizationType::L1_Regularization",
    "RegularizationType::L2_Regularization",
    "FeedForwardInitializationType::FanInUniform",
    "FeedForwardInitializationType::XavierUniform",
}
ENVIRONMENT_IDS = {name: index for index, name in enumerate(ENVIRONMENTS)}
ALGORITHM_IDS = {name: index for index, name in enumerate(ALGORITHMS)}
STOPPING_IDS = {name: index for index, name in enumerate(STOPPING_CRITERIA)}


def object_value(value, name):
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def array_value(value, name):
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty array")
    return value


def integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer greater than or equal to {minimum}")
    return value


def number(value, name, minimum=None, strict=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if minimum is not None and (result <= minimum if strict else result < minimum):
        raise ValueError(f"{name} is outside its supported range")
    return result


def boolean(value, name):
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be boolean")
    return value


def cpp_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("configuration numbers must be finite")
        return format(value, ".17g")
    if isinstance(value, str):
        return value
    raise ValueError("configuration axes support only strings, booleans and numbers")


def merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def resolve(table, name, stack=()):
    if name in stack:
        raise ValueError(f"cyclic base chain for {name}")
    entry = deepcopy(object_value(table.get(name), name))
    base = entry.pop("base", None)
    if base is None:
        return entry
    if not isinstance(base, str) or base not in table:
        raise ValueError(f"invalid base for {name}")
    return merge(resolve(table, base, stack + (name,)), entry)


def master_seed(value):
    if not isinstance(value, str):
        raise ValueError("master_seed must be a string")
    try:
        parsed = int(value, 0)
    except ValueError as error:
        raise ValueError("master_seed must be an integer literal") from error
    if parsed < 0 or parsed > (1 << 64) - 1:
        raise ValueError("master_seed must fit uint64_t")
    return f"0x{parsed:X}"


def checked_token(value, name):
    if value not in POLICY_TOKENS:
        raise ValueError(f"unsupported {name}")
    return value


def network_values(environment):
    network = object_value(environment.get("network"), "network")
    hidden = array_value(network.get("hidden_layers"), "network.hidden_layers")
    hidden = [integer(value, "hidden layer") for value in hidden]
    return (
        integer(network.get("input_size"), "network.input_size"),
        hidden,
        integer(network.get("output_size"), "network.output_size"),
    )


def parameter_count(environment, use_bias):
    input_size, hidden, output_size = network_values(environment)
    widths = [input_size, *hidden, output_size]
    return sum(
        left * right + (right if use_bias else 0)
        for left, right in zip(widths, widths[1:])
    )


def render_algorithm(configuration, environment_name, algorithm_name, dimensions):
    algorithms = object_value(configuration.get("algorithms"), "algorithms")
    algorithm = resolve(algorithms, algorithm_name)
    overrides = object_value(algorithm.pop("environment_overrides", {}), "environment_overrides")
    if environment_name in overrides:
        algorithm = merge(algorithm, object_value(overrides[environment_name], "environment override"))

    cpp_type = algorithm.get("cpp_type")
    if cpp_type not in CPP_TYPES:
        raise ValueError(f"unsupported C++ hyperparameter type for {algorithm_name}")
    axes = object_value(algorithm.get("axes", {}), "axes")
    fixed = object_value(algorithm.get("fixed", {}), "fixed")
    assignments = array_value(algorithm.get("assign"), "assign")
    if any(not isinstance(line, str) for line in assignments):
        raise ValueError("assign must contain strings")

    axis_names = list(axes)
    axis_values = [array_value(axes[name], f"axis {name}") for name in axis_names]
    entries = []
    for combination in itertools.product(*axis_values):
        raw = {**fixed, **dict(zip(axis_names, combination)), "parameter_count": dimensions}
        formatted = {
            name: cpp_value(value).replace("{parameter_count}", str(dimensions))
            for name, value in raw.items()
        }
        try:
            lines = [line.format_map(formatted) for line in assignments]
        except (KeyError, ValueError) as error:
            raise ValueError(f"invalid assignment for {algorithm_name}: {error}") from error
        body = "\n".join(f"      {line}" for line in lines)
        cpp = (
            "    {\n"
            f"      {cpp_type} hyperparameters;\n"
            f"{body}\n"
            "      append( hyperparameters );\n"
            "    }"
        )
        metadata = {
            name: value.replace("{parameter_count}", str(dimensions))
            if isinstance(value, str)
            else value
            for name, value in {**fixed, **dict(zip(axis_names, combination))}.items()
        }
        entries.append({"cpp": cpp, "configuration": metadata})
    return entries


def environment_fragments(name, environment):
    if name == "CARTPOLE":
        scenario = object_value(environment.get("scenario"), "CARTPOLE.scenario")
        low = cpp_value(number(scenario.get("state_minimum"), "state_minimum"))
        high = cpp_value(number(scenario.get("state_maximum"), "state_maximum"))
        return {
            "environment_header": "CartPole.hpp",
            "environment_aliases": (
                "  using Scenario = exps::cartpole::Scenario;\n"
                "  using Observation = exps::cartpole::Observation;\n"
                "  using Action = exps::cartpole::Action;\n"
                "  using Environment = exps::cartpole::CartPoleEnvironment;"
            ),
            "environment_seed_tag": "0xCA47F01E",
            "scenario_factory": (
                "  std::vector< Scenario > makeScenarios( std::size_t count, std::uint64_t seed )\n"
                "  {\n"
                "    std::mt19937_64 generator( seed );\n"
                f"    std::uniform_real_distribution< double > distribution( {low}, {high} );\n"
                "    std::vector< Scenario > scenarios;\n"
                "    scenarios.reserve( count );\n"
                "    for ( std::size_t index = 0; index < count; ++index )\n"
                "    {\n"
                "      scenarios.push_back( Scenario{ distribution( generator ), distribution( generator ),\n"
                "                                     distribution( generator ), distribution( generator ) } );\n"
                "    }\n"
                "    return scenarios;\n"
                "  }"
            ),
            "environment_factory": (
                "  Environment makeEnvironment( std::uint64_t )\n"
                "  {\n"
                "    return Environment{};\n"
                "  }"
            ),
        }

    scenario = object_value(environment.get("scenario"), f"{name}.scenario")
    angle_low = cpp_value(number(scenario.get("angle_minimum"), "angle_minimum"))
    angle_high = cpp_value(number(scenario.get("angle_maximum"), "angle_maximum"))
    velocity_low = cpp_value(number(scenario.get("velocity_minimum"), "velocity_minimum"))
    velocity_high = cpp_value(number(scenario.get("velocity_maximum"), "velocity_maximum"))
    stochastic = name == "PENDULUM_STOCHASTIC"
    cpp_environment = (
        "exps::pendulum::StochasticPendulumEnvironment"
        if stochastic else "exps::pendulum::PendulumEnvironment"
    )
    if stochastic:
        noise = cpp_value(number(environment.get("torque_noise_stddev"), "torque_noise_stddev", 0.0, True))
        factory = (
            "  Environment makeEnvironment( std::uint64_t seed )\n"
            "  {\n"
            f"    return Environment{{ seed, {noise} }};\n"
            "  }"
        )
    else:
        factory = (
            "  Environment makeEnvironment( std::uint64_t )\n"
            "  {\n"
            "    return Environment{};\n"
            "  }"
        )
    return {
        "environment_header": "PendulumStochastic.hpp" if stochastic else "Pendulum.hpp",
        "environment_aliases": (
            "  using Scenario = exps::pendulum::Scenario;\n"
            "  using Observation = exps::pendulum::Observation;\n"
            "  using Action = exps::pendulum::Action;\n"
            f"  using Environment = {cpp_environment};"
        ),
        "environment_seed_tag": "0x0EAD01A1",
        "scenario_factory": (
            "  std::vector< Scenario > makeScenarios( std::size_t count, std::uint64_t seed )\n"
            "  {\n"
            "    std::mt19937_64 generator( seed );\n"
            f"    std::uniform_real_distribution< double > angle( {angle_low}, {angle_high} );\n"
            f"    std::uniform_real_distribution< double > velocity( {velocity_low}, {velocity_high} );\n"
            "    std::vector< Scenario > scenarios;\n"
            "    scenarios.reserve( count );\n"
            "    for ( std::size_t index = 0; index < count; ++index )\n"
            "    {\n"
            "      scenarios.push_back( Scenario{ angle( generator ), velocity( generator ) } );\n"
            "    }\n"
            "    return scenarios;\n"
            "  }"
        ),
        "environment_factory": factory,
    }


def external_stopping(configuration, environment, algorithm, criterion):
    model_selection = object_value(configuration.get("model_selection"), "model_selection")
    table = object_value(model_selection.get("optimizer_stopping"), "optimizer_stopping")
    default = object_value(table.get("default"), "optimizer_stopping.default")
    special_job = environment == "PENDULUM_STOCHASTIC" and algorithm == "GPS_RINOTT"
    special = object_value(table.get("PENDULUM_STOCHASTIC_GPS_RINOTT", {}), "special stopping")
    safety = integer(
        special.get("safety_evaluations") if special_job else default.get("safety_evaluations"),
        "safety_evaluations",
    )
    if criterion == "MAXIMUM_ITERATIONS":
        iterations = integer(default.get("maximum_iterations"), "maximum_iterations")
        if special_job:
            return (
                "    return {\n"
                f"      MaximumIterationsStoppingConfiguration{{ {iterations} }},\n"
                f"      MaximumEvaluationsStoppingConfiguration{{ {safety} }}\n"
                "    };"
            )
        return f"    return {{ MaximumIterationsStoppingConfiguration{{ {iterations} }} }};"
    if criterion == "MAXIMUM_EVALUATIONS":
        evaluations = integer(
            special.get("maximum_evaluations") if special_job else default.get("maximum_evaluations"),
            "maximum_evaluations",
        )
        return f"    return {{ MaximumEvaluationsStoppingConfiguration{{ {evaluations} }} }};"
    if criterion == "NO_IMPROVEMENT":
        patience = integer(default.get("no_improvement_patience"), "no_improvement_patience")
        threshold = cpp_value(number(default.get("no_improvement_threshold"), "no_improvement_threshold", 0.0))
        return (
            "    return {\n"
            f"      NoImprovementStoppingConfiguration{{ {patience}, {threshold} }},\n"
            f"      MaximumEvaluationsStoppingConfiguration{{ {safety} }}\n"
            "    };"
        )
    environments = object_value(configuration.get("environments"), "environments")
    target = cpp_value(number(resolve(environments, environment).get("target_return"), "target_return"))
    return (
        "    return {\n"
        f"      TargetValueStoppingConfiguration{{ {target} }},\n"
        f"      MaximumEvaluationsStoppingConfiguration{{ {safety} }}\n"
        "    };"
    )


def render_source(configuration, template, environment_name, algorithm_name, criterion):
    environments = object_value(configuration.get("environments"), "environments")
    environment = resolve(environments, environment_name)
    policy = object_value(configuration.get("policy"), "policy")
    use_bias = boolean(policy.get("use_bias"), "policy.use_bias")
    dimensions = parameter_count(environment, use_bias)
    registry = render_algorithm(configuration, environment_name, algorithm_name, dimensions)
    if not registry:
        raise ValueError("configuration registry is empty")

    input_size, hidden, output_size = network_values(environment)
    scenarios = object_value(configuration.get("scenarios"), "scenarios")
    selection = object_value(configuration.get("model_selection"), "model_selection")
    values = environment_fragments(environment_name, environment)
    values.update({
        "master_seed": master_seed(configuration.get("master_seed")),
        "environment_id": str(ENVIRONMENT_IDS[environment_name]),
        "algorithm_id": str(ALGORITHM_IDS[algorithm_name]),
        "stopping_id": str(STOPPING_IDS[criterion]),
        "training_scenarios": str(integer(scenarios.get("training"), "scenarios.training")),
        "validation_scenarios": str(integer(scenarios.get("validation"), "scenarios.validation")),
        "test_scenarios": str(integer(scenarios.get("test"), "scenarios.test")),
        "runs_per_scenario": str(integer(environment.get("runs_per_scenario"), "runs_per_scenario")),
        "restart_count": str(integer(selection.get("restarts"), "model_selection.restarts")),
        "chang_job": "true" if algorithm_name == "CHANG_NELDER_MEAD" else "false",
        "chang_objective_offset": cpp_value(number(environment.get("chang_objective_offset"), "chang offset")),
        "network_input_size": str(input_size),
        "network_hidden_layers": "{ " + ", ".join(str(value) for value in hidden) + " }",
        "network_output_size": str(output_size),
        "hidden_activation": checked_token(policy.get("hidden_activation"), "hidden activation"),
        "output_activation": checked_token(policy.get("output_activation"), "output activation"),
        "use_bias": cpp_value(use_bias),
        "regularization_type": checked_token(policy.get("regularization_type"), "regularization type"),
        "regularization_coefficient": cpp_value(
            number(policy.get("regularization_coefficient"), "regularization coefficient", 0.0)
        ),
        "initialization_type": checked_token(policy.get("initialization"), "initialization"),
        "external_stopping": external_stopping(configuration, environment_name, algorithm_name, criterion),
        "registry_entries": "\n\n".join(entry["cpp"] for entry in registry),
        "maximum_configurations": str(integer(selection.get("maximum_configurations"), "maximum configurations")),
        "model_selection_patience": str(integer(selection.get("no_improvement_patience"), "patience")),
        "model_selection_threshold": cpp_value(
            number(selection.get("no_improvement_threshold"), "selection threshold", 0.0)
        ),
        "model_selection_batch_size": str(integer(selection.get("batch_size"), "batch size")),
    })
    return Template(template).substitute(values), registry, environment


def parse_output(output, registry_size):
    records = {}
    failures = {}
    summary = None
    for line in output.splitlines():
        fields = line.split("\t")
        if fields[0] == "RECORD" and len(fields) == 9:
            configuration_id = int(fields[1])
            records[configuration_id] = {
                "validation_mean": float(fields[2]),
                "validation_variance": float(fields[3]),
                "selection_score": float(fields[4]),
                "episodes": int(fields[5]),
                "simulation_steps": int(fields[6]),
                "policy_inferences": int(fields[7]),
                "network_macs": int(fields[8]),
            }
        elif fields[0] == "FAILURE" and len(fields) == 4:
            configuration_id = int(fields[1])
            failures[configuration_id] = {"kind": fields[2], "message": fields[3]}
        elif fields[0] == "SUMMARY" and len(fields) == 12 and summary is None:
            summary = {
                "best_configuration_id": int(fields[1]),
                "test_mean": float(fields[2]),
                "test_standard_deviation": float(fields[3]),
                "attempted_configurations": int(fields[4]),
                "termination_reason": fields[5],
                "final_training_iterations": int(fields[6]),
                "final_training_evaluations": int(fields[7]),
                "total_episodes": int(fields[8]),
                "total_simulation_steps": int(fields[9]),
                "total_policy_inferences": int(fields[10]),
                "total_network_macs": int(fields[11]),
            }
        elif fields[0] in ("RECORD", "FAILURE", "SUMMARY"):
            raise RuntimeError("malformed worker output")
    identifiers = set(records) | set(failures)
    if any(value < 0 or value >= registry_size for value in identifiers):
        raise RuntimeError("worker returned an invalid configuration id")
    if summary is None or summary["best_configuration_id"] not in records:
        raise RuntimeError("worker returned no valid summary")
    return records, failures, summary


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_results(path, registry, records, failures):
    fields = (
        "configuration_id", "status", "validation_mean", "validation_variance",
        "selection_score", "episodes", "simulation_steps", "policy_inferences",
        "network_macs", "failure_kind", "message", "configuration",
    )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for configuration_id in sorted(set(records) | set(failures)):
            row = {
                "configuration_id": configuration_id,
                "configuration": json.dumps(
                    registry[configuration_id]["configuration"], sort_keys=True, separators=(",", ":")
                ),
            }
            if configuration_id in records:
                row.update(records[configuration_id])
                row["status"] = "SUCCEEDED"
            else:
                row.update({
                    "status": "FAILED",
                    "failure_kind": failures[configuration_id]["kind"],
                    "message": failures[configuration_id]["message"],
                })
            writer.writerow(row)


def hash_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_sources(root):
    digest = hashlib.sha256()
    files = []
    for relative in ("include", "src", "exps/environments"):
        files.extend(path for path in (root / relative).rglob("*") if path.is_file())
    for path in sorted(files):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def displayed(command, root):
    result = []
    for item in command:
        path = Path(item)
        if path.is_absolute():
            try:
                item = path.relative_to(root).as_posix()
            except ValueError:
                pass
        result.append(str(item))
    return result


def compiler_version(compiler):
    process = subprocess.run(
        [compiler, "--version"], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, check=False,
    )
    return process.stdout.splitlines()[0] if process.returncode == 0 and process.stdout else "unavailable"


def options():
    parser = argparse.ArgumentParser()
    parser.add_argument("environment", choices=ENVIRONMENTS)
    parser.add_argument("algorithm", choices=ALGORITHMS)
    parser.add_argument("stopping", choices=STOPPING_CRITERIA)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "c++"))
    return parser.parse_args()


def main():
    arguments = options()
    experiment_root = Path(__file__).resolve().parent
    root = experiment_root.parent
    configuration_path = experiment_root / "experiments_configurations.json"
    job = "_".join((arguments.environment, arguments.algorithm, arguments.stopping)).lower()
    output = arguments.output or root / "build" / "experiments" / job
    output = output if output.is_absolute() else root / output
    output = output.resolve()
    if output == experiment_root or experiment_root in output.parents:
        raise ValueError("output must be outside exps")
    output.mkdir(parents=True, exist_ok=True)

    configuration = object_value(
        json.loads(configuration_path.read_text(encoding="utf-8")), "configuration"
    )
    template_path = experiment_root / "experiment.cpp.in"
    source, registry, environment = render_source(
        configuration, template_path.read_text(encoding="utf-8"),
        arguments.environment, arguments.algorithm, arguments.stopping,
    )
    source_path = output / "experiment.cpp"
    binary_path = output / "experiment_worker"
    policy_path = output / "best_policy.txt"
    source_path.write_text(source, encoding="utf-8")

    environment_sources = {
        "CARTPOLE": [experiment_root / "environments" / "CartPole.cpp"],
        "PENDULUM": [experiment_root / "environments" / "Pendulum.cpp"],
        "PENDULUM_STOCHASTIC": [
            experiment_root / "environments" / "Pendulum.cpp",
            experiment_root / "environments" / "PendulumStochastic.cpp",
        ],
    }[arguments.environment]
    include_root = root / "include"
    include_directories = [
        include_root,
        *(path for path in sorted(include_root.rglob("*")) if path.is_dir()),
        experiment_root / "environments",
    ]
    command = [
        arguments.cxx, "-std=c++20", "-O2",
        *(item for path in include_directories for item in ("-I", str(path))),
        str(source_path),
        *(str(path) for path in sorted((root / "src").rglob("*.cpp"))),
        *(str(path) for path in environment_sources), "-o", str(binary_path),
    ]
    subprocess.run(command, cwd=root, check=True)
    run_command = [str(binary_path), str(policy_path)]
    process = subprocess.run(
        run_command, cwd=root, text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(process.stderr.strip() or "experiment worker failed")

    records, failures, summary = parse_output(process.stdout, len(registry))
    results_path = output / "results.csv"
    best_path = output / "best_configuration.json"
    manifest_path = output / "manifest.json"
    write_results(results_path, registry, records, failures)
    best_id = summary["best_configuration_id"]
    write_json(best_path, {
        "environment": arguments.environment,
        "algorithm": arguments.algorithm,
        "stopping": arguments.stopping,
        "configuration_id": best_id,
        "hyperparameters": registry[best_id]["configuration"],
        "policy": configuration["policy"],
        "environment_configuration": environment,
        "validation": records[best_id],
        "test": {
            "mean": summary["test_mean"],
            "standard_deviation": summary["test_standard_deviation"],
        },
        "model_selection": {
            "attempted_configurations": summary["attempted_configurations"],
            "termination_reason": summary["termination_reason"],
        },
        "final_training": {
            "iterations": summary["final_training_iterations"],
            "evaluations": summary["final_training_evaluations"],
        },
        "policy_file": policy_path.name,
    })
    write_json(manifest_path, {
        "environment": arguments.environment,
        "algorithm": arguments.algorithm,
        "stopping": arguments.stopping,
        "registry_size": len(registry),
        "master_seed": configuration["master_seed"],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "compiler": {"command": arguments.cxx, "version": compiler_version(arguments.cxx)},
        "hashes": {
            "configuration": hash_file(configuration_path),
            "template": hash_file(template_path),
            "generator": hash_file(Path(__file__).resolve()),
            "generated_source": hash_file(source_path),
            "library_sources": hash_sources(root),
        },
        "compile_command": displayed(command, root),
        "run_command": displayed(run_command, root),
        "outputs": [results_path.name, best_path.name, policy_path.name, manifest_path.name],
    })
    print(output)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1)
