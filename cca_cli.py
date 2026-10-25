import argparse

from cca.config import get_settings
from cca.orchestrator.runner import CCAOrchestrator
from cca.storage.persistence import write_json_file


def _reset_state(settings, reset_logs: bool) -> None:
    write_json_file(settings.paths.shared_pool_path, [])
    write_json_file(
        settings.paths.student_pool_path,
        {"known": [], "known_unknown": [], "conflict": [], "unknown_unknown": []},
    )
    if settings.paths.round_log_path.exists():
        settings.paths.round_log_path.unlink()
    if reset_logs:
        log_file = settings.paths.log_dir / "cca.log"
        if log_file.exists():
            log_file.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Cooperative Cognitive Agents (CCA) CLI")
    parser.add_argument("--topic", required=True, help="Topic for an interactive CCA run")
    parser.add_argument("--config", help="Optional path to config.yaml")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset knowledge pools and round logs before starting",
    )
    parser.add_argument(
        "--reset-logs",
        action="store_true",
        help="Also clear logs/cca.log when used with --reset",
    )
    parser.add_argument(
        "--manual-select",
        action="store_true",
        help="Enable manual selection from candidate questions each round",
    )
    parser.add_argument(
        "--retrieval",
        action="store_true",
        help="Enable web retrieval for Expert agent",
    )
    parser.add_argument(
        "--retriever",
        choices=["duckduckgo", "tavily", "deepseek"],
        default=None,
        help="Enable this retrieval backend; otherwise preserve the YAML configuration",
    )
    args = parser.parse_args()

    settings = get_settings(args.config) if args.config else get_settings()
    if args.retrieval or args.retriever:
        settings.orchestrator.enable_retrieval = True
    if args.retriever:
        settings.orchestrator.retriever_type = args.retriever
    if args.reset:
        _reset_state(settings, reset_logs=args.reset_logs)
    orchestrator = CCAOrchestrator(settings=settings)
    orchestrator.run(args.topic, manual_select=args.manual_select)


if __name__ == "__main__":
    main()
