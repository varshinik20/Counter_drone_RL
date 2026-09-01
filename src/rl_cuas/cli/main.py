import argparse
from pathlib import Path
from ..config import load_config, ScenarioConfig
from ..train import train_model
from ..evaluate import evaluate_policy
from ..compare import compare_policies


def main():
    parser = argparse.ArgumentParser(prog="rl-cuas-cli")
    sub = parser.add_subparsers(dest="command", required=True)

    train_p = sub.add_parser("train")
    train_p.add_argument("--config", required=True)

    eval_p = sub.add_parser("evaluate")
    eval_p.add_argument("--policy", choices=["deeprl", "classic", "random"], required=True)
    eval_p.add_argument("--n_episodes", type=int, default=1)
    eval_p.add_argument("--seed", type=int, default=42)
    eval_p.add_argument("--model_path", default="results/models/commander_policy")

    cmp_p = sub.add_parser("compare")
    cmp_p.add_argument("--n_episodes", type=int, default=100)
    cmp_p.add_argument("--seeds", nargs="+", type=int, default=[10, 20, 30, 42, 50])
    cmp_p.add_argument("--model_path", default="results/models/commander_policy")
    cmp_p.add_argument("--results_dir", default="results")

    args = parser.parse_args()

    if args.command == "train":
        cfg = load_config(args.config)
        model_zip = train_model(cfg)
        print(f"Saved model to: {model_zip}")

    elif args.command == "evaluate":
        df = evaluate_policy(args.policy, args.model_path, args.n_episodes, args.seed, ScenarioConfig())
        print(df)
        out = Path("results") / f"evaluate_{args.policy}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"Saved: {out}")

    elif args.command == "compare":
        _, summary = compare_policies(args.model_path, args.n_episodes, args.seeds, ScenarioConfig(), args.results_dir)
        print(summary)
        print(f"Saved compare outputs in: {args.results_dir}")


if __name__ == "__main__":
    main()