# Orchestrates all verification checks and prints a PASS/FAIL summary with thresholds. Returns non‑zero exit code if any required check fails.
import argparse, json, sys, os
from .verify_featurizer_training import run as run_feat_trainable
from .verify_models import (
    check_shapes_and_masks,
    free_lunch_bc_overfit,
    value_head_regression,
    goal_head_regression,
)
from .verify_ppo import ppo_toy_improves
from .verify_search import search_winrate
from .verify_backbones import run as check_backbones
from .verify_cursor import cursor_alignment_checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")

    # BC / value / goal knobs
    ap.add_argument("--bc_train", type=int, default=256)
    ap.add_argument("--bc_eval", type=int, default=64)
    ap.add_argument("--bc_steps", type=int, default=400)
    ap.add_argument("--value_steps", type=int, default=300)
    ap.add_argument("--goal_steps", type=int, default=300)

    # PPO toy knobs
    ap.add_argument("--ppo_batch", type=int, default=512)
    ap.add_argument("--ppo_iters", type=int, default=4)
    ap.add_argument("--ppo_target_post", type=float, default=0.40)

    # Search check knobs
    ap.add_argument("--search_roots", type=int, default=64)
    ap.add_argument("--search_depth", type=int, default=2)
    ap.add_argument("--search_sims", type=int, default=64)
    ap.add_argument("--search_target_winrate", type=float, default=0.30)

    # Debug dump dir (optional)
    ap.add_argument(
        "--dump_debug",
        type=str,
        default=None,
        help="Directory to dump debug artifacts (masks/logits) for first failing item of a test",
    )

    args = ap.parse_args()
    overall_pass = True
    report = {}

    # Check shape & masking invariants
    r = check_shapes_and_masks(device=args.device)
    threshold = 1e-6
    # allow tiny numerical leakage
    model_shape_pass = r["illegal_mass_mean"] < threshold
    report["shapes_masks"] = {
        "result": r,
        "<_thresh": threshold,
        "verification_outcome": model_shape_pass,
    }
    overall_pass &= model_shape_pass

    # Check that featurizer gradients flow when training
    report["featurizer_trainable"] = run_feat_trainable(device=args.device)

    # Backbone checks
    backbones_shape_pass = check_backbones(device=args.device)
    report["backbones_extra"] = backbones_shape_pass
    overall_pass &= all(backbones_shape_pass)

    cursor_report = cursor_alignment_checks()
    report["cursor_alignment"] = cursor_report
    overall_pass &= cursor_report["overall_pass"]

    # Check if BC overfits quickly (which is expected)
    r = free_lunch_bc_overfit(
        device=args.device,
        n_train=args.bc_train,
        n_eval=args.bc_eval,
        steps=args.bc_steps,
    )
    threshold_act, threshold_line = 0.95, 0.90
    bc_overfit_pass = (
        r["action_acc"] >= threshold_act and r["line_acc"] >= threshold_line
    )
    report["bc_overfit"] = {
        "result": r,
        ">=_thresh": (threshold_act, threshold_line),
        "verification_outcome": bc_overfit_pass,
    }
    overall_pass &= bc_overfit_pass

    # Does value head overfit quickly?
    r = value_head_regression(
        device=args.device,
        n_train=args.bc_train,
        n_eval=args.bc_eval,
        steps=args.value_steps,
    )
    threshold = 0.05
    val_overfit_pass = r["value_rmse"] <= threshold
    report["value_regression"] = {
        "result": r,
        "<=_thresh": threshold,
        "verification_outcome": val_overfit_pass,
    }
    overall_pass &= val_overfit_pass

    # Does goal head overfit quickly?
    r = goal_head_regression(
        device=args.device,
        n_train=args.bc_train,
        n_eval=args.bc_eval,
        steps=args.goal_steps,
    )
    threshold = 0.01
    goal_overfit_pass = r["goal_mse"] <= threshold
    report["goal_regression"] = {
        "result": r,
        "<=_thresh": threshold,
        "verification_outcome": goal_overfit_pass,
    }
    overall_pass &= goal_overfit_pass

    # Is PPO working? (aka does training action and human action pred heads work)
    r = ppo_toy_improves(
        device=args.device,
        batch_size=args.ppo_batch,
        iters=args.ppo_iters,
        target_post=args.ppo_target_post,
    )
    ppo_fit_pass = r["passed"]
    report["ppo_toy"] = {
        "result": r,
        ">=_thresh": args.ppo_target_post,
        "verification_outcome": ppo_fit_pass,
    }
    overall_pass &= ppo_fit_pass

    # Search win-rate (PUCT vs greedy)
    dump_dir = args.dump_debug if args.dump_debug else None
    r = search_winrate(
        device=args.device,
        n_roots=args.search_roots,
        depth=args.search_depth,
        n_sims=args.search_sims,
        dump_debug_dir=dump_dir,
    )
    search_win_pass = r["win_rate"] >= args.search_target_winrate
    report["search_winrate"] = {
        "result": r,
        ">=_thresh": args.search_target_winrate,
        "verification_outcome": search_win_pass,
    }
    overall_pass &= search_win_pass

    # Summary
    summary = {"overall_pass": overall_pass, "results": report}

    # Emit locations of any debug artifacts
    if args.dump_debug:
        sd = report.get("search_winrate", {}).get("sample_debug")
        if sd:
            summary["debug_artifacts"] = {"search_first_nonwin": sd}
        else:
            summary["debug_artifacts"] = {"search_first_nonwin": None}

    print(json.dumps(summary, indent=2))
    sys.exit(0 if overall_pass else 2)


if __name__ == "__main__":
    main()
