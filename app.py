from __future__ import annotations

import matplotlib.pyplot as plt
import gradio as gr

from advanced_extraction import ExtractionConfig, list_strategies, run_extraction_simulation


def _plot_history(history):
    rounds = [row["queries"] for row in history]
    fidelity = [row["fidelity"] for row in history]
    kl_values = [row["kl_teacher_student"] for row in history]
    task_accuracy = [row["student_task_accuracy"] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(rounds, fidelity, marker="o", label="Fidelity")
    axes[0].plot(rounds, task_accuracy, marker="s", label="Task accuracy")
    axes[0].set_xlabel("Teacher queries")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0.0, 1.02)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(rounds, kl_values, marker="o", color="#9a3412")
    axes[1].set_xlabel("Teacher queries")
    axes[1].set_ylabel("KL(teacher || student)")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def _table(history):
    return [
        [
            row["round"],
            row["queries"],
            row["train_size"],
            row["pool_left"],
            round(row["fidelity"], 4),
            round(row["kl_teacher_student"], 4),
            round(row["ece"], 4),
            round(row["student_task_accuracy"], 4),
        ]
        for row in history
    ]


def run_demo(
    strategy: str,
    query_budget: int,
    initial_size: int,
    step_size: int,
    candidate_size: int,
    n_samples: int,
    random_state: int,
):
    initial_size = max(int(initial_size), 3)
    query_budget = max(int(query_budget), initial_size)
    config = ExtractionConfig(
        strategy=strategy,
        n_samples=int(n_samples),
        query_budget=int(query_budget),
        initial_size=initial_size,
        step_size=int(step_size),
        candidate_size=int(candidate_size),
        random_state=int(random_state),
    )
    result = run_extraction_simulation(config)
    summary = {
        "strategy": result.config["strategy"],
        "queries": result.query_count,
        "final_fidelity": round(result.final_fidelity, 4),
        "final_kl": round(result.final_kl, 4),
        "final_ece": round(result.final_ece, 4),
        "teacher_task_accuracy": round(result.teacher_task_accuracy, 4),
        "student_task_accuracy": round(result.student_task_accuracy, 4),
    }
    return _plot_history(result.history), _table(result.history), summary


with gr.Blocks(title="Advanced Adversarial Model Extraction Lab") as demo:
    gr.Markdown("# Advanced Adversarial Model Extraction Lab")
    with gr.Row():
        with gr.Column(scale=1, min_width=280):
            strategy = gr.Dropdown(
                choices=list_strategies(),
                value="hybrid",
                label="Query strategy",
            )
            query_budget = gr.Slider(60, 640, value=320, step=20, label="Query budget")
            initial_size = gr.Slider(6, 120, value=36, step=6, label="Initial queries")
            step_size = gr.Slider(8, 96, value=32, step=8, label="Round size")
            candidate_size = gr.Slider(80, 800, value=420, step=20, label="Candidate pool")
            n_samples = gr.Slider(900, 5000, value=2400, step=100, label="Synthetic samples")
            random_state = gr.Number(value=42, precision=0, label="Seed")
            run_button = gr.Button("Run simulation", variant="primary")
        with gr.Column(scale=2):
            plot = gr.Plot(label="Extraction trajectory")
            summary = gr.JSON(label="Final metrics")
    table = gr.Dataframe(
        headers=[
            "round",
            "queries",
            "train_size",
            "pool_left",
            "fidelity",
            "kl",
            "ece",
            "student_task_accuracy",
        ],
        datatype=["number"] * 8,
        label="Round metrics",
    )

    run_button.click(
        fn=run_demo,
        inputs=[strategy, query_budget, initial_size, step_size, candidate_size, n_samples, random_state],
        outputs=[plot, table, summary],
    )


if __name__ == "__main__":
    demo.launch()
