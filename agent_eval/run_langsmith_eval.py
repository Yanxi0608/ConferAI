from __future__ import annotations

from dotenv import load_dotenv
from langsmith import Client

from agent_testrun import writer_critic_target
from agent_graders import faithfulness, knowledge_recall


def main() -> None:
    load_dotenv()

    client = Client()

    results = client.evaluate(
        writer_critic_target,
        data="Conference-Eval-Set",
        evaluators=[
            faithfulness,
            knowledge_recall,
        ],
        experiment_prefix="writer-critic-lightweight",
        max_concurrency=1,
    )

    print("Evaluation started / completed.")
    print(results)


if __name__ == "__main__":
    main()