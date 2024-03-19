import asyncio
import os
from typing import List

import pandas as pd
from evaluate import load
from llama_index.core.evaluation import (CorrectnessEvaluator,
                                         SemanticSimilarityEvaluator)


class Evaluator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key
        self.bert_evaluator = load("bertscore")
        self.cosine_evaluator = SemanticSimilarityEvaluator()
        self.correctness_evaluator = CorrectnessEvaluator()

    def evaluate(self, questions: List[str], answers: List[str], ground_truths: List[str]) -> pd.DataFrame:
        """Evaluate scores averaged across all questions, answers, and ground truths."""
        results_each = self.evaluate_each(questions, answers, ground_truths)
        results = pd.DataFrame(
            {
                "metric": ["f1", "cosine", "correctness"],
                "score": [
                    results_each["f1"].mean(),
                    results_each["cosine"].mean(),
                    results_each["correctness"].mean(),
                ]
            }
        )
        return results

    def evaluate_each(self, questions: List[str], answers: List[str], ground_truths: List[str]) -> pd.DataFrame:
        """Evaluate scores for each question, answer, and ground truth."""
        f1 = self.f1_scores(answers, ground_truths)
        cosine = self.cosine_scores(answers, ground_truths)
        correctness = self.correctness_scores(questions, answers, ground_truths)
        results = pd.DataFrame(
            {
                "question": questions,
                "answer": answers,
                "ground_truth": ground_truths,
                "f1": f1,
                "cosine": cosine,
                "correctness": correctness,
            }
        )
        return results
    
    def evaluate_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run evaluation based on a DataFrame with columns: question, answer, ground_truth."""
        questions = df["question"].tolist()
        answers = df["answer"].tolist()
        ground_truths = df["ground_truth"].tolist()
        return self.evaluate(questions, answers, ground_truths)
    
    def evaluate_each_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run evaluation based on a DataFrame with columns: question, answer, ground_truth."""
        questions = df["question"].tolist()
        answers = df["answer"].tolist()
        ground_truths = df["ground_truth"].tolist()
        return self.evaluate_each(questions, answers, ground_truths)

    def evaluate_from_csv(self, file_path: str) -> pd.DataFrame:
        """Run evaluation based on a CSV file with columns: question, answer, ground_truth."""
        df = pd.read_csv(file_path)
        return self.evaluate_from_df(df)

    def evaluate_each_from_csv(self, file_path: str) -> pd.DataFrame:
        """Run evaluation based on a CSV file with columns: question, answer, ground_truth."""
        df = pd.read_csv(file_path)
        return self.evaluate_each_from_df(df)

    def f1_scores(self, answers: List[str], ground_truths: List[str]):
        """Calculate BERT's embedding-based similarity."""
        scores = self.bert_evaluator.compute(
            predictions=answers, references=ground_truths, lang="en"
        )
        return scores["f1"]

    def cosine_scores(self, answers: List[str], ground_truths: List[str]):
        """Calculate Ada's embedding-based similarity."""
        return asyncio.run(self.acosine_scores(answers, ground_truths))

    def correctness_scores(
        self, questions: List[str], answers: List[str], ground_truths: List[str]
    ):
        """Calculate correctness scores."""
        return asyncio.run(self.acorrectness_scores(questions, answers, ground_truths))

    async def acosine_scores(self, answers: List[str], ground_truths: List[str]):
        results = await asyncio.gather(
            *[
                self.cosine_evaluator.aevaluate(response=answer, reference=ground_truth)
                for answer, ground_truth in zip(answers, ground_truths)
            ]
        )
        scores = [result.score for result in results]
        return scores

    async def acorrectness_scores(
        self, questions: List[str], answers: List[str], ground_truths: List[str]
    ):
        results = await asyncio.gather(
            *[
                self.correctness_evaluator.aevaluate(
                    query=question, response=answer, reference=ground_truth
                )
                for question, answer, ground_truth in zip(questions, answers, ground_truths)
            ]
        )
        scores = [result.score for result in results]
        return scores


if __name__ == "__main__":
    questions = [
        "What is the capital of Vietnam?",
        "What is the capital of Vietnam?",
        "What is the capital of Vietnam?",
    ]
    answers = [
        "The capital of Vietnam is Ho Chi Minh City.",
        "The capital of Vietnam is Ho Chi Minh City.",
        "The capital of Vietnam is Ho Chi Minh City.",
    ]
    ground_truths = [
        "The capital of Vietnam is Ho Chi Minh City.",
        "The capital of Vietnam is Bangkok.",
        "I had lunch at Pho Ha in Saigon.",
    ]
    # save into csv file
    df = pd.DataFrame(
        {
            "question": questions,
            "answer": answers,
            "ground_truth": ground_truths,
        }
    )
    df.to_csv("example.csv", index=False)

    evaluator = Evaluator(api_key="")
    results = evaluator.evaluate(questions, answers, ground_truths)
    print(results)
    results = evaluator.evaluate_each(questions, answers, ground_truths)
    print(results)
    results = evaluator.evaluate_from_df(df)
    print(results)
    results = evaluator.evaluate_each_from_df(df)
    print(results)
    results = evaluator.evaluate_from_csv("example.csv")
    print(results)
    results = evaluator.evaluate_each_from_csv("example.csv")
    print(results)
