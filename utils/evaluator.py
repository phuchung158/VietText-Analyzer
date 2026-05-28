from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


class Evaluator:
    @staticmethod
    def _build_confusion_matrix_table(matrix, labels):
        string_labels = [str(label) for label in labels]
        row_header = "True\\Pred"
        row_width = max([len(row_header), *[len(label) for label in string_labels]]) + 2
        col_widths = [
            max(len(label), len(str(max(matrix[:, index], default=0)))) + 2
            for index, label in enumerate(string_labels)
        ]

        header = row_header.ljust(row_width) + "".join(
            label.rjust(col_widths[index]) for index, label in enumerate(string_labels)
        )
        rows = [header]

        for row_index, label in enumerate(string_labels):
            row = label.ljust(row_width) + "".join(
                str(matrix[row_index, col_index]).rjust(col_widths[col_index])
                for col_index in range(len(string_labels))
            )
            rows.append(row)

        return "\n".join(rows)

    @staticmethod
    def evaluate(y_true, y_pred, name="Model"):
        """Evaluate a classifier and print a readable report."""
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        labels = sorted({*y_true, *y_pred})
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=[str(label) for label in labels],
            zero_division=0,
        )
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        matrix_table = Evaluator._build_confusion_matrix_table(matrix, labels)

        print(f"\n{'=' * 50}")
        print(f"Results for {name}")
        print(f"{'=' * 50}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("\nConfusion Matrix:")
        print(matrix_table)
        print(f"\n{report}")

        return {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "labels": labels,
            "confusion_matrix": matrix.tolist(),
        }
