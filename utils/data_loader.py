import os
import xml.etree.ElementTree as ET
import zipfile


class DataLoader:
    XLSX_NAMESPACE = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    @staticmethod
    def load_training_excel(path):
        """Load one Excel split from the project's dataset directory."""
        rows = DataLoader._load_xlsx_rows(path)
        if len(rows) < 2:
            raise ValueError(f"Dataset {path} does not contain enough rows.")

        header_idx = DataLoader._detect_header_index(rows)
        header = [DataLoader._normalize_header_name(value) for value in rows[header_idx]]
        text_col, sentiment_col, topic_col = DataLoader._resolve_columns(header)

        records = []
        for row in rows[header_idx + 1 :]:
            padded = row + [""] * (len(header) - len(row))
            record = dict(zip(header, padded[: len(header)]))

            sentence = str(record.get(text_col, "")).strip()
            sentiment = str(record.get(sentiment_col, "")).strip()
            topic = str(record.get(topic_col, "")).strip()

            if not sentence or not sentiment or not topic:
                continue

            records.append(
                {
                    "sentence": sentence,
                    "sentiment": sentiment,
                    "topic": topic,
                }
            )

        if not records:
            raise ValueError(f"Dataset {path} does not contain valid training rows.")

        return records

    @staticmethod
    def load_project_splits(dataset_dir="dataset"):
        train_path = os.path.join(dataset_dir, "train.xlsx")
        valid_path = os.path.join(dataset_dir, "validation.xlsx")
        return (
            DataLoader.load_training_excel(train_path),
            DataLoader.load_training_excel(valid_path),
        )

    @staticmethod
    def _load_xlsx_rows(path):
        with zipfile.ZipFile(path) as archive:
            shared_strings = []
            if "xl/sharedStrings.xml" in archive.namelist():
                root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
                for string_item in root.findall("a:si", DataLoader.XLSX_NAMESPACE):
                    parts = [
                        node.text or ""
                        for node in string_item.iterfind(".//a:t", DataLoader.XLSX_NAMESPACE)
                    ]
                    shared_strings.append("".join(parts))

            sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
            rows = []
            for row in sheet.findall(".//a:row", DataLoader.XLSX_NAMESPACE):
                values = []
                for cell in row.findall("a:c", DataLoader.XLSX_NAMESPACE):
                    value_node = cell.find("a:v", DataLoader.XLSX_NAMESPACE)
                    if value_node is None:
                        values.append("")
                        continue

                    if cell.attrib.get("t") == "s":
                        values.append(shared_strings[int(value_node.text)])
                    else:
                        values.append(value_node.text)
                rows.append(values)
            return rows

    @staticmethod
    def _detect_header_index(rows):
        for index, row in enumerate(rows[:5]):
            normalized = [DataLoader._normalize_header_name(value) for value in row]
            if {"sentence", "sentiment", "topic"}.issubset(set(normalized)):
                return index
        raise ValueError("Could not detect dataset header row.")

    @staticmethod
    def _normalize_header_name(value):
        return str(value).strip().lower().replace(" ", "_")

    @staticmethod
    def _resolve_columns(header):
        aliases = {
            "sentence": {"sentence", "text", "content"},
            "sentiment": {"sentiment", "label", "emotion"},
            "topic": {"topic", "category", "aspect"},
        }

        resolved = {}
        for target, options in aliases.items():
            for column in header:
                if column in options:
                    resolved[target] = column
                    break

        missing = [key for key in ("sentence", "sentiment", "topic") if key not in resolved]
        if missing:
            raise ValueError(f"Missing required columns in dataset: {', '.join(missing)}")

        return resolved["sentence"], resolved["sentiment"], resolved["topic"]
