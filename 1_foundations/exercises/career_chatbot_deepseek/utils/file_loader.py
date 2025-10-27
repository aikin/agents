
from pypdf import PdfReader


class FileLoader:

    @staticmethod
    def load_pdf(file_path: str) -> str:
        try:
            reader = PdfReader(file_path)
            text = "\n".join(
                page.extract_text()
                for page in reader.pages
                if page.extract_text()
            )
            return text
        except Exception as e:
            return f"Error loading PDF: {e}"

    @staticmethod
    def load_text(file_path: str, encoding: str = "utf-8") -> str:
        try:
            with open(file_path, encoding=encoding) as f:
                return f.read()
        except Exception as e:
            return f"Error loading text file: {e}"
