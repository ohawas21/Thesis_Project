class StyleHTML:
    @staticmethod
    def apply(format: str, string: str):
        return f"<{format}>{string}</{format}>"

    @staticmethod
    def bold(string: str):
        return StyleHTML.apply("b", string)

    @staticmethod
    def strong(string: str):
        return StyleHTML.apply("strong", string)

    @staticmethod
    def italic(string: str):
        return StyleHTML.apply("i", string)

    @staticmethod
    def h1(string: str):
        return StyleHTML.apply("h1", string)

    @staticmethod
    def h2(string: str):
        return StyleHTML.apply("h2", string)

    @staticmethod
    def paragraph(string: str):
        return StyleHTML.apply("p", string)
