class LinePrinter:
    def __init__(self, character="-"):
        self.character = character

    def print_line(self, character=None, text='', number_of_repeat=20):
        if not character:
            character = self.character

        text_to_print = character * number_of_repeat + " " + str(text) + " " + character * number_of_repeat
        print(text_to_print)
        print("-" * len(text_to_print))
