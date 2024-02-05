class Chatbot:
    def __init__(self):
        self.current_node = self.get_initial_node()
        self.products = {
            "Windows": {
                "$500 - $1000": {
                    "Specific brand": {
                        "Dell": ["Dell Laptop 1", "Dell Laptop 2"],
                        "HP": ["HP Laptop 1", "HP Laptop 2"],
                        "Lenovo": ["Lenovo Laptop 1", "Lenovo Laptop 2"],
                    },
                    "Any brand": ["Laptop A", "Laptop B"],
                },
                "$1000 - $1500": {
                    "Specific brand": {
                        "Dell": ["Dell Laptop 3", "Dell Laptop 4"],
                        "HP": ["HP Laptop 3", "HP Laptop 4"],
                        "Lenovo": ["Lenovo Laptop 3", "Lenovo Laptop 4"],
                    },
                    "Any brand": ["Laptop C", "Laptop D"],
                },
            },
            # Add more options as needed
        }

    def get_initial_node(self):
        return {
            "message": "Hi! I'm your virtual shopping assistant. How can I help you today?",
            "options": ["I'm looking for a new laptop."],
        }

    def process_user_input(self, user_input):
        if user_input not in self.current_node["options"]:
            return "I'm sorry, I don't understand your request."

        if user_input in self.current_node["options"]:
            if user_input == "I'm looking for a new laptop.":
                self.current_node = {
                    "message": "Great! Let me help you find the perfect laptop. Do you have a preferred operating system?",
                    "options": ["Windows", "macOS", "Linux"],
                }
            else:
                self.current_node = {"message": "I'm sorry, I don't understand your request."}
        return self.current_node["message"]

    def recommend_products(self, user_input):
        if user_input in self.products:
            self.current_node = self.products[user_input]
            return f"Super! What's your budget range?\n{', '.join(self.current_node.keys())}"
        else:
            return "I'm sorry, I don't understand your request."

# Create an instance of the chatbot
chatbot = Chatbot()

# Simulate a conversation with the chatbot
while True:
    user_input = input(chatbot.current_node["message"] + " ")
    if user_input.lower() == "exit":
        print("Goodbye!")
        break
    response = chatbot.process_user_input(user_input)
