
# This code initializes a simple chatbot using GPT-2 from Hugging Face's Transformers library. 
# It loads the pre-trained GPT-2 model and tokenizer, and then generates a response based on user input.



from transformers import GPT2LMHeadModel, GPT2Tokenizer

class RetirementChatbot:
    def __init__(self):
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    def generate_response(self, user_input):
        input_ids = self.tokenizer.encode(user_input, return_tensors="pt")
        output = self.model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
        response = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return response

# Example usage:
chatbot = RetirementChatbot()
user_input = "What is the best retirement strategy?"
response = chatbot.generate_response(user_input)
print(response)
