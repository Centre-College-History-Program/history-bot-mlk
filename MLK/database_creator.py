
def save(question, keys, essential_keys):
    a

def main():
    cont = True
    while cont:
        question = input('What is your question?')
        keys = input('What are the keys for your question (Separate keys with a space)?')
        essential_keys = input('What are the essential keys for your question (Separate the keys with a space)?')
        
        keys = keys.split()
        essential_keys = essential_keys.split()
        save(question, keys, essential_keys)
        shouldContinue = input("Would you like to add another entry (y for yes, n for no)?")
        shouldContinue = shouldContinue.lower()
        cont = shouldContinue == "y"
        