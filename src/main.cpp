#include <iostream>
#include "tokenizer.h"

int main() {
    std::cout << "Hello, World!" << std::endl;
    
    // Create a tokenizer instance
    SimpleTokenizer tokenizer;
    
    // Example usage
    std::string text = "Hello, world! This is a test of the SimpleTokenizer.";
    std::cout << "\nOriginal text: " << text << std::endl;
    
    // Encode text to tokens
    std::vector<int> tokens = tokenizer.encode(text);
    std::cout << "Encoded tokens (" << tokens.size() << " tokens): ";
    for (int token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
    
    // Decode tokens back to text
    std::string decoded = tokenizer.decode(tokens);
    std::cout << "Decoded text: " << decoded << std::endl;
    
    return 0;
}
