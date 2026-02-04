#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include "tokenizer.h"

// Test fixture for tokenizer tests
class TokenizerTest : public ::testing::Test {
protected:
    SimpleTokenizer tokenizer;
};

// Test encoding with expected results from README
TEST_F(TokenizerTest, EncodeMatchesReadmeExample) {
    std::string text = "Hello, world! This is a test of the SimpleTokenizer.";
    
    // Expected tokens from README
    std::vector<int> expected = {3306, 267, 1002, 256, 589, 533, 320, 1628, 539, 518, 19018, 32634, 23895, 269};
    
    std::vector<int> tokens = tokenizer.encode(text);
    
    // Check if the tokens match
    ASSERT_EQ(tokens.size(), expected.size()) << "Number of tokens doesn't match";
    
    for (size_t i = 0; i < tokens.size(); ++i) {
        EXPECT_EQ(tokens[i], expected[i]) << "Token mismatch at position " << i;
    }
    
    // Print tokens for verification
    std::cout << "Encoded tokens (" << tokens.size() << " tokens): ";
    for (const auto& token : tokens) {
        std::cout << token << " ";
    }
    std::cout << std::endl;
}

// Test decoding
TEST_F(TokenizerTest, DecodeMatchesExpected) {
    // Using the same tokens from README
    std::vector<int> tokens = {3306, 267, 1002, 256, 589, 533, 320, 1628, 539, 518, 19018, 32634, 23895, 269};
    
    std::string decoded = tokenizer.decode(tokens);
    
    // The decoded text should contain the expected words (note: CLIP tokenizer may add spaces)
    std::string decoded_lower = decoded;
    std::transform(decoded_lower.begin(), decoded_lower.end(), decoded_lower.begin(), ::tolower);
    
    // Check that it contains the key words from the original
    EXPECT_NE(decoded_lower.find("hello"), std::string::npos);
    EXPECT_NE(decoded_lower.find("world"), std::string::npos);
    EXPECT_NE(decoded_lower.find("test"), std::string::npos);
    EXPECT_NE(decoded_lower.find("simpletokenizer"), std::string::npos);
    
    std::cout << "Decoded text: " << decoded << std::endl;
}

// Test encode-decode round trip
TEST_F(TokenizerTest, EncodeDecodeRoundTrip) {
    std::string original = "CLIP is a multimodal vision and language model.";
    
    // Encode
    std::vector<int> tokens = tokenizer.encode(original);
    
    // Decode
    std::string decoded = tokenizer.decode(tokens);
    
    // Normalize both strings for comparison (lowercase and remove extra spaces)
    std::string original_lower = original;
    std::string decoded_lower = decoded;
    std::transform(original_lower.begin(), original_lower.end(), original_lower.begin(), ::tolower);
    std::transform(decoded_lower.begin(), decoded_lower.end(), decoded_lower.begin(), ::tolower);
    
    // Remove all whitespace for comparison (CLIP tokenizer may add spaces around punctuation)
    auto remove_spaces = [](std::string& s) {
        s.erase(std::remove_if(s.begin(), s.end(), ::isspace), s.end());
    };
    remove_spaces(original_lower);
    remove_spaces(decoded_lower);
    
    EXPECT_EQ(original_lower, decoded_lower);
    
    std::cout << "Original: " << original << std::endl;
    std::cout << "Decoded:  " << decoded << std::endl;
}

// Test empty string
TEST_F(TokenizerTest, EncodeEmptyString) {
    std::string text = "";
    std::vector<int> tokens = tokenizer.encode(text);
    
    EXPECT_TRUE(tokens.empty() || tokens.size() <= 2) << "Empty string should produce no or minimal tokens";
}

// Test single word
TEST_F(TokenizerTest, EncodeSingleWord) {
    std::string text = "Hello";
    std::vector<int> tokens = tokenizer.encode(text);
    
    EXPECT_FALSE(tokens.empty()) << "Single word should produce at least one token";
    
    // Verify round trip
    std::string decoded = tokenizer.decode(tokens);
    std::string decoded_lower = decoded;
    std::transform(decoded_lower.begin(), decoded_lower.end(), decoded_lower.begin(), ::tolower);
    
    // Trim trailing whitespace
    decoded_lower.erase(decoded_lower.find_last_not_of(" \t\n\r") + 1);
    
    EXPECT_EQ(decoded_lower, "hello");
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
